"""Posed membrane patch. Distances are metres, time is seconds."""
import numpy as np
from .fracture import (FabricResponse, aperture, chart_delta, linked_runs, loose_stock,
                       open_area, released_shape, rim, rim_slice, seam_bend, shared_wales, strain)
from .knit_patch import PinchPatch, surface_normals
from .tear_surface import fair_rim, limit_fairing, limit_strain, strain_frame, support_strips
from .yarn_path import rolled_edge
from . import contact_index, seam_overlap

WARNING_DURATION = .115
DRY_WARNING_DURATION = .245
WARNING_MAX_STEP = .065


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def loaded_run(loops, seed, load_axis, tensile, minimum):
    """Follow intact neighbours on the prospective tear course or wale."""
    cells=[seed]
    for direction in (2*(1-load_axis),2*(1-load_axis)+1):
        index=seed
        for step in range(16):
            neighbour=loops.adjacent[index,direction]
            # A run follows decreasing load away from the seed. It ends at
            # the first unladen or disconnected loop, even inside the cap.
            floor = minimum * max(.38, 1. - .061*step)
            if neighbour==index or not loops.intact[neighbour] or tensile[neighbour]<floor:
                break
            cells.append(neighbour)
            index=neighbour
    return cells


def knit_chart(points, reference=None):
    # Cylindrical courses around the unposed leg. Preserve the chart when tearing.
    reference = points if reference is None else reference
    reference_z = reference[:, 2]
    z = points[:, 2]
    levels = np.linspace(reference_z.min(), reference_z.max(), 50)
    centers = []
    for level in levels:
        band = reference[abs(reference_z-level) < max(np.ptp(reference_z)/45, .004), :2]
        centers.append((band.min(0)+band.max(0))*.5 if len(band) else [np.nan]*2)
    centers = np.asarray(centers)
    ok = np.isfinite(centers[:, 0])
    xy = points[:, :2] - np.column_stack([np.interp(z, levels[ok], centers[ok, k]) for k in range(2)])
    return np.column_stack((np.arctan2(xy[:, 1], xy[:, 0])/(2*np.pi), z/.0020))


def unique_edges(edges, vertex_count, return_inverse=False):
    # Sorting one integer per edge preserves lexicographic (a, b) order.
    a,b=edges.T
    codes=np.minimum(a,b).astype(np.int64)*vertex_count+np.maximum(a,b)
    if return_inverse:
        keys,inverse=np.unique(codes,return_inverse=True)
    else:keys=np.unique(codes)
    a,b=np.divmod(keys,vertex_count)
    pairs=np.column_stack((a,b)).astype(np.int32)
    return (pairs,inverse) if return_inverse else pairs


def faces_meeting_box(points, faces, center, radius):
    # A long edge crossing the box survives even if both endpoints are outside.
    outside = np.zeros(len(points), dtype=np.uint8)
    for axis in range(3):
        outside |= (points[:, axis] < center[axis] - radius).astype(np.uint8) << (2 * axis)
        outside |= (points[:, axis] > center[axis] + radius).astype(np.uint8) << (2 * axis + 1)
    return (outside[faces[:, 0]] & outside[faces[:, 1]] & outside[faces[:, 2]]) == 0


def subdivide_edges(faces, mid):
    # Each template preserves winding. Indices 3, 4, 5 denote AB, BC, CA.
    bits=(mid>=0)@np.array([1,2,4])
    patterns=(((0,1,2),),((0,3,2),(3,1,2)),((1,4,0),(4,2,0)),
              ((3,1,4),(0,3,4),(0,4,2)),((2,5,1),(5,0,1)),
              ((5,0,3),(3,1,2),(3,2,5)),((4,2,5),(0,1,4),(0,4,5)),
              ((0,3,5),(3,1,4),(5,4,2),(3,4,5)))
    table=np.column_stack((faces,mid))
    children=[]
    for mask,patch in enumerate(patterns):
        matching=table[bits==mask]
        children.extend(matching[:,tri] for tri in patch)
    return np.concatenate(children)


def refine_surface(p, f, uv, pos, opening, center, spacing=.0030, interpolate=None, protected=None):
    # Share edge midpoints across neighbouring faces, including faces at the
    # refinement transition. Original landmarks and material pitch stay fixed.
    for _ in range(2):
        near=faces_meeting_box(p,f,center,.057)
        e=f[near][:,[[0,1],[1,2],[2,0]]]
        keys,inv=unique_edges(e.reshape(-1,2),len(p),return_inverse=True)
        within=np.linalg.norm((p[keys[:,0]]+p[keys[:,1]])*.5-center,axis=1)<.057
        long=within&(np.linalg.norm(p[keys[:,0]]-p[keys[:,1]],axis=1)>spacing)
        if protected is not None:
            # Bank pairs already have quarter-pitch knots. An independent
            # midpoint could reopen a pair that is still welded past its tip.
            long &= ~np.isin(keys,protected).any(axis=1)
        if not long.any():break
        a,b=keys[long].T
        mids=np.full(len(keys),-1,dtype=np.int32)
        mids[long]=np.arange(len(p),len(p)+len(a))
        uv_delta=uv[b]-uv[a]
        uv_delta[:,0]-=np.round(uv_delta[:,0])
        p=np.vstack((p,(p[a]+p[b])*.5))
        if interpolate is not None:
            interpolate(a,b,.5)
        pos=np.vstack((pos,(pos[a]+pos[b])*.5))
        opening=np.vstack((opening,(opening[a]+opening[b])*.5))
        uv=np.vstack((uv,uv[a]+uv_delta*.5))
        mid=np.full(f.shape,-1,dtype=np.int32)
        mid[near]=mids[inv].reshape(-1,3)
        f=subdivide_edges(f,mid)
    return p,f.astype(np.int32,copy=False),uv,pos,opening


def grip_weights(rest,anchor,grip):
    offset=rest-anchor
    d=np.linalg.norm(offset,axis=1)
    weight=np.exp(-.5*(d/.015)**2)
    taper=np.clip((.057-d)/.012,0.,1.)
    taper=taper*taper*(3.-2.*taper)
    weight*=taper
    if grip is not None:weight[grip]=1.
    return offset,d*d,taper,np.flatnonzero(taper>0.).astype(np.int32),weight


class Membrane:
    def __init__(self, points, triangles, chart, response=None, *, dry=False):
        self.fitted_vertices=len(points)
        self.rest = points.copy()
        self.pos = points.copy()
        self.faces = triangles.copy()
        self.uv = chart.copy()
        self.opening = np.zeros_like(points)
        self.seating = np.zeros_like(points)
        self.grip = None
        self.cracks = []
        self.run_links = []
        self.bridge_supports = {}
        self.loose_yarn = None
        self.changed = False
        self.moving = False
        self.patch = None
        self.response = response or FabricResponse()
        self.fracture_events = []
        self.step_number = 0
        self.elapsed = 0.
        self.active_crack = None
        self.nearby_crack = None
        self.contact_normals = None
        self.rim_strips = None
        self.tear_support_faces = None
        self.open_rest = None
        self.strain_cache = None
        self.grab_surface_offset = None
        self.warn_before_break = False
        self.dry = dry
        self.warning = None
        self.warning_dirty = False
        self.skip_recovery_break = False
        self.previous_strain = None
        self.yarn_first_row=int(np.floor(chart[:,1].min()))
        rows=int(np.floor(chart[:,1].max()))-self.yarn_first_row+1
        self.yarn_ledger=np.zeros((rows,144))
        self.failed_cells=np.zeros_like(self.yarn_ledger,dtype=bool)

    def fixed_material(self):
        return (getattr(self,'pose',None) is None or getattr(self,'fixed_wet_pose',False)
                or (bool(self.cracks) and getattr(self,'wet_fabric_contact',False)))

    def pick(self, face, hit):
        self.release()
        self.skip_recovery_break = False
        ids = self.faces[face]
        sides = self.pos[ids[1:]] - self.pos[ids[0]]
        ab = np.linalg.lstsq(sides.T, hit-self.pos[ids[0]], rcond=None)[0]
        self.bary = np.r_[1-ab.sum(), ab]
        self.grip = ids.copy()
        self.anchor = self.bary @ self.rest[ids]
        uv = self.uv[ids]
        self.anchor_uv = uv[0] + self.bary @ (chart_delta(uv, uv[0])/[144, 1])
        self.grip_chart = chart_delta(uv[1:], uv[0])
        self.update_axes()
        self.target = hit.copy()
        self.normal = unit(np.cross(*sides))
        if self.dry and self.cracks and self.fixed_material():
            # A thin face alongside a cut can already be tilted by residual
            # tension. Use the neutral cloth normal for a new pull direction.
            if self.contact_normals is None or len(self.contact_normals) != len(self.rest):
                self.contact_normals = surface_normals(self.rest, self.faces)
            self.normal = unit(self.bary @ self.contact_normals[ids])
        # Compact influence keeps the opposing side of the leg in place.
        self.weights()
        self.bind_patch()
        self.previous_strain = None
        self.load_axis = 0
        self.growth_budget = 0.
        self.recoil_budget = 0.
        self.grip_transfer = 1.
        self.hold_loaded = False
        self.hold_work = 0.
        self.held_seconds = 0.
        self.yarn_pending = 0.
        self.active_crack = None
        self.nearby_crack = self.crack_at_pick()
        self.peak = 1.
        self.pulled = False
        self.raw_opening = 0.
        self.slip_work = 0.

    def crack_at_pick(self):
        nearest = None
        # Leave room between the pinch and the tip to tension the intervening
        # stitches. A pinch on the very lip can carry the tip along with it.
        distance = np.inf
        for crack in self.cracks:
            centerline = rim(crack,self.rest).mean(1)
            offsets = np.linalg.norm(centerline-self.anchor,axis=1)
            reach = .014 + (min(.009,.72*crack['width']) if self.dry else 0.)
            if len(offsets) and offsets.min()<min(distance,reach):
                nearest,distance = crack,float(offsets.min())
        return nearest

    def bind_patch(self):
        prepared=getattr(self,'prepared_patch',None)
        if prepared is not None:
            self.prepared_patch=None
            self.patch=prepared
            prepared.loops.ledger=self.yarn_ledger;prepared.loops.failed=self.failed_cells
            if prepared.skin_plane is None and hasattr(self,'skin_planes'):
                prepared.skin_plane=self.skin_planes(self.rest[prepared.ids],prepared.normal)
            prepared.update_stitches(self.pos)
            return
        faces = self.faces[np.max(self.weight[self.faces], axis=1) > .04]
        self.patch = PinchPatch(self, faces)
        self.patch.update_stitches(self.pos)

    def update_axes(self):
        points = self.rest[self.grip]
        self.axes = np.linalg.lstsq(self.grip_chart,points[1:]-points[0],rcond=None)[0]

    def weights(self):
        self.offset,self.distance_sq,self.drag_taper,self.drag_ids,self.weight=grip_weights(self.rest,self.anchor,self.grip)
        if self.dry and getattr(self,'pose',None) is None and not self.cracks:
            self.intact_base=self.rest+self.opening+self.seating

    def side_span(self, direction, lift, spread):
        """Separate a taut pinch's outward profile from in-plane yarn travel."""
        if lift<=.0025 or self.active_crack is not None or self.nearby_crack is not None:
            return 0., None
        wale = unit(self.axes[1] - (self.axes[1]@self.normal)*self.normal)
        lateral = 1. - abs(direction@wale)
        coupling = np.clip((lateral-.32)/.40, 0., 1.)
        departure = np.clip((lift-.0025)/.0075, 0., 1.)
        tension = coupling*departure
        tension = tension*tension*(3.-2.*tension)
        if tension==0.:
            return 0., None
        distance = abs(self.offset@wale)
        flank = np.maximum(0.,1.-distance/(.025+.17*lift))
        join = np.minimum(1.,flank/.18)
        tapered = flank*join*join*(3.-2.*join)
        round_profile = np.exp(-.5*(distance/spread)**2)
        # The center course keeps its original load path while the outward
        # component draws the upper and lower shoulders into linear flanks.
        return tension, np.minimum(1.,tapered/np.maximum(round_profile,.10))

    def grip_base(self):
        ids = self.grip
        if self.dry and self.cracks and self.fixed_material():
            return self.bary @ self.relaxed_tears()[ids]
        return self.bary@(self.rest[ids]+self.opening[ids]+self.seating[ids])

    def limit_target(self,target,*,base=None):
        if base is None:base=self.grip_base()
        delta=target-base
        self.pointer_travel=delta.copy()
        distance=np.linalg.norm(delta)
        # The local membrane represents a pinch patch. Pointer travel outside
        # that patch can take up tension, but cannot translate the whole sock.
        reach=np.clip(.048*np.sqrt(self.response.break_extension/.80),.028,.066)
        shoulder=.72*reach
        if distance>shoulder:
            width=reach-shoulder
            delta*= (shoulder-width*np.expm1(-(distance-shoulder)/width))/distance
        if self.dry and self.fixed_material():
            raw=target-base
            outward=delta@self.normal
            slip=delta-outward*self.normal
            travel=np.linalg.norm(slip)
            crack=self.active_crack or self.nearby_crack
            if crack is not None and travel>.014:
                reach=max(.019,.029-.55*crack['width'])
                slip*= (.014-(reach-.014)*np.expm1(- (travel-.014)/(reach-.014)))/travel
            if crack is not None:
                side=np.sign((self.anchor-crack['center'])@crack['across']) or 1.
                extension=max(0.,side*(raw@crack['across'])+.45*max(0.,raw@self.normal))
                # Extra pointer travel feeds yarn through the loaded front
                # after the remaining sheet has reached its elastic extension.
                if self.raw_opening is not None:
                    self.slip_work+=.43*max(0.,extension-self.raw_opening)
                self.raw_opening=extension
            # A pinched sheet can lift while sliding across the leg. Driving
            # its target into the skin and projecting just the nearby vertices
            # concentrated the entire displacement at the grip.
            delta=slip+max(outward,min(.008,.22*np.linalg.norm(slip)))*self.normal
        elif (self.dry and self.run_links and self.nearby_crack is not None
                and any(self.nearby_crack is link[0] or self.nearby_crack is link[1] for link in self.run_links)):
            rest_width = self.nearby_crack['width']
            if rest_width>.008:
                normal = (delta@self.normal)*self.normal
                tangent = delta-normal
                reach = np.clip(.025-.85*rest_width,.013,.021)
                travel = np.linalg.norm(tangent)
                if travel>.70*reach:
                    remainder = .30*reach
                    allowed = .70*reach - remainder*np.expm1(-(travel-.70*reach)/remainder)
                    delta = normal + tangent*(allowed/travel)
        if self.warning is not None:
            # Let the user ease off immediately. More travel can load the
            # remaining knit once the already loaded stitch has snapped.
            length = np.linalg.norm(delta)
            reach = self.warning['reach']
            excess = max(0., length-reach)
            shoulder = .0032 if self.dry else 0.
            allowed = reach - shoulder*np.expm1(-excess/shoulder) if shoulder else reach
            delta *= min(1., allowed/max(length,1e-12))
        return base+self.grip_transfer*delta

    def release(self):
        if getattr(self,'neutral_job',None) is not None:
            from .neutral_worker import collect
            collect(self,settle=True)
        self.cancel_warning()
        if getattr(self,'warned_lanes',None) is not None:
            from .warned_lane import cancel
            cancel(self)
        if self.grip is not None and self.active_crack is not None:
            self.active_crack.pop('charged_loops',None)
            self.active_crack.pop('wale_reserve',None)
            self.active_crack.pop('wale_travel',None)
        if (self.grip is not None and self.pulled and not self.active_crack
                and self.patch.skin_plane is not None):
            if getattr(self,'pose',None) is not None:
                self.pose.seat_released_patch(self)
            else:
                ids=self.patch.ids
                origin,normal=self.patch.skin_plane
                weight=np.clip((.040-self.patch.distance)/.014,0,1)
                weight=weight**2*(3-2*weight)
                # The bound garment may float above skin in its import pose.
                # Seat the released intact patch on the sampled skin, retaining its
                # textile reference coordinates and any existing tear aperture.
                offset=(origin-self.rest[ids])*weight[:,None]
                closer=np.sum((offset-self.seating[ids])*normal,axis=1)<0
                self.seating[ids[closer]]=offset[closer]
                self.open_rest = None
        self.grip = None

    def cancel_warning(self):
        if self.warning is None:
            return
        if getattr(self,'warned_lanes',None) is not None:
            from .warned_lane import cancel
            cancel(self)
        self.warning=None
        self.warning_dirty=True
        self.skip_recovery_break=True
        self.hold_loaded=False
        self.hold_work=0.

    def warning_ready(self, dt):
        return (self.warning is not None and
                self.warning['age']+np.clip(dt,0.,WARNING_MAX_STEP)>=self.warning['duration']-1.e-10)

    def warning_offset(self):
        age = self.warning['age']
        u = min(1., age/self.warning['duration'])
        envelope = u*u*(3.-2.*u)
        amount = envelope*(.00020+.00024*(.5-.5*np.cos(2*np.pi*7.*age)))
        sway = .16*amount*np.sin(2*np.pi*8.*age)
        return amount*self.normal+sway*unit(self.axes[self.warning['axis']])

    def loaded_stitches(self):
        warning = self.warning
        ids = np.asarray(warning['stitches'])
        cells = self.patch.loops.cells[ids]
        uv = (cells+.5)/[144,1]
        distance = abs(chart_delta(uv,uv[0])[:,1-warning['axis']])
        x = min(1.,max(warning['age']/warning['duration'],warning.get('extension',0.)))
        ease = x*x*(3.-2.*x)
        front = .8 + (distance.max()+.45)*ease
        reveal = np.clip((front-distance)/1.7,0.,1.)*(.35+.65*ease)
        reveal *= (1.-.25*distance/(distance.max()+1.))*np.where(distance%2 < .25,1.,.68)
        return ids, cells, reveal

    def interpolate_vertices(self, a, b, fraction):
        reference=getattr(self,'cut_reference',None)
        if reference is not None:
            extra=reference[a]+fraction[:,None,None]*(reference[b]-reference[a])
            self.cut_reference=np.concatenate((reference,extra))
        if getattr(self,'pose',None) is not None:
            self.pose.interpolate(a,b,fraction)
        if (observe := getattr(self, 'monitor_interpolate', None)) is not None:
            observe(a, b, fraction)
        fields=('rest','pos','opening','seating','uv')
        if getattr(self,'bank_weave',None) is not None:fields+=('bank_weave',)
        for prop in fields:
            array = getattr(self,prop)
            diff = array[b]-array[a]
            if prop=='uv':diff[:,0] -= np.round(diff[:,0])
            setattr(self,prop,np.vstack((array,array[a]+diff*fraction[:,None])))
        if 'bank_weave' in fields:
            self.bank_weave=self.bank_weave.astype('f',copy=False);self.bank_rest_owner=self.rest

    def append_material_samples(self, parents, weights):
        reference=getattr(self,'cut_reference',None)
        if reference is not None:
            self.cut_reference=np.concatenate((reference,np.einsum('nc,ncij->nij',weights,reference[parents])))
        if getattr(self,'pose',None) is not None:
            self.pose.interpolate_faces(parents,weights)
        observe=getattr(self,'monitor_interpolate_faces',None)
        if observe is not None:
            observe(parents,weights)
        fields=('rest','pos','opening','seating','uv')
        if getattr(self,'bank_weave',None) is not None:fields+=('bank_weave',)
        for name in fields:
            values=getattr(self,name)
            triangle=values[parents]
            if name=='uv':
                triangle[:,:,0]-=np.round(triangle[:,:,0]-triangle[:,:1,0])
            new=np.einsum('ni,nij->nj',weights,triangle)
            setattr(self,name,np.vstack((values,new)))
        if 'bank_weave' in fields:
            self.bank_weave=self.bank_weave.astype('f',copy=False);self.bank_rest_owner=self.rest

    def split(self, center, uv, load_axis, across, along):
        # Rest vertices on the two sides of a broken stitch have identical factory
        # coordinates. Carry the already paid-out neutral sheet through the SAME
        # interpolation as the new cut so contact cannot weld old holes by position.
        if not self.cracks or getattr(self,'pose',None) is not None:
            return self.split_material(center,uv,load_axis,across,along)
        if getattr(self,'neutral_job',None) is not None:
            from .neutral_worker import collect
            collect(self,settle=True)
        neutral=self.relaxed_tears()
        history=getattr(self,'neutral_history',None)
        targets=history[1:3] if history is not None and history[0] is self.faces else (neutral,neutral)
        self.cut_reference=np.stack(targets,axis=1)
        try:
            crack=self.split_material(center,uv,load_axis,across,along)
            if crack is not None:
                previous=self.cut_reference
                self.neutral_history=(self.faces,previous[:,0],previous[:,1])
            return crack
        finally:
            del self.cut_reference

    def split_material(self, center, uv, load_axis, across, along):
        # Cut the connected chain of triangles crossed by a plane. Share the
        # end vertices and duplicate the interior banks. The prospective seam
        # stays kinematically welded ahead of the advancing stitch fronts.
        delta = chart_delta(self.uv, uv)
        run_axis = 1-load_axis
        phase = float(uv@[84.4,1.13]) if self.dry else None
        if self.dry:
            from .tear_strip import cut_strip
            strip=cut_strip(self,center,uv,load_axis,phase)
            if strip is not None:
                rails,stations=strip
                self.refresh_dry_support()
                return self.register_crack(center,uv,load_axis,across,along,phase,rails,stations,[])
            # Overlapping UV charts cannot share a CDT cavity. Split the
            # connected crossed source triangles and keep both chart banks.
        signed = delta[:,load_axis].copy()
        if phase is not None:
            signed -= seam_bend(delta[:,run_axis],phase)
        signed[np.abs(signed)<1e-9] = 1e-9
        f = self.faces
        positive=signed>0
        crosses=(positive[f[:,0]]!=positive[f[:,1]])|(positive[f[:,0]]!=positive[f[:,2]])
        candidates=np.flatnonzero(crosses)
        centers=self.rest[f[candidates]].mean(1)
        span=np.abs(delta[f[candidates],run_axis].mean(1))
        radius, length = (.041,16.5) if self.dry else (.026,10.5)
        candidates=candidates[(np.linalg.norm(centers-center,axis=1)<radius)&(span<length)]
        if self.cracks:
            previous=np.concatenate([c['rim_ids'].ravel() for c in self.cracks])
            candidates=candidates[~np.isin(f[candidates],previous).any(1)]
        crossing, adjacent = {}, {}
        for i in candidates:
            tri = f[i]
            cuts = [tuple(sorted((int(a),int(b)))) for a,b in zip(tri,np.roll(tri,-1)) if signed[a]*signed[b]<0]
            crossing[i] = cuts
            for key in cuts: adjacent.setdefault(key,[]).append(i)
        if not crossing:return None
        seed = min(crossing,key=lambda i:np.linalg.norm(self.rest[f[i]].mean(0)-center))
        connected, stack = set(), [seed]
        while stack:
            i = stack.pop()
            if i in connected:continue
            connected.add(i)
            stack.extend(j for k in crossing[i] for j in adjacent[k] if j not in connected)
        keys = sorted({k for i in connected for k in crossing[i]})
        if len(keys)<4:return None
        counts = [2 if len(adjacent[k])==2 else 1 for k in keys]
        if counts.count(1)!=2:return None
        a,b=np.asarray(keys).T
        fraction=signed[a]/(signed[a]-signed[b])
        cut_axis=delta[a,run_axis]+fraction*(delta[b,run_axis]-delta[a,run_axis])
        tip_axis=cut_axis[np.asarray(counts)==1]
        if tip_axis.min()>-.65 or tip_axis.max()<.65:
            return None
        offsets = np.cumsum([len(self.rest), *counts[:-1]])
        lookup = dict(zip(keys,zip(offsets,counts)))
        pairs = np.repeat(np.array(keys),counts,axis=0)
        a,b = pairs.T
        t = signed[a]/(signed[a]-signed[b])
        self.interpolate_vertices(a,b,t)
        # Quarter-pitch samples round the opening between the intact half-pitch
        # boundaries. Only the failed cell can open at first break.
        axis_by_edge=dict(zip(keys,cut_axis))
        intervals={}
        extra_a,extra_b,extra_t=[],[],[]
        for i in sorted(connected):
            left,right=sorted(crossing[i],key=axis_by_edge.get)
            lo,hi=axis_by_edge[left],axis_by_edge[right]
            knots=np.arange(np.floor(4*lo)+1,np.ceil(4*hi))*.25
            knots=knots[(knots-lo>1e-6)&(hi-knots>1e-6)]
            start=len(self.rest)+len(extra_a)
            intervals[i]=(left,right,np.arange(start,start+2*len(knots),2))
            for key,indices in ((left,extra_a),(right,extra_b)):
                base,count=lookup[key]
                indices.extend([base,base+count-1]*len(knots))
            extra_t.extend(np.repeat((knots-lo)/(hi-lo),2))
        new_banks=np.arange(len(self.rest),len(self.rest)+len(extra_a)).reshape(-1,2)
        if len(new_banks):
            self.interpolate_vertices(np.asarray(extra_a),np.asarray(extra_b),np.asarray(extra_t))
        new_faces=[]
        for i in sorted(connected):
            tri=f[i]
            left,right,mid=intervals[i]
            for sign in (-1,1):
                def bank(key):
                    base,count=lookup[key]
                    return base+int(sign>0 and count==2)
                polygon=[]
                for a,b in zip(tri,np.roll(tri,-1)):
                    if signed[a]*sign>0:polygon.append(a)
                    if signed[a]*signed[b]<0:
                        polygon.append(bank(tuple(sorted((a,b)))))
                start=next(k for k,vertex in enumerate(polygon) if vertex in tri)
                polygon=polygon[start:]+polygon[:start]
                l,r=polygon.index(bank(left)),polygon.index(bank(right))
                middle=mid+int(sign>0)
                polygon[min(l,r)+1:min(l,r)+1]=(middle if l<r else middle[::-1]).tolist()
                new_faces.extend((polygon[0],polygon[j],polygon[j+1]) for j in range(1,len(polygon)-1))
        outer=np.delete(f,sorted(connected),axis=0)
        ends=np.full(outer.shape,-1,dtype=np.int32)
        for key,(vertex,count) in lookup.items():
            if count==1:
                a,b=key
                for k,j in ((0,1),(1,2),(2,0)):
                    ends[((outer[:,k]==a)&(outer[:,j]==b))|((outer[:,k]==b)&(outer[:,j]==a)),k]=vertex
        self.faces=np.vstack((subdivide_edges(outer,ends),new_faces)).astype(np.int32)
        paired_edges=np.asarray(counts)==2
        edge_banks=np.column_stack((offsets[paired_edges],offsets[paired_edges]+1))
        edge_parents=np.repeat(np.asarray(keys)[paired_edges,None,:],2,axis=1)
        banks=np.vstack((edge_banks,new_banks))
        tip_ids=np.asarray([i for i,c in zip(offsets,counts) if c==1])
        rim_ids=np.vstack((banks,np.repeat(tip_ids[:,None],2,axis=1)))
        rim_axis=chart_delta(self.uv[rim_ids[:,0]],uv)[:,run_axis]
        order=np.argsort(rim_axis)
        rim_ids,rim_axis=rim_ids[order],rim_axis[order]
        paired=rim_ids[:,0]!=rim_ids[:,1]
        banks=rim_ids[paired]
        # Subdivide the prospective seam for individual loop failures. Its
        # intact portions keep the parent surface, so the finer seam alone
        # cannot insert extra folds into a coarser intact triangle.
        seam=[(edge_banks,edge_parents,fraction[paired_edges],cut_axis[paired_edges])]
        if len(new_banks):
            parents=np.column_stack((extra_a,extra_b)).reshape(-1,2,2)
            material_axis=chart_delta(self.uv[new_banks[:,0]],uv)[:,run_axis]
            seam.append((new_banks,parents,np.asarray(extra_t)[::2],material_axis))
        if self.dry and self.fixed_material():self.refresh_dry_support()
        return self.register_crack(center,uv,load_axis,across,along,phase,rim_ids,rim_axis,seam)

    def register_crack(self,center,uv,load_axis,across,along,phase,rim_ids,rim_axis,seam):
        paired=rim_ids[:,0]!=rim_ids[:,1]
        crack=dict(id=len(self.cracks)+1,banks=rim_ids[paired],locations=rim_axis[paired],
                   center=center.copy(),uv=uv.copy(),across=across.copy(),normal=self.normal.copy(),load_axis=load_axis,
                   rim_ids=rim_ids,rim_axis=rim_axis,seam=seam,fronts=np.zeros(2),progress=0.,width=0.,
                   cells=[],yarn_feed=[],edge_load=[],curve_phase=phase,seed_recoil=0.,along=along.copy())
        if self.dry:
            crack['pitch']=np.linalg.norm(self.rest[rim_ids[-1,0]]-self.rest[rim_ids[0,0]])/np.ptp(rim_axis)
        packet=getattr(self,'draw_packet',None)
        prepared=(packet[10] if packet is not None and packet[0] is self.faces and not self.cracks else None)
        if (prepared is not None and np.array_equal(uv,prepared[0]) and np.array_equal(across,prepared[1])
                and np.array_equal(along,prepared[2])):
            crack.update(prepared[3])
            for name,value in prepared[4].items():setattr(self,name,value)
        else:self.set_tear_support(crack)
        self.cracks.append(crack)
        self.open_rest=None
        self.changed=True
        if packet is not None and packet[0] is self.faces:
            self.offset,self.distance_sq,self.drag_taper,self.drag_ids,weight=packet[14]
            self.weight=weight.copy()
        else:self.weights()
        return crack

    def refresh_dry_support(self):
        # A later grab may insert vertices into a previously loaded shoulder.
        # They must carry every existing aperture on the next change of width.
        if not self.cracks:
            return
        self.bridge_supports.clear()
        self.opening=np.zeros_like(self.rest)
        for crack in self.cracks:
            self.set_tear_support(crack)
            offset=released_shape(crack,self.rest)
            crack['offset']=offset
            self.opening[crack['ids']]+=offset
        self.open_rest=None

    def set_tear_support(self, crack):
        center,uv,across,along=(crack[k] for k in ('center','uv','across','along'))
        load_axis,phase,banks,rim_axis=(crack[k] for k in ('load_axis','curve_phase','banks','rim_axis'))
        run_axis=1-load_axis
        q=self.rest-center
        lateral=q@across
        depth=q@crack['normal']
        material=chart_delta(self.uv,uv)
        axis=material[:,run_axis]
        lateral_span, skin_span, support_radius = (.041,.023,.059) if self.dry else (.023,.012,.034)
        if 'pitch' in crack:
            lateral_span,support_radius=.053,.075
        ids=np.flatnonzero((abs(lateral)<lateral_span)&(axis>rim_axis[0])&(axis<rim_axis[-1])&
                           (abs(depth)<skin_span)&(np.linalg.norm(q,axis=1)<support_radius))
        ids=np.union1d(ids,banks.ravel())
        bend = seam_bend(axis,phase) if phase is not None else 0.
        side=np.sign(material[:,load_axis]-bend)
        side[banks[:,0]]=-1
        side[banks[:,1]]=1
        if self.dry:
            # A wide opening needs a wider band to absorb the released
            # travel. A steep Gaussian folded the material beside the cut.
            u = np.clip(1.-abs(lateral[ids])/lateral_span,0.,1.)
            d = np.clip((skin_span-abs(depth[ids]))/.013,0.,1.)
            g = np.clip((support_radius-np.linalg.norm(q[ids],axis=1))/.012,0.,1.)
            decay = u*u*(3.-2.*u)*d*d*(3.-2.*d)*g*g*(3.-2.*g)
        else:
            decay=np.exp(-.5*(lateral[ids]/.006)**2-.5*(depth[ids]/.006)**2)
        if getattr(self,'pose',None) is None or 'pitch' in crack:
            self.contact_normals=surface_normals(self.rest,self.faces)
            normal=self.contact_normals[ids]
        else:
            normal=np.broadcast_to(self.normal,(len(ids),3))
        direction=across-(normal@across)[:,None]*normal
        direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-9)
        crack.update(bank_side=side[ids],ids=ids,axis=axis[ids],offset=np.zeros((len(ids),3)),
                   direction=decay[:,None]*(side[ids,None]*direction+.055*normal),
                   tip_direction=(3.*axis[ids]*decay/(1.+.45*axis[ids]**2)**2)[:,None]*along)
        if 'pitch' in crack:
            crack['cross_cells']=abs(material[ids,load_axis]-bend[ids])
            line=self.rest[crack['rim_ids'][[-1,0],0]]
            up=unit(line[0]-line[1])
            tangents=up-(normal@up)[:,None]*normal
            taper=decay*np.exp(-.38*crack['cross_cells'])
            crack['cap_tangents']=tangents*taper[:,None]
            crack['course_tangents']=direction*taper[:,None]
            crack['bias_tangents']=direction*decay[:,None]

    def step(self, dt, target=None, *, driven_input=None):
        if self.warning is not None and target is not None and driven_input is not False:
            warning=self.warning
            if 'slack_m' in warning:
                travel=(getattr(self,'pointer_travel',self.target-self.grip_base())-warning['stroke'])@warning['direction']
                warning['extension']=max(0.,float(travel)/warning['slack_m'])
        if self.grip is None or target is None or driven_input is False or self.warning is not None:
            self.solve(dt,target,driven_input)
            return
        start=self.target.copy()
        movement=target-start
        # A coalesced mouse event may jump across the rupture point. Sample
        # the loading path and stop at the first cut so it can unload first.
        steps=max(1,int(np.ceil(np.linalg.norm(movement)/.012)))
        previous_crack=self.active_crack
        for index in range(1,steps+1):
            self.solve(dt/steps,start+movement*(index/steps),driven_input)
            if self.active_crack is not previous_crack or self.warning is not None:
                break

    def solve(self, dt, target, driven_input):
        loaded = self.grip is not None
        driven = (loaded and target is not None and driven_input is not False
                  and np.linalg.norm(target-self.target)>1e-7)
        fixed=getattr(self,'pose',None) is None or (bool(self.cracks) and getattr(self,'wet_fabric_contact',False))
        if loaded and fixed and not driven:
            self.hold(dt)
            return
        dt=max(0.,dt)
        previous_target=self.target if loaded else None
        if loaded and target is not None:self.target=target.copy()
        self.elapsed += dt
        self.step_number += 1
        if not loaded and not self.moving:return
        previous_pos = None if loaded else self.pos.copy()
        damaged = fixed and self.dry and len(self.cracks)>0
        native_intact=(loaded and fixed and self.dry and not self.cracks and self.rest.dtype==np.float64
                       and contact_index._index is not None)
        goal=(self.intact_base.copy() if native_intact else self.relaxed_tears().copy() if damaged
              else self.rest+self.opening+self.seating)
        grasp_rest = goal[self.grip].copy() if loaded else None
        pull_increment=0.
        deformation_dt=dt
        if loaded and fixed:
            # Move events are the same regardless of the subdivision of the
            # stationary time between them. Integrate that time at the last
            # committed shape, and commit the new geometry once per move.
            self.slide_held_yarn(dt)
            deformation_dt+=self.held_seconds
            self.held_seconds=0.
        if loaded:
            grip_base=self.grip_base()
            translation=self.target-grip_base
            length=np.linalg.norm(translation)
            if driven:
                pull_increment=length-np.linalg.norm(previous_target-grip_base)
            direction=unit(translation)
            longitudinal=abs(direction@unit(self.axes[1]))
            spread=(.018-.004*longitudinal)*(1.15-.20*self.response.fit)
            lift=max(0.,translation@self.normal) if self.dry else 0.
            spread+=min(.0048,.24*lift)
            tension, flank = self.side_span(direction, lift, spread)
            open_crack = self.active_crack or self.nearby_crack
            nearby_cuts = ([c for c in self.cracks if np.linalg.norm(c['center']-self.anchor)<.048]
                           if fixed and self.dry and len(self.cracks)>1 else [])
            crowded = len(nearby_cuts)>1
            beside_cut = fixed and self.dry and (open_crack is not None or crowded)
            slip = translation - (translation@self.normal)*self.normal
            travel = np.linalg.norm(slip)
            normal_shift=None
            if native_intact:
                normal_shift=contact_index.pinched_profile(self,goal,translation,direction,spread,length,lift,tension,flank)
            elif beside_cut and travel>1e-6:
                # Bound the compressive gradient along the drag. With this
                # support length the fade cannot overtake cloth ahead of it.
                # After a large opening, adjacent material coordinates can
                # belong to banks now several centimetres apart. Transport
                # this grasp on the actual released sheet.
                released_offset=self.relaxed_tears()-grip_base
                axial = released_offset @ (slip/travel)
                relief = max(.048, 2.25*travel + .006)
                u = np.clip(1.-np.where(axial>0., axial/relief, -axial/.042), 0., 1.)
                transverse = np.maximum(0.,np.einsum('ij,ij->i',released_offset,released_offset)-axial**2)
                width = .0165 + .17*lift
                self.weight = u*u*(3.-2.*u)*np.exp(-.5*transverse/width**2)
            else:
                ids=self.drag_ids
                forward=np.maximum(0.,self.offset[ids]@direction)
                radial=np.maximum(0.,self.distance_sq[ids]-forward**2)
                profile=radial/spread**2+(forward/(spread+.75*length))**2
                influence=np.exp(-.5*profile)
                if lift>0.:
                    influence+=min(.64,lift/.036)*(np.exp(-.83*np.sqrt(profile))-influence)
                self.weight.fill(0.)
                self.weight[ids]=self.drag_taper[ids]*influence
            self.weight[self.grip]=1
            if not native_intact:goal+=self.weight[:,None]*translation
            if not native_intact and lift>0. and fixed and not self.cracks:
                # The fingertip supports a small area normally, while the
                # material can slip tangentially out from under the finger.
                radius=np.sqrt(np.maximum(0.,profile))
                pad=np.exp(-.5*(np.maximum(0.,radius-.0038/spread)/.89)**2)
                gain=self.drag_taper[ids]*np.maximum(0.,pad-influence)
                gain[np.isin(ids,self.grip)]=0.
                goal[ids]+=gain[:,None]*(lift*self.normal)
            if flank is not None and not native_intact:
                normal_shift = tension*(flank-1.)*self.weight
                normal_shift[self.grip] = 0.
                goal+=normal_shift[:,None]*(lift*self.normal)
            if self.response.wrinkles:
                center=open_crack['center']-self.anchor if open_crack is not None else None
                weights = self.weight
                if flank is not None:
                    # A taut span lets excess cloth fold at the attached
                    # shoulders. It cannot pile the folds onto the lifted tip.
                    weights = weights*(1.-.58*tension*flank)**2
                if crowded and center is None:
                    previous = min(nearby_cuts, key=lambda c: np.linalg.norm(c['center']-self.anchor))
                    center = previous['center']-self.anchor
                if beside_cut:
                    # The released edge cannot sustain the compression of
                    # intact cloth. Keep the folds on the supported shoulder.
                    cut_distance = np.linalg.norm(self.offset-center,axis=1)
                    band = np.clip((cut_distance-.009)/.018,0.,1.)
                    weights = weights*(.12+.88*band*band*(3.-2.*band))*.63
                self.patch.buckle(goal,translation,self.normal,weights,center)
            for crack in self.cracks:
                # The continuous cloth field carries the older holes. Only
                # the loaded opening gets a separate elastic lip displacement.
                if not crowded or crack is self.active_crack or crack is self.nearby_crack:
                    goal[crack['ids']]+=self.lip_offset(crack,translation)
            if self.run_links:
                self.pull_linked_lips(goal,open_crack,translation)
            if damaged:
                goal+=self.weight[:,None]*(self.target-self.bary@goal[self.grip])
            else:
                goal[self.grip]=grasp_rest+translation
            self.pulled |= length>.0002
        self.tie_fronts(goal)
        deformation_dt=min(max(deformation_dt,.0001),.10)
        factor=-np.expm1(-deformation_dt/(.018 if loaded else .095/self.response.recovery))
        native_hold=loaded and fixed and self.dry and contact_index.advance_grip(self,goal,factor,normal_shift)
        if not loaded and self.patch is not None:
            previous=self.pos[self.patch.ids].copy()
            self.pos+=(goal-self.pos)*factor
            self.pos[self.patch.ids]=previous
            self.patch.recover(self.pos,goal,deformation_dt)
        elif not native_hold:
            self.pos+=(goal-self.pos)*factor
        if loaded and not native_hold:
            correction=self.target-self.bary@self.pos[self.grip]
            self.pos+=self.weight[:,None]*correction
            if flank is not None:
                self.pos+=normal_shift[:,None]*((correction@self.normal)*self.normal)
        if self.patch is not None:
            if fixed:
                self.constrain_skin()
            self.tie_fronts(self.pos)
            if fixed and self.dry and self.cracks:
                self.constrain_tear_support()
            self.patch.update_stitches(self.pos,0. if loaded and fixed else deformation_dt)
        self.moving=(loaded or np.max(np.abs(previous_pos-self.pos))>1e-6
                     or (self.patch is not None and self.patch.loops.sliding))
        self.advance_fracture(translation if loaded else None, driven, pull_increment, dt)

    def lip_offset(self, crack, translation):
        center = self.rest[crack['banks']].mean(axis=(0,1))
        across = crack['across']
        side = np.sign((self.anchor-center)@across) or 1.
        separation = side*(translation@across)
        lift = max(0., translation@self.normal)
        influence = np.exp(-.5*(np.linalg.norm(self.anchor-center)/.026)**2)
        if self.dry:
            remaining = max(.0016,.015-crack['width'])
            elastic_limit=min(remaining,.007,.0013+.00088*(len(crack['cells'])-1)+crack['seed_recoil'])
        else:
            elastic_limit=min(.0055,.00090+.00065*(len(crack['cells'])-1)+crack['seed_recoil'])
        width = np.clip(influence*(.40*separation+.24*lift), -.75*crack['width'], elastic_limit)
        tip_spread=crack['seed_recoil']*min(1.,max(0.,width)/.001)*max(0.,(5-len(crack['cells']))/4)
        return aperture(crack,width)[:,None]*crack['direction']+tip_spread*crack['tip_direction']

    def pull_linked_lips(self, goal, open_crack, translation):
        if open_crack is None:
            return
        side = np.sign((self.anchor-open_crack['center'])@open_crack['across']) or 1.
        pull = .56*side*(translation@open_crack['across'])+.41*max(0.,translation@self.normal)
        if pull <= 0.:
            return
        for first, second, _, _, _, _, strength in self.run_links:
            if first is open_crack or second is open_crack:
                neighbour = second if first is open_crack else first
                remaining = max(.0013,.015-neighbour['width'])
                transmitted=.76*pull
                if 'pitch' in open_crack:
                    from .tear_strip import bridge_samples
                    remaining=np.clip(.001+.24*neighbour['width'],.0025,.007)
                    key=open_crack['id'],neighbour['id'],side
                    if key not in self.bridge_supports:
                        self.bridge_supports[key]=bridge_samples(self,open_crack,neighbour,side)
                    support=self.bridge_supports[key]
                    if support is None:
                        continue
                    ids,weights=support
                    neutral=np.einsum('ni,nij->nj',weights,self.relaxed_tears()[ids])
                    loaded=np.einsum('ni,nij->nj',weights,goal[ids])
                    displacement=loaded-neutral
                    gradient=displacement[0]-displacement[1]
                    shear=side*(gradient@open_crack['across'])
                    extension=np.linalg.norm(loaded[1]-loaded[0])-np.linalg.norm(neutral[1]-neutral[0])
                    transmitted=max(0.,shear+.45*max(0.,gradient@open_crack['normal']),extension)
                width = min(remaining,.0068 if 'pitch' in open_crack else .0055,transmitted*np.sqrt(strength)*self.response.run)
                goal[neighbour['ids']] += aperture(neighbour,width)[:,None]*neighbour['direction']

    def tie_fronts(self, positions, *, relaxed=False):
        grip = None if relaxed else self.grip
        for crack in self.cracks:
            banks=self.closed_banks(crack)
            positions[banks]=positions[banks].mean(1)[:,None]
            lo,hi=crack['fronts']
            for pairs,parents,fraction,axis in crack['seam']:
                intact=(axis<=lo+1e-8)|(axis>=hi-1e-8)
                if grip is not None:
                    intact &= ~np.isin(pairs,grip).any(1)
                left,right=positions[parents[intact]].mean(1).transpose(1,0,2)
                bound=left+(right-left)*fraction[intact,None]
                positions[pairs[intact]]=bound[:,None]
            if 'pitch' not in crack and len(crack['cells'])<5:
                # Once released, a short lip follows the tensile line between
                # the two intact tips. Its sub-pitch samples cannot double back.
                region=rim_slice(crack)
                pairs=crack['rim_ids'][region]
                if len(pairs)<3:continue
                front=positions[pairs[[0,-1]]].mean(1)
                run=front[1]-front[0]
                direction=unit(run)
                if grip is not None and self.active_crack is crack:
                    # Under the initial load the intact stitches at the tips
                    # can elongate without disengaging. Start from their
                    # bound positions refreshed above, so this cannot accumulate.
                    stretch=1.65*crack['seed_recoil']*max(0.,(3-len(crack['cells']))/2)
                    tips=pairs[[0,-1]]
                    intact=~np.isin(tips,grip).any(1)
                    increase=np.array([-stretch,stretch])*intact
                    positions[tips]+=(increase[:,None]*direction)[:,None]
                    front+=increase[:,None]*direction
                    run=front[1]-front[0]
                along=crack['rim_axis'][region][1:-1]
                fraction=(along-lo)/(hi-lo)
                inner=pairs[1:-1]
                if grip is not None:
                    free=~np.isin(inner,grip).any(1)
                    inner,fraction=inner[free],fraction[free]
                distance=(positions[inner]-front[0])@direction
                correction=fraction[:,None]*np.linalg.norm(run)-distance
                positions[inner]+=((5-len(crack['cells']))*.25*correction[:,:,None])*direction

    def closed_banks(self, crack):
        return crack['banks'][(crack['locations']<=crack['fronts'][0])|(crack['locations']>=crack['fronts'][1])]

    def display_faces(self):
        if not self.cracks:return self.faces
        key=tuple(tuple(c['fronts']) for c in self.cracks)
        cache=getattr(self,'weld_cache',None)
        if cache is not None and cache[0] is self.faces and cache[1]==key:return cache[2]
        welding=np.arange(len(self.pos),dtype=np.int32)
        for crack in self.cracks:
            banks=self.closed_banks(crack)
            welding[banks[:,1]]=banks[:,0]
        result=welding[self.faces]
        self.bank_map=welding
        self.weld_cache=self.faces,key,result
        return result

    def current_tear_faces(self):
        if self.tear_support_faces is None or self.tear_support_faces[0] is not self.faces:
            support = np.unique(np.concatenate([crack['ids'] for crack in self.cracks]))
            nearby = np.isin(self.faces, support).any(axis=1)
            adjacent = np.unique(self.faces[nearby])
            self.tear_support_faces = self.faces, np.isin(self.faces, adjacent).any(axis=1)
        visible=self.display_faces()
        cache=getattr(self,'local_faces_cache',None)
        if cache is None or cache[0] is not visible or cache[1] is not self.tear_support_faces:
            cache=self.local_faces_cache=(visible,self.tear_support_faces,visible[self.tear_support_faces[1]])
        return cache[2]

    def tear_contact_band(self):
        # A displaced shoulder can meet stationary cloth on the far side of
        # the strain patch. Include its full swept neighbourhood for contact,
        # without adding longitudinal springs to that stationary cloth.
        if self.tear_support_faces is None or self.tear_support_faces[0] is not self.faces:
            self.current_tear_faces()
        surface=self.display_faces()
        support=self.tear_support_faces
        cache=getattr(self,'contact_band',None)
        if cache is not None and cache[0] is surface and cache[1] is support:return cache[2:]
        domain=getattr(self,'contact_material_band',None)
        if domain is None or domain[0] is not support:
            near=np.zeros(len(self.rest),bool)
            for crack in self.cracks:
                offset=self.rest-crack['center']
                radius=np.linalg.norm(offset[crack['ids']],axis=1).max()+.025
                near|=np.einsum('ij,ij->i',offset,offset)<radius*radius
            domain=self.contact_material_band=(support,near[self.faces].any(1)|support[1])
        ids,local=np.unique(surface[domain[1]],return_inverse=True)
        tri=local.reshape(-1,3)
        self.contact_band=surface,support,ids,tri
        return ids,tri

    def open_rim_ids(self):
        return np.concatenate([crack['rim_ids'][rim_slice(crack)].ravel() for crack in self.cracks])

    def relaxed_tears(self):
        """The neutral shape keeps the released yarn; the next grip adds to it."""
        if self.open_rest is not None and self.open_rest[0] is self.faces:
            return self.open_rest[1]
        shape = self.rest+self.opening+self.seating
        self.tie_fronts(shape,relaxed=True)
        faces = self.current_tear_faces()
        if hasattr(self,'project_skin'):
            normals = self.contact_normals
            if normals is None or len(normals) != len(shape):
                normals = self.contact_normals = surface_normals(self.rest,self.faces)
            ids = np.flatnonzero(np.linalg.norm(shape-self.rest,axis=1)>.0001)
            if hasattr(self,'seat_opening'):
                shape[ids]=self.seat_opening(self.rest[ids]+self.seating[ids],shape[ids],normals[ids])
            else:
                shape[ids]=self.project_skin(shape[ids],normals[ids])
        seated=shape.copy()
        previous=getattr(self,'neutral_history',None)
        if previous is not None and previous[0] is self.faces:
            # The accepted, unloaded sheet is the starting point for the
            # next paid-out stitch. Reuse its contact corrections where the
            # material target stayed nearby, without integrating past pulls.
            old_target,old_solution=previous[1:]
            delta=seated-old_target
            u=np.clip(1.-np.einsum('ij,ij->i',delta,delta)/(.004**2),0.,1.)
            shape+=u[:,None]*(old_solution-old_target)
        limit_strain(shape,self.tear_frame(faces))
        from .tear_surface import limit_held_travel
        limit_held_travel(shape,self.rest,self.tear_frame(faces))
        banks = np.concatenate([self.closed_banks(crack) for crack in self.cracks])
        shape[banks[:,1]] = shape[banks[:,0]]
        # On a narrow bend, seating the released stitches can drive a shoulder
        # through the next row. Repair the neutral target before a new pull uses it.
        ids,tri=self.tear_contact_band()
        target=shape[ids]
        base=self.rest[ids]
        old_rest=previous[2][ids] if previous is not None and previous[0] is self.faces else None
        folded=seam_overlap.settle_paid_out(base,target,tri,old_rest)
        # A curved calf can also cross the inside of a face even when all of
        # its corners clear the skin. Seat the final triangles after limiting strain.
        contact=getattr(self,'monitor_contact',None)
        if contact is not None or (getattr(self,'wet_fabric_contact',False) and hasattr(self,'clearance')):
            from .seam_contact import clear_skin
            if not hasattr(self,'clearance'):
                # This fixed-pose membrane is solved on Blender's main thread.
                # Its skin BVH already describes the posed collider. Measure the
                # actual first skin intersection along the garment normal.
                tree=contact[0]
                def clearance(points,normals):
                    origins=(points+.014*normals).tolist()
                    directions=(-normals).tolist()
                    distances=np.asarray([tree.ray_cast(o,d,.025)[3] for o,d in zip(origins,directions)],dtype=float)
                    return np.where(np.isfinite(distances),distances-.014,.025)
                self.clearance=clearance
            normals=self.contact_normals[ids]
            if clear_skin(self,base,target,tri,normals) and seam_overlap.settle_paid_out(base,target,tri,old_rest):
                folded=True
                clear_skin(self,base,target,tri,normals)
        if folded:
            if len(seam_overlap.intersections(target,tri)):
                target=seam_overlap.paid_out_reference(base,target,old_rest,tri)[0].copy()
        shape[ids]=target
        shape[banks[:,1]]=shape[banks[:,0]]
        if getattr(self,'pose',None) is None and not getattr(self,'wet_fabric_contact',False):
            from .seam_gap import retain_gap
            for c in self.cracks:
                pairs=c['rim_ids'][rim_slice(c)]
                pairs=pairs[pairs[:,0]!=pairs[:,1]]
                if len(pairs)<3:continue
                stock=np.linalg.norm(seated[pairs[:,0]]-seated[pairs[:,1]],axis=1)
                peak=float(stock.max())
                if peak>.0006:
                    retain_gap(shape,self.display_faces(),pairs,stock/peak,min(.009,.28*peak),self,self.contact_normals)
            shape[banks[:,1]]=shape[banks[:,0]]
        fold_region=(ids,tri) if folded else None
        self.neutral_history=self.faces,seated,shape
        self.open_rest = self.faces,shape,fold_region
        return shape

    def tear_frame(self,faces):
        spans=tuple(tuple(c['fronts']) for c in self.cracks)
        cache=self.strain_cache
        if cache is None or cache[0] is not self.faces or cache[1]!=spans:
            material=getattr(self,'material_frame',None)
            if material is None or material[0] is not self.tear_support_faces:
                frame=strain_frame(self.rest,self.faces[self.tear_support_faces[1]])
                material=self.material_frame=(self.tear_support_faces,frame)
            raw=material[1]
            ids=self.bank_map[raw[0]]
            corners=ids.ravel()
            order=np.argsort(corners,kind='stable')
            slots=np.r_[0,np.cumsum(np.bincount(corners,minlength=len(self.rest)))]
            frame=(ids,*raw[1:5],(slots,order//3))
            cache=self.strain_cache=(self.faces,spans,frame)
        return cache[2]

    def held_needs_reference(self,frame):
        faces,length,_,height,normal,_=frame
        a,b,c=self.pos[faces].transpose(1,0,2)
        signed=np.einsum('ij,ij->i',np.cross(b-a,c-a),normal)
        # A small rounding guard retains the native solver at the threshold.
        return np.any(signed<=.018*length*height+1e-15)

    def constrain_tear_support(self):
        banks=np.concatenate([self.closed_banks(c) for c in self.cracks])
        faces=self.current_tear_faces()
        limit_strain(self.pos,self.tear_frame(faces))
        if self.grip is not None and (self.active_crack is not None or self.nearby_crack is not None):
            from .tear_surface import limit_held_travel
            frame=self.tear_frame(faces)
            # The released shape is needed only to repair a compressed fan.
            # On the first opening the neutral solve would otherwise block the
            # first visible frame even when all held triangles meet its bound.
            if self.open_rest is not None or self.held_needs_reference(frame):
                limit_held_travel(self.pos,self.relaxed_tears(),frame)
        elif self.grip is None and self.open_rest is not None:
            # Released yarn and the pinch patch recover on different time scales.
            # Contact is enforced on the actual intermediate surface, even when
            # the fully released rest state had no intersections.
            ids,tri=self.tear_contact_band()
            recovering=self.pos[ids];neutral=self.open_rest[1][ids]
            if seam_overlap.uncross(neutral,recovering,tri,released_from=(self.rest[ids],neutral)):
                self.pos[ids]=recovering
        self.pos[banks[:,1]] = self.pos[banks[:,0]]

    def display_positions(self):
        if not self.cracks or 'pitch' in self.cracks[0]:
            return self.apply_warning_display(self.pos)
        if self.rim_strips is None or self.rim_strips[0] is not self.faces:
            self.rim_strips = (self.faces,support_strips(self.rest,self.uv,self.faces,self.cracks))
        correction = np.zeros_like(self.pos)
        count = np.zeros(len(self.pos),int)
        for crack,strip in zip(self.cracks,self.rim_strips[1]):
            ids,delta = fair_rim(self.pos,self.rest,crack,strip,self.grip)
            if self.dry and self.grip is not None and (self.active_crack is not None or self.nearby_crack is not None):
                d = np.linalg.norm(self.rest[ids]-self.anchor,axis=1)
                u = np.clip((d-.010)/.018,0.,1.)
                delta *= (.42+.58*u*u*(3.-2.*u))[:,None]
            keep = np.linalg.norm(delta,axis=1)>5e-7
            ids,order = np.unique(ids[keep],return_index=True)
            correction[ids] += delta[keep][order]
            count[ids] += 1
        ids = np.flatnonzero(count)
        positions = self.pos.copy()
        positions[ids] += correction[ids]/count[ids,None]
        pose = getattr(self,'pose',None)
        if len(ids):
            if pose is not None:
                pose.constrain_display_rim(self.pos, positions, ids)
            elif hasattr(self,'project_skin'):
                if self.contact_normals is None or len(self.contact_normals)!=len(self.rest):
                    self.contact_normals = surface_normals(self.rest,self.faces)
                positions[ids] = self.project_skin(positions[ids],self.contact_normals[ids])
        if self.grip is not None and self.grab_surface_offset is not None:
            if pose is None and self.dry:
                u = np.clip(1.-self.patch.distance/.009, 0., 1.)
                positions[self.patch.ids] += (u*u*(3.-2.*u))[:, None]*self.grab_surface_offset
                positions[self.grip] = self.pos[self.grip] + self.grab_surface_offset
            else:
                positions[self.grip] += self.grab_surface_offset
        if pose is None and self.dry:
            limit_fairing(self.pos, positions, self.display_faces())
        return self.apply_warning_display(positions)

    def apply_warning_display(self, positions):
        if self.warning is None:
            return positions
        if positions is self.pos:
            positions=positions.copy()
        patch=self.patch
        near=(patch.distance<.011)&(patch.normal@self.normal>.55)&~np.isin(patch.ids,self.grip)
        ids=patch.ids[near]
        taper=(1.-(patch.distance[near]/.011)**2)**2
        offset=self.warning_offset()
        positions[ids]+=taper[:,None]*offset
        # Keep the on-screen pinch and the three corners it rests on together.
        positions[self.grip]+=offset
        return positions

    def seat_held_faces(self):
        if not self.dry or self.grip is None or not hasattr(self,'clearance'):
            return
        from .seam_contact import seat_held_faces
        if not self.cracks:
            if getattr(self,'pose',None) is None:
                if self.contact_normals is None:self.contact_normals=surface_normals(self.rest,self.faces)
                # Long cuff triangles can cut into the skin between three clear vertices.
                seat_held_faces(self,self.contact_normals,self.clearance,self.drag_ids,
                                face_needs=getattr(self,'held_face_lifts',None),grip_vertices=self.drag_ids)
            return
        banks=np.concatenate([self.closed_banks(c) for c in self.cracks])
        seat_held_faces(self,self.contact_normals,self.clearance,self.faces,banks,
                        face_needs=getattr(self,'held_face_lifts',None))
        if self.pose is None and (self.active_crack is not None or self.nearby_crack is not None):
            ids,tri=self.tear_contact_band()
            held=self.pos[ids]
            if self.open_rest is None:
                pairs=seam_overlap.intersections(held,tri)
                if not len(pairs):return
            else:pairs=None
            neutral=self.relaxed_tears()[ids]
            if seam_overlap.uncross(neutral,held,tri,released_from=(self.rest[ids],neutral),initial_pairs=pairs):
                self.pos[ids]=held
                self.pos[banks[:,1]]=self.pos[banks[:,0]]

    def constrain_skin(self):
        self.patch.constrain(self.pos)
        if not hasattr(self,'project_skin'):return
        if getattr(self,'static_skin',None) is not None:
            if self.contact_normals is None or len(self.contact_normals)!=len(self.rest):
                self.contact_normals=surface_normals(self.rest,self.faces)
            if contact_index.project_grab(self,self.static_skin):return
        # The asymmetric drag reaches beyond the stitch telemetry patch.
        moved=np.linalg.norm(self.pos-self.rest-self.seating,axis=1)
        ids=np.flatnonzero(moved>.00010)
        if self.grip is not None:
            ids=ids[~np.isin(ids,self.grip)]
        if self.contact_normals is None or len(self.contact_normals)!=len(self.rest):
            self.contact_normals=surface_normals(self.rest,self.faces)
        normals=self.contact_normals[ids].copy()
        _,indices,local=np.intersect1d(ids,self.patch.ids,assume_unique=True,return_indices=True)
        normals[indices]=self.patch.normal[local]
        self.pos[ids]=self.project_skin(self.pos[ids],normals)

    def hold(self, dt):
        dt=max(0.,dt)
        self.elapsed += dt
        self.step_number += 1
        self.held_seconds += dt
        # Integrate yarn exactly up to failure, so the amount paid out at the
        # first open interlock does not depend on how a window timer was split.
        before=dt
        if self.warning is not None and self.warning_ready(dt):
            before=max(0.,self.warning['duration']-self.warning['age'])
        self.held_optics_changed=self.slide_held_yarn(before)
        if self.hold_loaded or self.warning is not None or self.recoil_budget>=1.:
            self.advance_fracture(self.target-self.grip_base(), False, 0., before)
            if dt>before:self.held_optics_changed|=self.slide_held_yarn(dt-before)
        else:
            self.previous_strain = strain(self.patch.loops)

    def slide_held_yarn(self, dt):
        self.yarn_pending+=min(dt,.10)
        count=int(120*self.yarn_pending+1e-9)
        if count:
            interval=count/120
            self.yarn_pending=max(0.,self.yarn_pending-interval)
            # Pick, deformation and fracture all measure the patch after committing
            # its positions. A stationary hand advances only the yarn ledger.
            self.patch.loops.slide(interval,substeps=count)
            return self.patch.loops.sliding
        return False

    def advance_fracture(self, translation, driven, pull_increment, dt):
        loops=self.patch.loops
        if not len(loops.cells):
            self.cancel_warning()
            return
        current=strain(loops)
        rise=np.zeros_like(current) if self.previous_strain is None else current-self.previous_strain
        self.previous_strain=current
        distance=np.linalg.norm(self.patch.rest_centers-self.anchor,axis=1)
        valid=loops.intact & (distance>.007) & (distance<.034)
        if not valid.any():
            self.cancel_warning()
            return
        crack=self.active_crack
        warning=self.warning
        thresholds=self.response.break_extension*np.array([1.,1.14])-self.response.pretension
        peaks=np.quantile(current[valid],.96,axis=0)
        axis=crack['load_axis'] if crack is not None else int(np.argmax(peaks/thresholds))
        if warning is not None:
            axis=warning['axis']
        self.load_axis=axis
        tensile=current[:,axis]
        peak=float(peaks[axis])
        self.peak=max(self.peak,1.+peak)
        threshold=thresholds[axis]
        extending=pull_increment>1e-7
        payout=0.
        if self.dry and self.fixed_material() and driven and translation is not None:
            payout,self.slip_work=self.slip_work,0.
            if payout>0.:
                pull_increment+=payout
                extending=True
        phase='recovery' if translation is None else ('loading' if extending else 'unloading' if driven else 'held')
        if crack is None:
            if translation is None and self.skip_recovery_break:
                return
            previous = self.nearby_crack
            if warning is None and extending and driven and previous is not None and self.response.run>0:
                old_axis=previous['load_axis']
                old_tension=current[:,old_axis]
                old_threshold=thresholds[old_axis]
                side = np.sign((self.anchor-previous['center'])@previous['across']) or 1.
                opening = side*(translation@previous['across'])
                if self.dry:
                    opening += .45*max(0.,translation@self.normal)
                if opening>0.:
                    tip_threshold = old_threshold*(.78 if self.dry else 1.)
                    tip_load=old_tension
                    fresh=rise[:,old_axis]>0.
                    if 'pitch' in previous:
                        # Applied traction is carried by the intact stitches between
                        # the grip and the two tips, including when the lips slip.
                        span=np.linalg.norm(self.patch.rest_centers-self.anchor,axis=1)
                        traction=opening/(.007+.45*span)
                        tip_load=np.maximum(old_tension,traction)
                        fresh=fresh|(payout>0.)
                    fronts = self.run_fronts(previous) & (tip_load>=tip_threshold) & fresh
                    if fronts.any():
                        # On a new grab only work delivered to a loaded old
                        # front can resume it. Bulk strain may be far away.
                        tip = float(tip_load[fronts].max())
                        self.add_loading_work(pull_increment,tip,tip_threshold)
                        if self.growth_budget>=1.:
                            self.active_crack=previous
                            self.growth_budget-=1.
                            index=self.choose_front(previous,fronts,old_tension)
                            self.fail_stitch(previous,index,current[index],False,phase,translation,resumed=True)
                            self.fracture_events[-1]['resumed']=True
                            if self.dry:
                                self.recoil_budget=3.6*self.response.run*min(1.,tip/old_threshold)
                        return
            if warning is None:
                if driven:
                    # An animated leg moving under an idle cursor cannot
                    # start the warning or arm sustained failure by itself.
                    self.hold_loaded=extending and peak>=.90*threshold
                    self.hold_work=0.
                if phase=='held' and self.hold_loaded:
                    load=np.clip((peak/threshold-.90)/.10,0.,1.)
                    self.hold_work=self.hold_work+min(dt,.10)*load if load>0. else 0.
                sustained=phase=='held' and self.hold_work>=.14 and peak>=.90*threshold
                increasing=(driven or phase=='recovery') & (rise[:,axis]>(.001 if extending else .012))
                limit=(.90 if sustained else 1.)*threshold
                eligible=valid & (tensile>=limit) & (increasing | sustained)
            else:
                limit=warning['minimum']*threshold
                if peak<limit:
                    self.cancel_warning()
                    return
                # Holding still lets the loaded row slip over time. A further
                # outward stroke can instead consume the yarn still available
                # between the two anchors of the warning's real stitch.
                ready=self.warning_ready(dt) or (driven and warning.get('extension',0.)>=1.)
                if ready and not driven and not getattr(self,'predicting_birth',False) and getattr(self,'birth_job',None) is not None:
                    from .warning_birth import accept
                    if accept(self,dt):return
                warning['age']+=np.clip(dt,0.,WARNING_MAX_STEP)
                self.warning_dirty=True
                if not ready:
                    lanes=getattr(self,'warned_lanes',None)
                    if lanes is not None and driven:
                        seed=loops.cells[warning['stitches'][0],axis]
                        choices=np.flatnonzero(valid & (tensile>=limit) & (loops.cells[:,axis]==seed))
                        if len(choices):
                            from .warned_lane import follow
                            follow(self,choices[np.argmax(tensile[choices])],axis)
                    return
                eligible=valid & (tensile>=limit)
            if peak<limit or not eligible.any():
                if (warning is None and not self.cracks and self.warn_before_break and driven and extending
                        and peak>.68*threshold and getattr(self,'static_skin',None) is not None):
                    # This only allocates material-space data. Failure still uses the
                    # measured peak, strain and payout at the later eligible input.
                    from .warned_lane import begin
                    begin(self,np.flatnonzero(valid),tensile,axis)
                return
            indices=np.flatnonzero(eligible)
            if self.cracks:
                old=np.concatenate([rim(c,self.rest)[:,0] for c in self.cracks])
                separation=np.linalg.norm(self.patch.rest_centers[indices,None]-old,axis=2).min(1)
                indices=indices[separation>.011]
            if self.warn_before_break and warning is None and translation is not None and len(indices):
                self.warning=dict(axis=axis,minimum=.92*limit/threshold,
                                  reach=float(np.linalg.norm(translation)),age=0.,
                                  duration=DRY_WARNING_DURATION if self.dry else WARNING_DURATION)
                if self.dry:
                    seed=int(indices[np.argmax(tensile[indices])])
                    self.warning['stitches']=loaded_run(loops,seed,axis,tensile,.19*threshold)
                    if hasattr(self,'pointer_travel'):
                        # Two loop heads support the loaded wale. Take their measured yarn stock
                        # minus the straight gauge chords as the available extra hand travel.
                        slack=2.*(loops.stock[seed]-loops.reference[seed].max())
                        self.warning.update(stroke=self.pointer_travel.copy(),direction=unit(translation),
                                            slack_m=max(.002,slack),extension=0.)
                    if getattr(self,'static_skin',None) is not None:
                        from .warned_lane import begin
                        begin(self,indices[loops.cells[indices,axis]==loops.cells[seed,axis]],tensile,axis)
                self.warning_dirty=True
                return
            if self.dry and warning is not None:
                lane=loops.cells[warning['stitches'][0],axis]
                indices=indices[loops.cells[indices,axis]==lane]
                if not len(indices):
                    self.cancel_warning()
                    return
            for index in indices[np.argsort(-tensile[indices])[:12]]:
                cell=loops.cells[index]
                uv=(cell+[.5,.5])/[144,1]
                local=self.patch.directions[index,axis]
                across=unit(local-(local@self.normal)*self.normal)
                if np.linalg.norm(across)<.5:continue
                along=unit(self.patch.directions[index,1-axis])
                crack=self.split(self.patch.rest_centers[index],uv,axis,across,along)
                if crack is not None:
                    if 'pitch' in crack and warning is not None:
                        # Retain the pre-break load in this actual wale. The
                        # first open loop unloads the patch before its immediate
                        # neighbours get a chance to drop in subsequent draws.
                        selected=warning['stitches']
                        crack['charged_loops']={tuple(cell):float(load/threshold)
                            for cell,load in zip(loops.cells[selected],tensile[selected])}
                    self.warning=None
                    self.active_crack=crack
                    self.fail_stitch(crack,index,current[index],True,phase,translation)
                    if warning is not None:
                        self.fracture_events[-1]['warning_ms']=float(warning['age']*1000)
                    return
            self.cancel_warning()
            return
        if self.response.run<=0:return
        candidates=self.run_fronts(crack)
        if 'wale_travel' in crack and driven and extending:
            travel=getattr(self,'pointer_travel',translation)
            along=crack['wale_pull_side']*float(travel@crack['along'])
            extra=max(0.,along-crack['wale_travel'])
            if extra:
                crack['wale_travel']=along
                steps=2.*self.response.run*crack['wale_run']*extra/crack['pitch']
                room=np.ptp(crack['rim_axis'])-len(crack['cells'])-crack['wale_reserve']
                paid=min(max(0.,room),steps)
                crack['wale_reserve']+=paid
                self.recoil_budget+=paid
        if crack.get('wale_reserve',0.)>=1. and (phase=='held' or extending):
            fronts=np.flatnonzero(candidates)
            if len(fronts):
                rows=loops.cells[fronts,1]-crack['cells'][0][1]
                # Unreleased hand travel supplies each paid row once. The
                # shorter arm receives more of it as the longer arm unloads.
                force=np.clip(tensile[fronts]/threshold,0.,1.5)
                score=force+.45*np.sign(rows)*crack['wale_pull_side']-.27*abs(rows)
                # The cross-course anchors the short lobe. Once a few
                # rows have cleared it, excess yarn runs down the free arm.
                held_side=np.sign(crack['cross_course'])
                beyond=held_side*(rows-crack['cross_course'])
                score-=.37*crack['wale_run']*np.maximum(beyond-1.5,0.)
                index=fronts[np.argmax(score)]
                crack['wale_reserve']-=1.
                self.recoil_budget=max(0.,self.recoil_budget-1.)
                self.fail_stitch(crack,index,current[index],False,phase,translation)
                return
        # Only fresh displacement supplies work from the hand. Redistribution
        # during release can supply local work when a front loop is stretching.
        if phase=='held':
            stored=crack.get('charged_loops')
            if stored and (self.recoil_budget>=1. or self.growth_budget>=1.):
                front=np.flatnonzero(candidates)
                charged=np.array([stored.get(tuple(loops.cells[i]),0.) for i in front])
                scores=tensile.copy()
                scores[front]=np.maximum(scores[front],threshold*charged)
                candidates&=scores>=threshold*.30
                if candidates.any():
                    if self.recoil_budget>=1.:
                        self.recoil_budget-=1.
                    else:
                        self.growth_budget-=1.
                    index=self.choose_front(crack,candidates,scores)
                    self.fail_stitch(crack,index,current[index],False,phase,translation)
                else:
                    self.recoil_budget=0.
                return
            if self.recoil_budget<1.:return
            self.recoil_budget-=1.
            candidates&=tensile>=threshold*.40
            if candidates.any():
                index=self.choose_front(crack,candidates,tensile)
                self.fail_stitch(crack,index,current[index],False,phase,translation)
            return
        if extending:
            if self.dry:
                tip = tensile[candidates].max(initial=0.)
                self.add_loading_work(pull_increment,max(tip,peak),.80*threshold)
            else:
                self.add_loading_work(pull_increment,peak,threshold)
            if 'pitch' in crack and payout>0.:
                span=np.linalg.norm(self.patch.rest_centers-self.anchor,axis=1)
                opening=abs(translation@crack['across'])+.45*max(0.,translation@self.normal)
                candidates&=np.maximum(tensile,opening/(.007+.45*span))>threshold*.55
            else:
                candidates&=(tensile>threshold*.30)&(rise[:,axis]>0.)
        else:
            candidates&=(tensile>=threshold)&(rise[:,axis]>.012)
            self.growth_budget=min(2.,self.growth_budget+12.*self.response.run*rise[candidates,axis].max(initial=0.))
        if self.growth_budget<1.:return
        if not candidates.any():return
        index=self.choose_front(crack,candidates,tensile)
        self.growth_budget-=1.
        self.fail_stitch(crack,index,current[index],False,phase,translation)

    def add_loading_work(self, increment, extension, threshold):
        overload=np.clip(extension/max(threshold,1e-6)-.90,0.,2.)
        rate = 1450. if self.dry else 520.
        self.growth_budget=min(2.,self.growth_budget+increment*self.response.run*rate*overload)

    def choose_front(self, crack, candidates, tensile):
        score = np.where(candidates,tensile,-np.inf)
        if self.run_links:
            cells = (self.patch.loops.cells+.5)/[144,1]
            along = chart_delta(cells,crack['uv'])[:,1-crack['load_axis']]
            for first, second, direction, _, _, _, strength in self.run_links:
                if crack is first or crack is second:
                    side = direction if crack is first else -direction
                    affected = candidates & (side*(along-np.mean(crack['span'])) > 0.)
                    score[affected] *= 1.+.60*strength
        return int(np.argmax(score))

    def run_fronts(self, crack):
        loops=self.patch.loops
        axis=crack['load_axis']
        run_axis=1-axis
        positions=chart_delta((loops.cells+.5)/[144,1],crack['uv'])
        next_offset=np.array(crack['span'])+[-1,1]
        front=np.isclose(positions[:,run_axis,None],next_offset,atol=1e-6).any(1)
        return (loops.intact & front & (abs(positions[:,axis])<.1) &
                (positions[:,run_axis]>crack['rim_axis'][0]+.6) &
                (positions[:,run_axis]<crack['rim_axis'][-1]-.6))

    def fail_stitch(self, crack, index, extensions, first, phase, translation, *, resumed=False):
        loops=self.patch.loops
        x,y=map(int,loops.cells[index])
        chord_extension=(loops.stretch[index]-1.).tolist()
        self.failed_cells[y-self.yarn_first_row,x]=True
        crack['cells'].append((x,y))
        crack['yarn_feed'].append(float(loops.feed[index]))
        if self.dry:
            load = float(extensions[crack['load_axis']])
            threshold = self.response.break_extension*(1.14 if crack['load_axis'] else 1.)-self.response.pretension
            crack['edge_load'].append(float(np.clip(load/max(threshold,.10),.15,1.60)))
        if first:
            gain,cap=(.0018,.0026) if self.dry else (.00135,.0021)
            crack['seed_recoil']=min(cap,gain*extensions[crack['load_axis']])
            if 'pitch' in crack:
                travel=crack['across'] if translation is None else translation
                along_load=abs(travel@crack['along'])
                cross_load=abs(travel@crack['across'])
                pull_across=cross_load/max(cross_load+along_load,.0001)
                crack['aperture_aspect']=.34+.88*pull_across**2.1
                crack['draw_side']=float(np.sign(travel@crack['across']))
                # Choose the more heavily loaded front of the captured wale.
                # A short transverse course at that end can leave a small eye
                # beside the main opening. An oblique pull gives a slender run.
                charge=crack.get('charged_loops',{})
                course=1-crack['load_axis']
                origin=crack['cells'][0][course]
                offsets=np.array([k[course]-origin for k in charge])
                if course==0:
                    offsets=(offsets+72)%144-72
                force=np.fromiter(charge.values(),dtype=float)
                side=1 if force[offsets>0].sum()>=force[offsets<0].sum() else -1
                # The shearable shoulder needs room on the actual worn tube.
                # A wide planar release around a tight bend can run back into
                # the other bank. Measure the available transverse radius.
                from .fracture import tube_radius
                base=slice(0,self.fitted_vertices)
                packet=getattr(self,'draw_packet',None)
                radius=(packet[15] if packet is not None and packet[0] is self.faces
                        else tube_radius(self.rest[base],self.uv[base],crack['uv'][1]))
                room=np.clip((radius-.025)/.025,0.,1.)
                room=room*room*(3.-2.*room)
                crack['tube_radius']=radius
                crack['tube_room']=float(room)
                stagger=abs(2*x+y)%3
                crack['cross_course']=side*((4.+stagger)*(1.-room)+(2.25+.37*stagger)*room)
                crack['course_hold']=.28+.65*pull_across**2
                # Store the paid-out bias in the material frame. It survives
                # release and regrabbing from the opposite side of the rim.
                along_pull=float(travel@crack['along'])
                shear=along_pull/max(np.linalg.norm(travel),1e-8)
                if room>0.:
                    crack['rim_bias']=room*(crack['draw_side']*(.54+.20*pull_across)+.26*shear)
                if (crack['load_axis']==0 and self.response.run>0. and
                        not getattr(self,'wet_fabric_contact',False)):
                    oblique=np.clip((1.-pull_across-.25)/.16,0.,1.)
                    carry=oblique*shared_wales(crack,self.cracks,self.response.run)
                    if carry>0.:
                        crack['wale_run']=float(carry)
                        crack['wale_pull_side']=float(np.sign(along_pull))
                        raw=getattr(self,'pointer_travel',travel)
                        reached=max(along_load,crack['wale_pull_side']*float(raw@crack['along']))
                        crack['wale_travel']=reached
                        crack['wale_reserve']=min(2.*reached/crack['pitch']*self.response.run*carry,
                                                  np.ptp(crack['rim_axis'])-1.)
                        # In the loaded half, the first course past the peak
                        # still has a loop head held by the adjacent wale.
                        weak=side*offsets[(side*offsets>=2)&(force<.72)]
                        if len(weak):crack['cross_course']=side*float(weak.min())
        elif resumed and crack.get('wale_run',0.) and translation is not None:
            crack['wale_pull_side']=float(np.sign(translation@crack['along']))
            crack['wale_travel']=abs(float(translation@crack['along']))
            crack['wale_reserve']=0.
        axis=1-crack['load_axis']
        steps=np.rint(chart_delta((np.array(crack['cells'])+.5)/[144,1],crack['uv'])[:,axis]).astype(int)
        lo,hi=steps.min(),steps.max()
        crack['span']=(lo,hi)
        crack['fronts']=np.clip([lo-.50,hi+.50],crack['rim_axis'][0],crack['rim_axis'][-1])
        crack['progress']=float(np.ptp(crack['fronts'])/np.ptp(crack['rim_axis']))
        if 'rim_bias' in crack and crack['load_axis']==0 and not getattr(self,'wet_fabric_contact',False):
            crack['cap_round']=float(crack['tube_room']*np.clip((len(steps)-5)/6.,0.,1.))
        # Pretension keeps a residual aperture. Only elastic opening recoils.
        length_factor = 1.+.64*np.clip((len(steps)-3)/8,0,1)
        base,pretension=(.00064,.023) if self.dry else (.00045,.018)
        if self.dry:
            load = np.mean(crack['edge_load'])
            stock = np.mean(np.abs(crack['yarn_feed']))
            gain = (.86+.28*load+1.9*stock)*(1.+.14*min(9,max(0,len(steps)-4)))
            width = (base+pretension*self.response.pretension)*np.sqrt(len(steps))*length_factor
            crack['width'] = (min(.025,(.75+.72*crack['aperture_aspect'])*width*gain)
                              if 'pitch' in crack else min(.013,width*gain))
            self.run_links = linked_runs(self.cracks)
            self.bridge_supports.clear()
            if self.loose_yarn is None:
                self.loose_yarn = np.zeros(self.failed_cells.shape, dtype=np.float32)
            loose_stock(self.loose_yarn,self.yarn_first_row,self.cracks,self.run_links,self.response.run)
        else:
            crack['width']=min(.0063,(base+pretension*self.response.pretension)*np.sqrt(len(steps))*length_factor)
        if getattr(self,'async_neutral',False) and hasattr(self,'run_image') and self.warning is None:
            from .run_finish import prepare_run
            prepare_run(self,len(self.fracture_events)+1)
        offset=released_shape(crack,self.rest)
        change=offset-crack['offset']
        ids=crack['ids']
        self.opening[ids]+=change
        previous_neutral=self.open_rest
        self.open_rest = None
        if not first and self.grip is not None and self.dry:
            from .neutral_worker import schedule
            schedule(self,previous_neutral)
        self.pos[ids]+=change
        grip_recoil=0.
        if translation is not None:
            if first:
                recoil=aperture(crack,crack['seed_recoil'])[:,None]*crack['direction']
                if 'pitch' not in crack:
                    self.pos[ids]+=.25*self.lip_offset(crack,translation)+recoil
                    self.pos[ids]+=.75*crack['seed_recoil']*crack['tip_direction']
                self.grip_transfer=.62 if self.dry else .72
                self.recoil_budget=(5. if self.dry else 3.)*self.response.run*min(1.,np.linalg.norm(translation)/.026)
                if crack.get('charged_loops'):
                    stored=sum(min(1.,load) for load in crack['charged_loops'].values())
                    self.recoil_budget+=min(8.,stored*.72*self.response.run)
                self.recoil_budget+=crack.get('wale_reserve',0.)
                self.hold_loaded=False
                self.raw_opening=None
                unloaded=(1.-self.grip_transfer)*translation
                grip_recoil=np.linalg.norm(unloaded)
                self.target-=unloaded
                self.pos+=self.weight[:,None]*(self.target-self.bary@self.pos[self.grip])
            elif self.dry and resumed:
                self.pos[ids]+=.45*self.lip_offset(crack,translation)
            self.pos[self.grip]+=self.target-self.bary@self.pos[self.grip]
        fixed = self.fixed_material()
        if fixed:
            self.constrain_skin()
        self.tie_fronts(self.pos)
        if fixed and self.dry:
            self.constrain_tear_support()
        crack['offset']=offset
        self.changed=True
        if first:
            self.bind_patch()
            self.previous_strain=strain(self.patch.loops)
        else:
            self.patch.update_stitches(self.pos)
        self.fracture_events.append(dict(id=len(self.fracture_events)+1,hole_id=crack['id'],
                                         kind='first_break' if first else 'unloop',cell=[x,y],
                                         material_uv=[(x+.5)/144,y+.5],step=self.step_number,t_s=self.elapsed,
                                         phase=phase,
                                         course_extension=float(extensions[0]),wale_extension=float(extensions[1]),
                                         local_chord_extension=chord_extension,
                                         run_axis=('course','wale')[axis],
                                         grip_recoil_mm=float(grip_recoil*1000),
                                         opening_mm2=open_area(rim(crack,self.pos))*1e6))

    def damage_geometry(self):
        """Width-paired world-space rails and free ends, keyed by stable hole ID."""
        for crack in self.cracks:
            section=rim_slice(crack)
            banks=self.pos[crack['rim_ids'][section]]
            if len(banks)<2:continue
            edge=banks.mean(1)
            run=crack['rim_axis'][section]
            samples=chart_delta((np.array(crack['cells'])+.5)/[144,1],crack['uv'])[:,1-crack['load_axis']]
            rails=banks.transpose(1,0,2)
            loose=[]
            tangent=unit(edge[-1]-edge[0])
            roots=np.array([[np.interp(samples,run,rail[:,k]) for k in range(3)] for rail in rails]).transpose(2,0,1)
            cells=np.asarray(crack['cells']);feeds=np.asarray(crack['yarn_feed'])
            grains=np.sin(.71*cells[:,0]+1.63*cells[:,1])
            bank=np.r_[0,1,(feeds[1:]+.03*grains[1:]>0).astype(int)]
            ix=np.r_[0,np.arange(len(cells))]
            starts=roots[ix,bank];delta=roots[ix,1-bank]-starts
            gap=np.linalg.norm(delta,axis=1)
            cross=delta/np.maximum(1e-12,gap[:,None])
            grain=grains[ix,None,None];sign=(1-2*bank)[:,None,None]
            reach=np.clip(.25+.90*feeds[ix]+.11*grains[ix]*(1-2*bank),.10,.45)
            length=np.minimum(.0026,reach*gap)[:,None,None];u=np.linspace(0,1,12)[None,:,None]
            loose=starts[:,None]+length*(.88*np.sin(u*np.pi*(.66+.19*grain))*cross[:,None]
                +(.43+.30*grain)*u*u*sign*tangent+(.18+.13*grain)*np.sin(u*np.pi*.88)*crack['normal'])
            yield crack['id'],rails,loose

    def damage_yarns(self):
        """Rendered edge filaments and loose ends, with taper multipliers."""
        for hole,rails,loose in self.damage_geometry():
            crack=self.cracks[hole-1]
            phase=crack['uv']@[144*.81,.37]
            wrapped=list(rolled_edge(rails,crack['normal'],crack['rim_axis'][rim_slice(crack)],phase))
            taper=np.linspace(.94,.13,loose.shape[1])
            ends=[(path,taper) for path in loose]
            if self.dry:
                from .yarn_path import retained_courses
                rows=crack['rim_axis'][rim_slice(crack)]
                wrapped.extend(retained_courses(rails,rows,crack,phase))
                taper=taper*.52
                ends=[(path,taper) for path in loose]
            yield hole,wrapped,ends

    def fracture_readings(self, uv=None):
        """Extensions are relative chord changes; material UV wraps at U=1."""
        sample=None
        if self.patch is not None:
            loops=self.patch.loops
            selected=loops.stretch[loops.intact]-1.
            if len(selected):
                sample=dict(course_wale_extension_p05_p50_p95=np.quantile(selected,[.05,.5,.95],axis=0).tolist())
            if uv is not None and len(loops.cells):
                delta=chart_delta((loops.cells+.5)/[144,1],uv)
                matches=np.flatnonzero((abs(delta)<.5+1e-8).all(1))
                if len(matches):
                    i=matches[0]
                    sample={**(sample or {}),'cell':loops.cells[i].tolist(),'failed':bool(not loops.intact[i]),
                            'extension':(loops.stretch[i]-1.).tolist(),'yarn_stock_mm':float(loops.stock[i]*1000)}
        holes=[]
        for crack in self.cracks:
            banks=rim(crack,self.pos)
            gap=np.linalg.norm(banks[:,0]-banks[:,1],axis=1).max(initial=0.)
            holes.append(dict(id=crack['id'],material_uv=crack['uv'].tolist(),cells=crack['cells'].copy(),
                              run_axis=('course','wale')[1-crack['load_axis']],
                              gap_mm=float(gap*1000),area_mm2=open_area(banks)*1e6))
        gaps=[]
        for i,a in enumerate(self.cracks):
            first=rim(a,self.pos).reshape(-1,3)
            for b in self.cracks[i+1:]:
                second=rim(b,self.pos).reshape(-1,3)
                if len(first) and len(second):
                    spacing=np.linalg.norm(first[:,None]-second,axis=2).min()*1000
                    gaps.append(dict(hole_ids=[a['id'],b['id']],rim_sample_distance_mm=float(spacing)))
        return dict(step=self.step_number,t_s=self.elapsed,material=sample,holes=holes,
                    damage_area_mm2=sum(h['area_mm2'] for h in holes),failed_cells=int(self.failed_cells.sum()),
                    rim_spacing=gaps,run_links=[dict(hole_ids=[a['id'],b['id']],
                        bridge_in_stitches=float(abs(right-left)), coupling=float(strength))
                        for a,b,_,_,left,right,strength in self.run_links],events=self.fracture_events.copy())
