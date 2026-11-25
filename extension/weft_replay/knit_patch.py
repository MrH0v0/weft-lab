"""Local compression folds and stitch strain on a posed knitted surface (metres)."""
import numpy as np
from .loop_load import LoopGauges


def surface_normals(points, triangles):
    from .contact_index import material_normals
    native=material_normals(points,triangles)
    if native is not None:return native
    sides = points[triangles[:,1:]]-points[triangles[:,:1]]
    cross = np.cross(sides[:,0],sides[:,1])
    normal = np.zeros_like(points)
    for axis in range(3):
        weights=cross[:,axis]
        normal[:,axis]=sum(np.bincount(corner,weights=weights,minlength=len(points)) for corner in triangles.T)
    normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    return normal


class PinchPatch:
    def __init__(self, sim, faces):
        used=np.zeros(len(sim.rest),bool)
        used[faces]=True
        self.ids=np.flatnonzero(used)
        to_local=np.empty(len(used),dtype=np.intp)
        to_local[self.ids]=np.arange(len(self.ids))
        self.tri=to_local[faces].reshape(-1,3)
        rest = sim.rest[self.ids]
        self.normal = surface_normals(rest,self.tri)
        self.offset = rest - sim.anchor
        self.dry=sim.dry and getattr(sim,'pose',None) is None
        if self.dry:
            # Rest-space shape gradients give the actual loss of cloth length
            # along the pull on each triangle. Share it by material area.
            edge=rest[self.tri[:,1:]]-rest[self.tri[:,:1]]
            a,b=edge.transpose(1,0,2)
            n=np.cross(a,b)
            self.fold_area=np.linalg.norm(n,axis=1)
            n/=np.maximum(self.fold_area[:,None]**2,1e-24)
            self.fold_basis=(np.cross(b,n),np.cross(n,a))
            self.fold_edges=(a,b)
            self.corner_area=np.maximum(np.bincount(self.tri.ravel(),weights=np.repeat(self.fold_area,3),minlength=len(rest)),1e-12)
            wale=sim.axes[1]-(sim.axes[1]@sim.normal)*sim.normal
            self.wale=wale/max(np.linalg.norm(wale),1e-12)
        self.distance = np.linalg.norm(self.offset, axis=1)
        self.response = sim.response
        self.loops=LoopGauges(sim.uv[self.ids],self.tri,rest,sim.yarn_ledger,sim.yarn_first_row,sim.failed_cells)
        self.rest_centers = self.loops.centers.copy()
        self.directions = self.loops.vectors.copy()
        self._skin_plane = None
        self._sample_skin_plane = None
        pose = getattr(sim, 'pose', None)
        if pose is not None:
            pose.set_patch_planes(self, rest)
        elif getattr(sim, 'skin_planes', None):
            self.skin_plane = sim.skin_planes(rest, self.normal)

    @property
    def skin_plane(self):
        if self._sample_skin_plane is not None:
            self.skin_plane = self._sample_skin_plane()
        return self._skin_plane

    @skin_plane.setter
    def skin_plane(self, value):
        self._skin_plane = value
        self._sample_skin_plane = None

    def defer_skin_plane(self, sample):
        self._skin_plane = None
        self._sample_skin_plane = sample

    def update_stitches(self, pos, dt=0, *, substeps=None):
        self.loops.update(pos[self.ids],dt,substeps=substeps)

    def material_path(self, uv, pos, cell, samples, normal):
        """Follow a sub-stitch path on the actual posed triangle banks."""
        faces = self.ids[self.tri]
        chart = uv[faces].copy()
        chart[:,:,0] -= np.round(chart[:,:,0]-cell[0]/144)
        chart = chart*[144,1]-cell
        nearby = ((chart.min(1)<=samples.max(0))&(chart.max(1)>=samples.min(0))).all(1)
        chart,faces = chart[nearby],faces[nearby]
        basis = (chart[:,1:]-chart[:,:1]).transpose(0,2,1)
        valid = abs(np.linalg.det(basis))>1e-8
        chart,faces,basis = chart[valid],faces[valid],basis[valid]
        ab = np.einsum('fij,pfj->pfi',np.linalg.inv(basis),samples[:,None,:]-chart[None,:,0])
        bary = np.concatenate((1-ab.sum(2,keepdims=True),ab),axis=2)
        score = bary.min(2)
        chosen = score.argmax(1)
        weights = bary[np.arange(len(samples)),chosen]
        if weights.min() < -1e-5:
            raise ValueError(f'Stitch {cell} lies outside the traced knit patch')
        vertices = pos[faces[chosen]]
        sides = vertices[:,1:]-vertices[:,:1]
        outward = np.cross(sides[:,0],sides[:,1])
        outward /= np.maximum(np.linalg.norm(outward,axis=1,keepdims=True),1e-12)
        outward *= np.where(outward@normal<0,-1.,1.)[:,None]
        return np.einsum('ij,ijk->ik',weights,vertices)+.00011*outward

    def buckle(self, goal, translation, grip_normal, weight, open_center=None):
        slip = translation - grip_normal * (translation @ grip_normal)
        travel = np.linalg.norm(slip)
        if travel < 1e-6:
            return
        direction = slip / travel
        if self.dry:
            from .contact_index import buckle_patch
            if buckle_patch(self,goal,direction,grip_normal,weight,open_center,travel):return
        along = self.offset @ direction
        across = self.offset @ np.cross(grip_normal, direction)
        if self.dry:
            local=goal[self.ids]
            a,b=(local[self.tri[:,1:]]-local[self.tri[:,:1]]).transpose(1,0,2)
            da,db=self.fold_basis
            carried=(da@direction)[:,None]*a+(db@direction)[:,None]*b
            # Tangency varies around a curved leg. Compare each face with
            # its own projected reference length to avoid folds on the far wall.
            rest_a,rest_b=self.fold_edges
            natural=np.linalg.norm((da@direction)[:,None]*rest_a+(db@direction)[:,None]*rest_b,axis=1)
            short=np.clip(natural-np.linalg.norm(carried,axis=1)-.035,0.,.48)
            stress=np.bincount(self.tri.ravel(),weights=np.repeat(short*self.fold_area,3),
                               minlength=len(local))/self.corner_area
            # The fold crests run obliquely through short compressed strips.
            # A stiff wale biases their orientation. Yarn just under the finger
            # or at an open edge can slip, so it carries less compression.
            skew=.18+.23*np.dot(np.cross(grip_normal,direction),self.wale)
            strip=across+skew*along
            scale=.008+.13*min(travel,.036)
            phase=(along+skew*across+.0022*np.sin(strip/.011))/scale
            envelope=np.exp(-.5*(strip/(.011+.15*min(travel,.045)))**2)
            anchor=np.clip((self.distance-.0035)/.007,0.,1.)
            taper=np.clip((.049-self.distance)/.016,0.,1.)
            lift=.46*scale/np.pi*np.sqrt(stress)*(1.-np.cos(2*np.pi*phase))
            lift*=anchor*taper*envelope*np.sqrt(np.clip(weight[self.ids],0.,1.))*np.clip(self.normal@grip_normal,0.,1.)
            if open_center is not None:
                d=self.offset-open_center
                lift*=1.-.76*np.exp(-.5*np.einsum('ij,ij->i',d,d)/.029**2)
            goal[self.ids]+=lift[:,None]*self.normal
            return
        # Lost projected length ahead of a grab supplies the excess arc length
        # for buckling. No compression gives a flat patch. Phase is material-fixed.
        width = .015 + .75 * np.linalg.norm(translation)
        slack = np.maximum(0, travel * along * weight[self.ids] / width**2)
        # The lifted unsupported span admits wider folds. The wave number
        # increases towards the skin-supported fringe. Integrate it along the
        # material strip so ridges remain continuous as local pitch changes.
        free_span=.015
        support=np.clip(self.normal@grip_normal,0,1)
        height=max(0.,translation@grip_normal)
        relief=np.clip(.36+height/.045,.36,.69)*support
        run=np.maximum(along,0)
        phase=(run-relief*free_span*(-np.expm1(-run/free_span))+
               .26*across**2/.018)/.0058
        wavelength=.0058/(1-relief*np.exp(-run/free_span))
        lift = wavelength / np.pi * np.sqrt(slack) * (1-np.cos(2*np.pi*phase))
        lift *= np.clip(along / .004, 0, 1) * np.clip((.043-self.distance)/.010, 0, 1)
        if open_center is not None:
            # Free banks cannot carry the same transverse compression as the
            # intact sheet. Keep smaller folds inside the relieved band.
            distance_sq = np.sum((self.offset-open_center)**2,axis=1)
            lift *= 1.-.76*np.exp(-.5*distance_sq/.029**2)
        goal[self.ids] += lift[:, None] * self.normal

    def recover(self, pos, goal, dt):
        ids = self.ids
        delta = pos[ids]-goal[ids]
        height = np.sum(delta*self.normal, axis=1)
        # Tangential slip arrests first. Local normal return is overdamped;
        # the contact front reaches shallow valleys before the higher ridges.
        relaxation = dt * self.response.recovery
        decay = np.exp(-relaxation/.066)
        remaining = np.maximum(0, (np.abs(height)+.0018)*np.exp(-relaxation/.092)-.0018)
        pos[ids] = goal[ids] + decay*(delta-height[:,None]*self.normal)
        pos[ids] += (np.sign(height)*remaining)[:,None]*self.normal

    def constrain(self, pos):
        if self.skin_plane is None:
            return
        origin, normal = self.skin_plane
        depth = np.minimum(0, np.sum((pos[self.ids]-origin)*normal,axis=1))
        pos[self.ids] -= depth[:,None]*normal
