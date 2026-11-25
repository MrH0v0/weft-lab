"""Fixed-pose dry fabric continued from the displayed, already torn mesh."""
import numpy as np

from .knit_patch import surface_normals
from .open_seam import ClothEdges,OpenSeam,bridge_at,chart_offset,closest_tips,cut_fans,material_path,normalized,smooth


class TornSheet:
    pose=None
    dry=True
    warning=None
    patch=None

    def __init__(self,points,faces,uv,readings,response,neutral=None):
        self.rest=(neutral if neutral is not None else points).astype(float,copy=True)
        self.pos=points.astype(float,copy=True)
        self.faces=faces.copy()
        self.knit=uv.copy()
        self.response=response
        self.fracture_events=list(readings.get('events',[]))
        self.elapsed=float(readings.get('t_s',0))
        self.step_number=int(readings.get('step',0))
        self.sources={}
        self.grip=None;self.plan=None;self.moving=False
        self.changed=False
        self.moving=bool(abs(self.pos-self.rest).max()>.00004)
        self.refresh_layout()
        available=list(self.layout.cycles)
        for hole in readings['holes']:
            seeds=hole.get('rim_seeds')
            if seeds:
                for anchor in seeds:
                    self.sources[int(anchor['id'])]=dict(anchor)
                continue
            center=np.asarray(hole['material_uv'])*[144.,1.]
            def error(loop):
                return np.linalg.norm(chart_offset(self.knit[loop],center),axis=1).min()
            if not available:raise ValueError('快照的孔缘与网格边界不匹配')
            loop=min(available,key=error)
            if error(loop)>4.:raise ValueError('在 Knit 坐标中未找到记录的孔缘')
            available=[x for x in available if x is not loop]
            vertex=int(loop[np.argmin(np.linalg.norm(chart_offset(self.knit[loop],center),axis=1))])
            self.sources[hole['id']]=dict(id=hole['id'],vertex=vertex,material_uv=hole['material_uv'],
                                          cells=list(hole['cells']),run_axis=hole['run_axis'])
        from .seam_boundary import factory_cuff
        saved_cuff=readings.get('cuff_vertices')
        self.cuff=set(map(int,saved_cuff if saved_cuff is not None else
                      factory_cuff(self.layout.cycles,self.knit,(s['vertex'] for s in self.sources.values()))))
        self.identify()
        if len(self.cracks)!=len(readings['holes']):raise ValueError('快照的孔数与已保存网格不一致')

    def refresh_layout(self):
        self.layout=ClothEdges(self.faces,len(self.rest))
        self.normals=surface_normals(self.rest,self.faces)
        if hasattr(self,'support_normals'):return
        # Old rims can have tiny rolled shoulders. Let the stocking slide
        # in the tangent plane of the surrounding leg, not of each shoulder.
        a,b=self.layout.edges.T
        src=np.r_[a,b];dst=np.r_[b,a]
        count=np.maximum(1,np.bincount(dst,minlength=len(self.rest)))[:,None]
        normals=self.normals.copy()
        for _ in range(6):
            around=np.column_stack([np.bincount(dst,weights=normals[src,i],minlength=len(normals)) for i in range(3)])
            normals=.30*normals+.70*around/count
        self.support_normals=normalized(normals)

    def identify(self):
        by_vertex={a['vertex']:a['id'] for a in self.sources.values()}
        self.cracks=[]
        from .seam_boundary import torn_arcs
        for cycle in self.layout.cycles:
            for bank,notch in torn_arcs(cycle,self.cuff):
                members=sorted({by_vertex[int(v)] for v in bank if int(v) in by_vertex})
                if members:self.cracks.append(dict(id=members[0],members=members,rim=bank,notch=notch))
        self.cracks.sort(key=lambda c:c['id'])

    @property
    def uv(self):
        return self.knit/[144.,1.]

    @property
    def failed_cells(self):
        first=int(np.floor(self.knit[:,1].min()))
        cells=np.zeros((int(np.floor(self.knit[:,1].max()))-first+1,144),bool)
        for source in self.sources.values():
            for x,y in source['cells']:
                if 0<=y-first<len(cells):cells[y-first,x%144]=True
        return cells

    def display_faces(self):
        return self.faces

    def display_positions(self):
        return self.pos

    def duplicate(self,ids):
        new=np.arange(len(self.rest),len(self.rest)+len(ids),dtype='i')
        self.rest=np.vstack((self.rest,self.rest[ids]))
        self.pos=np.vstack((self.pos,self.pos[ids]))
        self.knit=np.vstack((self.knit,self.knit[ids]))
        if getattr(self,'bank_weave',None) is not None:self.bank_weave=np.vstack((self.bank_weave,self.bank_weave[ids]))
        if hasattr(self,'material_reference'):
            old=self.material_reference;self.material_reference=np.vstack((old,old[ids]))
        self.support_normals=np.vstack((self.support_normals,self.support_normals[ids]))
        self.cuff.update(int(dst) for src,dst in zip(ids,new) if int(src) in self.cuff)
        if getattr(self,'toe',None) is not None:self.toe=np.r_[self.toe,self.toe[ids]]
        return new

    def project_bridge(self,point):
        return bridge_at(self,point)

    def width(self,crack,across):
        return float(np.ptp(self.rest[crack['rim']]@across))

    def choose_path(self,point,face):
        from .seam_boundary import nearest_edge
        cuff_dist,cuff_v=min((nearest_edge(self.rest,loop,point,self.cuff) for loop in self.layout.cycles),
                             key=lambda entry:entry[0],default=(np.inf,None))
        old=min(self.cracks,key=lambda c:np.linalg.norm(self.rest[c['rim']]-point,axis=1).min(),default=None)
        rim=old['rim'] if old is not None else np.empty(0,dtype='i')
        near=rim[np.argmin(np.linalg.norm(self.rest[rim]-point,axis=1))] if len(rim) else None
        attached=near is not None and np.linalg.norm(self.rest[near]-point)<.026
        # When the narrow ligament up to the cuff is pulled, open through it
        # before considering a second eye further down the same wale.
        if attached and not old.get('notch') and cuff_dist<.020:
            edge_point=self.rest[cuff_v]
            bank=int(rim[np.argmin(np.linalg.norm(self.rest[rim]-edge_point,axis=1))])
            if np.linalg.norm(self.rest[bank]-edge_point)<.036:
                return bank,cuff_v,np.array([.009,.011]),old
        pair=self.project_bridge(point)
        if pair:
            first,other,a,b=pair
            across=normalized(np.cross(self.normals[a],self.rest[b]-self.rest[a]))
            widths=np.clip(.41*np.array([self.width(first,across),self.width(other,across)]),.006,.022)
            return a,b,widths,first
        if attached:
            seed=self.sources[old['members'][0]]
            axis=1 if seed['run_axis']=='wale' else 0
            chart=chart_offset(self.knit[rim],np.asarray(seed['material_uv'])*[144.,1.])
            tips=rim[[np.argmin(chart[:,axis]),np.argmax(chart[:,axis])]]
            if old.get('notch'):
                # Extend the inside tip. Both ends of this arc are on the cuff.
                mouth=self.rest[rim[[0,-1]]].mean(0)
                a=int(rim[np.argmax(np.linalg.norm(self.rest[rim]-mouth,axis=1))])
                mouth_uv=self.knit[rim[0]]
                sign=1 if chart_offset(self.knit[a],mouth_uv)[axis]>=0 else -1
            else:
                end=int(np.argmin(np.linalg.norm(self.rest[tips]-point,axis=1)))
                a=int(tips[end]);sign=(-1,1)[end]
                d,edge=min((nearest_edge(self.rest,loop,self.rest[a],self.cuff) for loop in self.layout.cycles),
                           key=lambda item:item[0],default=(np.inf,None))
                if d<.030 and edge is not None:
                    return a,edge,np.array([.009,.011]),old
        elif cuff_dist<.012:
            a=cuff_v;axis=1
            around=np.linalg.norm(self.rest-self.rest[a],axis=1)<.039
            interior=around&~self.layout.border
            sign=1 if np.median(self.knit[interior,axis])>self.knit[a,axis] else -1
        else:
            a=int(self.faces[face][np.argmin(np.linalg.norm(self.rest[self.faces[face]]-point,axis=1))])
            axis=1;sign=1
        def station(start,steps):
            off=chart_offset(self.knit,self.knit[start])
            goal=np.zeros(2);goal[axis]=steps
            distance=np.linalg.norm(self.rest-self.rest[start],axis=1)
            valid=(distance>.005)&(distance<.066)&self.layout.used&(~self.layout.border)&(self.support_normals@self.support_normals[start]>.42)
            valid&=off[:,axis]*np.sign(steps)>.5
            ids=np.flatnonzero(valid)
            if not len(ids):return None
            return int(ids[np.argmin(np.linalg.norm(off[ids]-goal,axis=1))])
        b=station(a,sign*14.)
        if b is None and not attached and cuff_dist>=.012:b=station(a,-14.)
        if b is None:return None
        if not attached and cuff_dist>=.012:
            back=station(a,-np.sign(chart_offset(self.knit[b],self.knit[a])[axis])*5.)
            if back is not None:a=back
        return a,b,np.array([.011,.009]),old if attached else None

    def pick(self,face,point):
        self.release()
        self.held_face=face;self.grip=self.faces[face].copy()
        tri=self.pos[self.grip]
        ab=np.linalg.lstsq((tri[1:]-tri[0]).T,point-tri[0],rcond=None)[0]
        self.bary=np.r_[1.-ab.sum(),ab]
        self.anchor=self.bary@self.rest[self.grip]
        self.normal=normalized(self.bary@self.normals[self.grip])
        local_chart=chart_offset(self.knit[self.grip],self.knit[self.grip[0]])
        self.anchor_uv=(self.knit[self.grip[0]]+self.bary@local_chart)/[144.,1.]
        self.axes=np.linalg.lstsq(local_chart[1:],tri[1:]-tri[0],rcond=None)[0]
        choice=self.choose_path(self.anchor,face)
        self.plan=None;self.peak_travel=0.;self.paid=0.;self.pointer=np.zeros(3);self.recoil=1.
        self.affected=None;self.seed=None;self.parent=None
        if choice:
            a,b,widths,owner=choice
            path=material_path(self,a,b)
            last_link=(path is not None and len(path)==2 and owner is not None and
                       not hasattr(self,'contact_pose') and any(c is not owner and b in c['rim'] for c in self.cracks))
            if path is not None and (len(path)>=3 or last_link):
                fans=cut_fans(self.faces,path,self.layout.border)
                if fans is not None:
                    self.plan=OpenSeam(self,path,fans,widths,bool(self.layout.border[b]))
                    self.parent=owner['id'] if owner is not None else None
        self.moving=True

    def release(self):
        if getattr(self,'pending',None) is not None:
            self.pending.cancel();self.pending=None
        if self.grip is not None:self.moving=True
        self.grip=None;self.plan=None;self.affected=None

    def step(self,dt,delta=None,*,settle=False):
        self.elapsed+=dt;self.step_number+=1
        self.changed=False
        if self.grip is None:
            if not self.moving:return False
            diff=self.pos-self.rest
            self.pos=self.rest+diff*np.exp(-min(dt,.12)*self.response.recovery/.063)
            self.moving=abs(self.pos-self.rest).max()>.00004
            if not self.moving:self.pos[:]=self.rest
            return True
        same=delta is None or np.array_equal(delta,self.pointer)
        if delta is not None:self.pointer=np.array(delta,copy=True)
        travel=np.linalg.norm(self.pointer)
        plan=self.plan
        if plan is not None and not plan.done and self.response.run>0:
            outward=max(0.,self.pointer@self.normal)
            tension=abs(self.pointer@plan.across)+.66*outward+.26*abs(self.pointer@plan.tangent)
            threshold=.013*self.response.break_extension/.8
            strength=.65+1.15*self.response.run
            before=max(0.,self.peak_travel-threshold)*strength
            self.peak_travel=max(self.peak_travel,tension)
            budget=max(0.,self.peak_travel-threshold)*strength
            # Pay out real courses over successive redraws, also when the
            # cursor stops after a fast pull. Easing off unloads the front.
            if not settle and tension>=.68*self.peak_travel:
                # A fast mouse stroke delivers work even before the next timer.
                self.paid=min(budget,self.paid+.48*(budget-before)+min(.008,.13*dt+.0014),plan.length+1e-9)
            if (settle or tension>=.68*self.peak_travel) and self.paid+1e-9>plan.stations[plan.reached]:
                old=plan.reached
                if getattr(self,'deferred',False):
                    from .seam_worker import advance
                    grew=advance(self,self.paid+1e-9,settle=settle)
                else:grew=plan.grow(self,self.paid+1e-9)
                if grew:
                    self.grip=self.faces[self.held_face].copy()
                    self.changed=True;self.affected=None
                    first_break=self.parent is None
                    if first_break and plan.copies:
                        vid=next(iter(plan.copies.values()))
                        self.parent=max(self.sources,default=0)+1
                        self.sources[self.parent]=dict(id=self.parent,vertex=int(vid),material_uv=(self.knit[vid]/[144.,1.]).tolist(),cells=[],run_axis='wale')
                    if self.parent is not None:
                        src=self.sources[self.parent]
                        cells=src['cells']
                        for v in plan.path[old:plan.reached+1]:
                            cell=np.floor(self.knit[v]).astype(int);cell[0]%=144
                            if cell.tolist() not in cells:cells.append(cell.tolist())
                        self.fracture_events.append(dict(kind='first_break' if first_break else 'unloop',
                            resumed=old==0 and not first_break,hole_id=self.parent,step=self.step_number,t_s=self.elapsed,
                            joined=bool(plan.joining and plan.done),cell=cells[-1]))
                    self.recoil=.65 if plan.done else .79
                    if not getattr(self,'deferred',False):self.refresh_layout()
                    self.identify()
        if same and self.affected is not None and not self.changed:return False
        if self.affected is None:
            relative=self.rest-self.anchor
            radius=np.linalg.norm(relative,axis=1)
            self.affected=np.flatnonzero(radius<.080)
            self.weights=np.exp(-.5*(radius[self.affected]/.028)**2)*smooth((.080-radius[self.affected])/.024)
            affected_faces=np.flatnonzero((radius[self.faces]<.080).any(1))
            self.hold_ids,local=np.unique(self.faces[affected_faces],return_inverse=True)
            from .tear_surface import strain_frame
            self.hold_frame=strain_frame(self.rest[self.hold_ids],local.reshape(-1,3))
            self.held_contact_patch=None
            if not hasattr(self,'contact_pose'):
                from .seam_overlap import LoadedContact
                self.held_contact_patch=LoadedContact(self.rest[self.hold_ids],local.reshape(-1,3))
            self.previous_hold_frame=None
            if plan and plan.copies:
                before=self.hold_ids.copy()
                for k,dup in plan.copies.items():before[before==dup]=plan.path[k]
                self.previous_hold_frame=strain_frame(plan.base[before],local.reshape(-1,3))
        slip=self.pointer-(self.pointer@self.normal)*self.normal
        slide=np.linalg.norm(slip)
        slip*=.022*np.tanh(slide/.022)/max(slide,1e-12)
        lift=.032*np.tanh((max(0.,self.pointer@self.normal)+.32*min(travel,.045))/.032)
        self.pos=self.rest.copy()
        motion=self.weights[:,None]*self.recoil*(slip+lift*self.normal)
        supports=self.support_normals[self.affected]
        # The hand lifts its own patch. On the opposite shoulder of a curved
        # shin its lateral motion can point into the leg; those courses slide.
        inward=np.minimum(0.,np.einsum('ij,ij->i',motion,supports))
        self.pos[self.affected]+=motion-inward[:,None]*supports
        from .tear_surface import limit_dry_held_travel,limit_held_travel
        limit=limit_held_travel if hasattr(self,'contact_pose') else limit_dry_held_travel
        local=self.pos[self.hold_ids];reference=self.rest[self.hold_ids]
        limit(local,reference,self.hold_frame)
        if self.previous_hold_frame:limit(local,reference,self.previous_hold_frame)
        if self.previous_hold_frame:
            tri,l,_,h,n,_=self.previous_hold_frame
            def folded(p):
                a,b,c=p[tri].transpose(1,0,2)
                return np.any(np.einsum('ij,ij->i',np.cross(b-a,c-a),n)<.003*l*h)
            if folded(local) and not folded(reference):
                # Ease only the hand displacement; the paid-out slit stays
                # open even at the narrowest material fan of a diagonal join.
                shift=local-reference
                low,high=0.,1.
                for _ in range(12):
                    mid=.5*(low+high)
                    if folded(reference+mid*shift):high=mid
                    else:low=mid
                local=reference+low*shift
        if self.held_contact_patch is not None:self.held_contact_patch.correct(local,reference)
        self.pos[self.hold_ids]=local
        return True

    def fracture_readings(self,uv=None):
        holes=[]
        for c in self.cracks:
            xyz=self.pos[c['rim']]
            centered=xyz-xyz.mean(0)
            area=np.linalg.norm(np.cross(centered,np.roll(centered,-1,axis=0)).sum(0))*.5
            _,_,basis=np.linalg.svd(centered,full_matrices=False)
            members=[self.sources[k] for k in c['members']]
            cells=sorted({tuple(v) for s in members for v in s['cells']})
            holes.append(dict(id=c['id'],material_uv=members[0]['material_uv'],cells=cells,
                run_axis=members[0]['run_axis'],gap_mm=float(np.ptp(centered@basis[1])*1000),
                area_mm2=float(area*1e6),rim_seeds=members,boundary_vertices=len(xyz)))
        return dict(holes=holes,events=self.fracture_events,step=self.step_number,t_s=self.elapsed,
                    cuff_vertices=sorted(self.cuff),
                    dressed_members=sorted(getattr(self,'dressed',())),
                    material=None,run_links=[],rim_spacing=[],failed_cells=sum(len(c['cells']) for c in holes),
                    damage_area_mm2=sum(c['area_mm2'] for c in holes))
