"""Prepare the warned knit lane from immutable material coordinates while the hand loads it."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from .knit_patch import PinchPatch
from .tear_strip import plan_strip,strip_seating

_pool=None


def prepare(source,center,uv,axis,phase,directions):
    plan=plan_strip(source,center,uv,axis,phase)
    if plan is None:return None,None,None
    parents,weights,first,faces,rails,station=plan
    # All gauges use material coordinates. Only their posed sample and
    # shared yarn ledger have to be bound again when rupture actually occurs.
    def extended(values,chart=False):
        triangles=values[parents].copy()
        if chart:triangles[:,:,0]-=np.round(triangles[:,:,0]-triangles[:,:1,0])
        new=np.einsum('ni,nij->nj',weights,triangles)
        return np.concatenate((values,new,new[first[1:-1]-len(values)]))
    rest=extended(source.rest);chart=extended(source.uv,True)
    from .live_membrane import grip_weights
    from .fracture import tube_radius
    grip_data=grip_weights(rest,source.anchor,source.grip)
    weight=grip_data[-1]
    fitted=slice(0,source.fitted_vertices)
    radius=tube_radius(rest[fitted],chart[fitted],uv[1])
    cloth=SimpleNamespace(rest=rest,uv=chart,anchor=source.anchor,normal=source.normal,axes=source.axes,
              dry=True,response=source.response,skin_planes=source.skin_planes,yarn_first_row=source.yarn_first_row,
              yarn_ledger=np.zeros(source.shape),failed_cells=np.zeros(source.shape,bool))
    from .live_sections import layout_groups
    from .contact_index import edge_arrays
    all_banks=[c['banks'] for c in source.cracks]+[rails]
    groups=layout_groups(rest,faces,all_banks,source.anchor)
    weld=np.arange(len(rest),dtype='i')
    closed=[c['closed'] for c in source.cracks]+[rails[abs(station)>=.5]]
    for pair in closed:weld[pair[:,1]]=pair[:,0]
    shown=weld[faces]
    corner_uv=chart[shown].copy()
    corner_uv[:,:,0]-=np.round(corner_uv[:,:,0]-corner_uv[:,:1,0]);corner_uv[:,:,0]*=144.
    arrays=edge_arrays(shown,len(rest))
    from .live_view import make_edge_cache
    cache=make_edge_cache(arrays) if arrays is not None else None
    toe=None
    if source.toe is not None:
        v=source.toe;r=np.clip((rest@v[:3]-v[3])/(v[4]-v[3]),0,1);toe=(r*r*(3-2*r)).astype('f')
    uvs=corner_uv.astype('f')
    # A later unloop can change only corners on the second side of a seam.
    # These rows and their section-local indices are known before rupture.
    right=np.zeros(len(rest),bool)
    for bank in all_banks:right[bank[:,1]]=True
    changing=right[faces].any(1)
    bank_rows=np.flatnonzero(changing)
    sections={k:(shown[rows],np.flatnonzero(changing[rows])) for k,(_,rows,ids,remap) in groups.items()}
    tiles={}
    cut=(faces>=len(source.rest)).any(1)
    for k,(_,rows,ids,remap) in groups.items():
        if not cut[rows].any():continue
        tri=remap[shown[rows]];edges=edge_arrays(tri,len(ids))
        if edges is not None:tiles[k]=(tri,uvs[rows],edges,make_edge_cache(edges))
    seating=None
    if source.project_skin is not None:
        reference=SimpleNamespace(rest=rest,faces=faces,seating=extended(source.seating),project_skin=source.project_skin)
        seating=strip_seating(reference,faces[cut])
    support=None
    if not source.cracks:
        from .live_membrane import Membrane,unit
        up=source.normal;local=directions[axis]
        across=unit(local-(local@up)*up);along=unit(directions[1-axis])
        pair=rails[1:-1]
        c=dict(center=center,uv=uv,across=across,along=along,normal=up,load_axis=axis,curve_phase=phase,
               rim_ids=rails,rim_axis=station,banks=pair,locations=station[1:-1],fronts=np.array([-.5,.5]),
               pitch=np.linalg.norm(rest[rails[-1,0]]-rest[rails[0,0]])/np.ptp(station))
        before=set(c)
        material=object.__new__(Membrane)
        material.rest=material.pos=rest;material.faces=faces;material.uv=chart;material.dry=True;material.pose=None
        material.normal=up;material.cracks=[c];material.tear_support_faces=material.strain_cache=None
        material.set_tear_support(c)
        material.tear_frame(material.current_tear_faces())
        material.tear_contact_band()
        frame={k:getattr(material,k) for k in ('tear_support_faces','strain_cache','material_frame','weld_cache',
                                               'bank_map','local_faces_cache','contact_normals','contact_material_band','contact_band')}
        from .seam_contact import held_face_groups
        frame['held_contact']=(faces,held_face_groups(rest,faces))
        support=(uv,across,along,{k:c[k] for k in c.keys()-before},frame)
    gauge_faces=faces[np.max(weight[faces],axis=1)>.04]
    return plan,PinchPatch(cloth,gauge_faces),(faces,groups,shown,uvs,arrays,cache,rest,toe,tiles,seating,support,chart,bank_rows,sections,grip_data,radius)


def begin(sim,indices,tension,axis):
    global _pool
    strongest=indices[np.argmax(tension[indices])]
    previous=getattr(sim,'warning_material',None)
    if (getattr(sim,'warned_lanes',None) is not None and previous is not None and
            previous.faces is sim.faces and previous.response==sim.response and
            np.array_equal(sim.anchor,previous.anchor) and np.array_equal(sim.grip,previous.grip)):
        follow(sim,strongest,axis)
        prefetch_neighbour(sim,indices,tension,strongest,axis)
        return
    cancel(sim)
    if _pool is None:_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='Weft warned lane')
    source=SimpleNamespace(rest=sim.rest,uv=sim.uv,faces=sim.faces,fitted_vertices=sim.fitted_vertices,grip=sim.grip.copy(),anchor=sim.anchor.copy(),
            axes=sim.axes.copy(),normal=sim.normal.copy(),response=sim.response,yarn_first_row=sim.yarn_first_row,
            skin_planes=getattr(sim,'skin_planes',None),seating=sim.seating.copy(),project_skin=getattr(sim,'project_skin',None),toe=getattr(sim,'toe_finish',None),shape=sim.failed_cells.shape,cracks=[dict(rim_ids=c['rim_ids'],banks=c['banks'],closed=sim.closed_banks(c)) for c in sim.cracks])
    sim.warning_material=source
    sim.warned_lanes=sim.faces,{}
    follow(sim,strongest,axis)
    prefetch_neighbour(sim,indices,tension,strongest,axis)


def submit_plan(sim,index,axis):
    uv=(sim.patch.loops.cells[index]+.5)/[144.,1.]
    return _pool.submit(prepare,sim.warning_material,sim.patch.rest_centers[index].copy(),uv,axis,
                        float(uv@[84.4,1.13]),sim.patch.directions[index].copy())


def prefetch_neighbour(sim,indices,tension,peak,axis):
    # Near-tied adjacent stitches can exchange the strain maximum before the
    # next pull. Prepare their immutable material on two workers. Only the
    # measured winner may be staged or cut.
    layout,jobs=sim.warned_lanes
    if sim.warning is not None or len(jobs)>=2:return
    cells=sim.patch.loops.cells
    near=indices[np.abs(cells[indices]-cells[peak]).sum(axis=1)==1]
    if not len(near):return
    index=near[np.argmax(tension[near])]
    if tension[index]<.8*tension[peak]:return
    cell=cells[index];key=(axis,int(cell[0]),int(cell[1]))
    if key not in jobs:sim.warned_lanes=(layout,{**jobs,key:submit_plan(sim,index,axis)})


def follow(sim,index,axis):
    # Advance display priority with the measured peak without discarding the
    # previous candidate while it can still be loaded.
    layout,jobs=sim.warned_lanes
    cell=sim.patch.loops.cells[index]
    key=(axis,int(cell[0]),int(cell[1]))
    if next(iter(jobs),None)==key:return
    job=jobs.get(key)
    if job is None:
        job=submit_plan(sim,index,axis)
    ordered={key:job}
    for existing,future in jobs.items():
        if existing==key:continue
        if len(ordered)<3:ordered[existing]=future
        elif not future.done():future.cancel()
    sim.warned_lanes=layout,ordered


def get(sim,uv,axis):
    lookup=getattr(sim,'warned_lanes',None)
    if lookup is None or lookup[0] is not sim.faces:return None
    key=(axis,int(np.floor(uv[0]*144)),int(np.floor(uv[1])))
    job=lookup[1].get(key)
    if job is None or job.cancelled():return None
    # At actual failure this matching in-flight plan is exactly the one a
    # synchronous retry would build. Let it finish instead of running a
    # second triangulation and competing with the drawing thread.
    return job.result()


def cancel(sim):
    previous=getattr(sim,'warned_lanes',None)
    if previous is not None:
        # A trial contact shares the worker's immutable predicted lane jobs.
        # Only the live solver owns and cancels those futures.
        if not getattr(sim,'predicting_birth',False):
            for job in previous[1].values():job.cancel()
        sim.warned_lanes=None


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True,cancel_futures=True);_pool=None
