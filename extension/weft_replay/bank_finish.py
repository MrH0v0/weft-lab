"""Finish unloaded, compressed dry courses outside the mouse-move path."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import drawn_courses,yarn_chart
from .rim_relief import relax,smooth_joined_banks

_pool=None


def solve(points,faces,knit,rims,open_ends,stock,skin,protected,values,seed_ids,
          update_banks,eyes,joined,run):
    changed=False
    if joined:
        from .open_seam import closest_tips
        corridors=[]
        for bank,_ in joined:
            for other,_ in eyes:
                if other is bank:continue
                a,b=closest_tips(points,knit,bank,other)
                if np.linalg.norm(points[a]-points[b])<.065:corridors.append((a,b))
        changed=smooth_joined_banks(points,faces,joined,skin,protected,corridors)
    if update_banks:
        if seed_ids is not None:
            from .open_seam import ClothEdges
            cycles=ClothEdges(faces,len(points)).cycles
            rims=[cycle for cycle in cycles if np.isin(cycle,seed_ids).any()]
        previous=None if values is None else np.clip(np.linalg.norm(values[:,:2],axis=1),0.,1.)
        region=relax(points,faces,knit,rims,skin,open_ends,protected,previous)
        if region:
            values=yarn_chart.fit(points,knit,stock,region,faces);changed=False
    if len(eyes)>1:
        fabric=drawn_courses.mapped_fabric(points,knit,eyes,run,stock)
        if values is None and fabric[:,0].any():
            values=np.zeros((len(points),3*len(yarn_chart.FIELDS)),'f')
            changed=True
        if values is not None:values[:,12:]=fabric
    elif values is not None:values[:,12:]=0.
    if values is None and not changed:return None
    if changed and values is not None:
        seated=np.linalg.norm(values[:,3:6],axis=1)>1.e-5
        values[seated,6:9]=points[seated]
        values[:,9:12]=yarn_chart.neutral_normals(points,faces)
    simple=all(len(owners)==1 and not owners[0].get('open_arc') for _,owners in eyes)
    if len(eyes)>1 and simple and values is not None:
        strength=np.linalg.norm(values[:,3:6],axis=1)
        ids=np.flatnonzero(strength>.0001)
        if len(ids):
            _,valid,stock_normal=yarn_chart.stock_offset(points[ids],knit[ids],stock,shading=True)
            # Where a released course already follows the leg, retain its
            # own smooth normal. A crushed fan follows the posed, uncut nylon.
            # Flat collision-face normals gave whole panels a dark shoulder.
            neutral=values[ids,9:12]
            tilt=1.-np.einsum('ij,ij->i',neutral,stock_normal)
            repair=drawn_courses.smooth((tilt-.018)/.16)[:,None]
            direction=neutral+repair*(stock_normal-neutral)
            direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1.e-12)
            values[ids[valid],3:6]=direction[valid]*strength[ids[valid],None]
    return points,values


def schedule(tool,dg,protected=(),*,immediate=False):
    global _pool
    sim=tool.sim
    if tool.wet_mode or not sim.cracks or (sim.pose is not None and tool.speed):return False
    count=len(sim.fracture_events)
    update_banks=getattr(sim,'bank_style_count',-1)!=count
    if not update_banks and getattr(sim,'drawn_style_count',-1)==count:return False
    job=getattr(tool,'bank_job',None)
    if job is not None:
        if job[0] is sim and job[1] is sim.faces and job[2]==count:return False
        job[-1].cancel();tool.bank_job=None
    source=tool.obj['weft_live_source']
    if not hasattr(tool,'bank_stock'):
        tool.bank_stock=yarn_chart.reference(source,dg)
    if tool.bank_stock is None:return False
    sheet=hasattr(sim,'corner_uv')
    if sheet:
        points=sim.rest.copy();knit=sim.knit
        rims=[c['rim'].copy() for c in sim.cracks]
        open_ends={int(c['rim'][0]) for c in sim.cracks if c.get('notch')}
        seed=None;skin=getattr(getattr(sim,'clearance',None),'__self__',None)
    else:
        from .fracture import rim_slice
        points=sim.relaxed_tears().copy();knit=sim.uv*[144.,1.]
        seed=np.unique(np.concatenate([c['rim_ids'][rim_slice(c)].ravel() for c in sim.cracks]))
        rims=[];open_ends=set()
        if not hasattr(tool,'bank_skin'):
            contact=getattr(sim,'monitor_contact',None)
            from .seam_contact import FixedSkin
            tool.bank_skin=FixedSkin(contact[1]) if contact is not None else None
        skin=tool.bank_skin
    values=getattr(sim,'bank_weave',None)
    if values is not None:
        # The worker receives an immutable snapshot of the event's domain.
        data=np.zeros((len(points),3*len(yarn_chart.FIELDS)),'f')
        n=min(len(points),len(values));data[:n,:values.shape[1]]=values[:n];values=data
    eyes=drawn_courses.samples(sim)
    done=set(getattr(sim,'rounded_joins',()))
    joined=[]
    for ring,owners in eyes:
        if len(owners)>1:
            key='+'.join(str(k) for k in sorted(s['id'] for s in owners))
            if key not in done:joined.append((ring,owners[0].get('open_arc',False)));done.add(key)
    if _pool is None:_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Weft dry courses')
    future=_pool.submit(solve,points,sim.display_faces().copy(),knit.copy(),rims,open_ends,
                        tool.bank_stock,skin,np.asarray(protected,dtype=int),values,seed,
                        update_banks,eyes,joined,sim.response.run)
    tool.bank_job=(sim,sim.faces,count,done,future)
    if immediate:return collect(tool,wait=True)
    return False


def collect(tool,*,wait=False):
    job=getattr(tool,'bank_job',None)
    if job is None:return False
    sim,faces,count,done,future=job
    if not wait and not future.done():return False
    tool.bank_job=None
    if tool.sim is not sim or sim.faces is not faces or sim.grip is not None or len(sim.fracture_events)!=count:
        future.cancel()
        return False
    result=future.result()
    sim.bank_style_count=count
    mesh=tool.obj.data
    mesh[yarn_chart.STAMP]=count
    sim.drawn_style_count=count;sim.rounded_joins=done
    mesh[drawn_courses.STAMP]=count;mesh[drawn_courses.JOINS]=' '.join(sorted(done))
    if result is None:return False
    rest,sim.bank_weave=result
    sheet=hasattr(sim,'corner_uv')
    if not sheet:sim.bank_rest_owner=sim.rest
    neutral=sim.rest if sheet else sim.relaxed_tears()
    delta=rest-neutral
    # A continuing release retains its remaining recoil. A settled sheet
    # adopts the new neutral at once, including after opening an older file.
    sim.pos+=delta
    neutral[:]=rest
    if sheet:
        sim.refresh_layout()
        from .continue_view import publish_yarns
        publish_yarns(tool.obj,sim)
        mesh.vertices.foreach_set('co',sim.pos.astype('f').ravel())
    else:
        from .live_view import write_positions
        write_positions(mesh,sim)
    if sim.bank_weave is not None:
        material=mesh.materials[0];yarn_chart.bind(material)
        if sim.bank_weave[:,12].any() or material.node_tree.nodes.get(drawn_courses.BAND):
            drawn_courses.bind(material,tool.obj['weft_live_source'].get('weft_visual_knit_scale',1.))
        yarn_chart.write(mesh,sim.bank_weave)
    sections=getattr(sim,'sections',None)
    if sections is not None:
        for part in sections.active:yarn_chart.write(part.obj.data,sim.bank_weave,part.ids)
    mesh.update();tool.region.tag_redraw()
    return True


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True,cancel_futures=True);_pool=None
