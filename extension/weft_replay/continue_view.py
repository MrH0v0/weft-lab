"""Resume a dry, posed Blender mesh at its visible free edges."""
import json
import time

import numpy as np
from .dry_fabric import TOE_FINISH_ATTRIBUTE
from .fracture import rim
from .live_rims import publish_paths
from .open_seam import normalized
from .seam_surface import TornSheet

NEUTRAL='Weft relaxed tear'
EDGE='Weft freed courses'


def snapshot_data(obj):
    note=obj.get('weft_snapshot')
    if note is None:raise ValueError('当前袜面没有封存的断口记录')
    data=json.loads(note.as_string())
    if data['mode']!='dry' or not data['fracture']['holes']:
        raise ValueError('请从已有断口的干态袜面接着撕')
    return data


def read_sheet(obj,readings,response,neutral=None):
    mesh=obj.data
    if len(mesh.loops)!=3*len(mesh.polygons) or 'Knit' not in mesh.uv_layers:
        raise ValueError('缺少续撕所需的三角面或 Knit 坐标')
    points=np.empty((len(mesh.vertices),3),'f')
    mesh.vertices.foreach_get('co',points.ravel())
    faces=np.empty((len(mesh.polygons),3),'i')
    mesh.loops.foreach_get('vertex_index',faces.ravel())
    corners=np.empty((faces.size,2),'f')
    mesh.uv_layers['Knit'].uv.foreach_get('vector',corners.ravel())
    uv=np.zeros((len(points),2))
    uv[faces.ravel()]=corners
    uv[:,0]%=144.
    saved=mesh.attributes.get(NEUTRAL)
    if saved and neutral is None:
        neutral=np.empty_like(points)
        saved.data.foreach_get('vector',neutral.ravel())
    sheet=TornSheet(points,faces,uv,readings,response,neutral)
    sheet.deferred=True
    sheet.corner_uv=corners
    from . import yarn_chart
    sheet.bank_weave=yarn_chart.read(mesh)
    if sheet.bank_weave is not None:
        if yarn_chart.FIELDS[3] not in mesh.attributes:
            shading_rest=sheet.rest
            if neutral is None:
                # An older working-session recovery can contain the raised
                # display but still carry the unloaded position of each bank.
                shading_rest=points.copy()
                known=np.linalg.norm(sheet.bank_weave[:,3:6],axis=1)>1.e-5
                shading_rest[known]=sheet.bank_weave[known,6:9]
            sheet.bank_weave[:,9:12]=yarn_chart.neutral_normals(shading_rest,sheet.faces)
            yarn_chart.write(mesh,sheet.bank_weave)
        yarn_chart.bind(mesh.materials[0])
    sheet.bank_style_count=int(mesh.get(yarn_chart.STAMP,-1))
    from . import drawn_courses
    sheet.drawn_style_count=int(mesh.get(drawn_courses.STAMP,-1))
    if drawn_courses.BAND in mesh.attributes and drawn_courses.SPAN not in mesh.attributes:
        sheet.drawn_style_count=-1
    sheet.rounded_joins=set(mesh.get(drawn_courses.JOINS,'').split())
    toe=mesh.attributes.get(TOE_FINISH_ATTRIBUTE)
    sheet.toe=None
    if toe:
        sheet.toe=np.empty(len(points),'f')
        toe.data.foreach_get('value',sheet.toe)
    sheet.yarn_ends_obj=next((c for c in obj.children if c.get('weft_yarn_ends') and c.type=='CURVE' and 'loose yarn' in c.name),None)
    if sheet.yarn_ends_obj:
        from .live_rims import update_yarn_appearance
        update_yarn_appearance(sheet,obj['weft_live_source'])
    sheet.original_yarns={}
    if sheet.yarn_ends_obj and sheet.cracks:
        all_ids=np.concatenate([c['rim'] for c in sheet.cracks])
        owner=np.concatenate([np.full(len(c['rim']),c['id']) for c in sheet.cracks])
        for spline in sheet.yarn_ends_obj.data.splines:
            p=np.empty((len(spline.points),4),'f');r=np.empty(len(p),'f')
            spline.points.foreach_get('co',p.ravel());spline.points.foreach_get('radius',r)
            near=np.linalg.norm(p[:,None,:3]-points[all_ids][None],axis=2).argmin(1)
            key=int(owner[near[len(near)//2]])
            ids=all_ids[near]
            sheet.original_yarns.setdefault(key,[]).append((ids,p[:,:3]-points[ids],r))
    sheet.dressed=set(readings.get('dressed_members',())).intersection(sheet.sources)
    sheet.new_yarn=set()
    mat=mesh.materials[0]
    sheet.run_image=next((n.image for n in mat.node_tree.nodes if n.label=='Surviving wale and course yarn' and n.image),None)
    if sheet.run_image:
        w,h=sheet.run_image.size
        sheet.run_pixels=np.empty((h,w,4),'f')
        sheet.run_image.pixels.foreach_get(sheet.run_pixels.ravel())
        sheet.first_row=int(sheet.run_image.get("weft_first_course", np.floor(corners[:,1].min())))
    return sheet


def contact(sheet,collider):
    from .seam_contact import FixedSkin
    skin=FixedSkin(collider[1])
    sheet.held_clearance=skin.ray_gaps
    sheet.held_face_lifts=skin.index.wet_face_lifts if skin.index is not None else None
    sheet.clearance=skin.clearance
    sheet.seat=skin.seat


def bridge_middle(sim,point):
    banks=[rim(c,sim.pos).reshape(-1,3) for c in sim.cracks]
    for i,one in enumerate(banks):
        for other in banks[i+1:]:
            dist=np.linalg.norm(one[:,None]-other,axis=2)
            a,b=np.unravel_index(dist.argmin(),dist.shape)
            edge=other[b]-one[a];length=np.linalg.norm(edge)
            if .005<length<.142:
                t=(point-one[a])@edge/length**2
                if .18<t<.82 and np.linalg.norm(point-one[a]-t*edge)<min(.013,.006+.11*length):
                    return True
    return False


def adopt(tool,point):
    old=tool.sim
    if tool.monitor is not None or (old.pose is not None and not getattr(old,'fixed_wet_pose',False)):return False
    from .open_seam import ClothEdges
    from .seam_boundary import factory_cuff,nearest_edge
    if not hasattr(old,'factory_opening'):
        layout=ClothEdges(old.display_faces(),len(old.pos))
        old.factory_opening=factory_cuff(layout.cycles,old.uv*[144.,1.])
    near_cuff=nearest_edge(old.pos,old.factory_opening,point)[0]<.012
    if not near_cuff:
        for crack in old.cracks:
            lips=rim(crack,old.pos).reshape(-1,3)
            tip=lips[np.argmin(np.linalg.norm(lips-point,axis=1))]
            if np.linalg.norm(tip-point)<.018 and nearest_edge(old.pos,old.factory_opening,tip)[0]<.030:
                near_cuff=True
                break
    if not near_cuff and (len(old.cracks)<2 or not bridge_middle(old,point)):return False
    from .atlas_transfer import flush
    flush(old)
    neutral=old.relaxed_tears() if old.cracks else old.rest+old.seating
    try:sheet=read_sheet(tool.obj,old.fracture_readings(),old.response,neutral)
    except ValueError:return False
    if not near_cuff and sheet.project_bridge(point) is None:return False
    if getattr(old,'fixed_wet_pose',False):
        pose=old.pose
        contact(sheet,(pose.tree,pose.skin_vertices[pose.skin_ids]))
        sheet.contact_pose=pose;sheet.yarn_first_row=old.yarn_first_row
        sheet.material_reference=(tool.wet.reference if tool.wet is not None else old.rest).copy()
        if tool.wet is not None:
            tool.wet.sim=sheet;tool.wet.rebind()
            tool.wet_display.publish()
    elif getattr(old,'monitor_contact',None):
        contact(sheet,old.monitor_contact)
    flush(old,close=True)
    if hasattr(old,'loop_image'):old.loop_image.pack()
    if getattr(old,'run_image',None):old.run_image.pack()
    from .live_skin import restore_skin
    from .knit_tangent import finish
    finish(old,tool.obj.data)
    if getattr(old,'sections',None) is not None:
        old.sections.close();old.sections=None
    restore_skin(old)
    tool.sim=sheet;tool.continued=True
    return True


def store_neutral(mesh,sheet):
    attr=mesh.attributes.get(NEUTRAL) or mesh.attributes.new(NEUTRAL,'FLOAT_VECTOR','POINT')
    attr.data.foreach_set('vector',sheet.rest.astype('f').ravel())


def topology(mesh,sheet):
    # Face and corner order stay fixed. Retain the original corner-level
    # chart, including the two different U values at the wrap of the leg.
    mesh.clear_geometry()
    mesh.vertices.add(len(sheet.pos))
    mesh.vertices.foreach_set('co',sheet.pos.astype('f').ravel())
    mesh.loops.add(sheet.faces.size)
    mesh.attributes['.corner_vert'].data.foreach_set('value',sheet.faces.ravel())
    mesh.polygons.add(len(sheet.faces))
    mesh.polygons.foreach_set('loop_start',np.arange(0,sheet.faces.size,3,dtype='i'))
    mesh.polygons.foreach_set('use_smooth',np.ones(len(sheet.faces),bool))
    mesh.update(calc_edges=True)
    uv=mesh.uv_layers.get('Knit') or mesh.uv_layers.new(name='Knit')
    uv.uv.foreach_set('vector',sheet.corner_uv.ravel())
    if getattr(sheet,'bank_weave',None) is not None:
        from .yarn_chart import write
        write(mesh,sheet.bank_weave)
    if sheet.toe is not None:
        attr=mesh.attributes.get(TOE_FINISH_ATTRIBUTE) or mesh.attributes.new(TOE_FINISH_ATTRIBUTE,'FLOAT','POINT')
        attr.data.foreach_set('value',sheet.toe)
    mesh.update()


def shade_edge(obj,sheet):
    mesh=obj.data;material=mesh.materials[0];nodes=material.node_tree.nodes
    if nodes.get(EDGE) is None:
        from .sheer_optics import math_socket
        tree=material.node_tree
        shader=nodes.get('Principled BSDF')
        if shader is None or not shader.inputs['Alpha'].is_linked:return
        old=shader.inputs['Alpha'].links[0].from_socket
        attr=nodes.new('ShaderNodeAttribute');attr.name=EDGE;attr.attribute_name=EDGE
        knit=nodes.new('ShaderNodeUVMap');knit.uv_map='Knit'
        split=nodes.new('ShaderNodeSeparateXYZ');tree.links.new(knit.outputs['UV'],split.inputs[0])
        x,y=split.outputs[:2]
        def op(kind,*p):return math_socket(tree,kind,*p)
        lane=op('MULTIPLY_ADD',op('SINE',op('MULTIPLY',y,.49)),.22,x)
        swing=op('MULTIPLY_ADD',op('SINE',op('MULTIPLY',lane,2.1)),.34,y)
        def fiber(q,pitch,power):
            wave=op('MULTIPLY_ADD',op('COSINE',op('MULTIPLY',q,pitch)),.5,.5)
            return op('POWER',wave,power)
        wales=fiber(lane,9.4,3.5)
        rungs=fiber(swing,6.1,7.5)
        # Keep separate twisted wales and occasional course heads in the
        # translucent band on the fabric side of the freshly broken ends.
        stock=op('MULTIPLY_ADD',wales,.52,op('MULTIPLY_ADD',rungs,.26,.09))
        opacity=op('SUBTRACT',1.,op('MULTIPLY',attr.outputs['Fac'],op('SUBTRACT',1.,stock)))
        tree.links.new(op('SUBTRACT',1.,op('POWER',op('MAXIMUM',.0001,op('SUBTRACT',1.,old)),opacity)),shader.inputs['Alpha'])
    rims=[c['rim'] for c in sheet.cracks if set(c['members'])&sheet.dressed]
    if not rims:return
    source=np.concatenate(rims)
    dist=np.full(len(sheet.rest),.016);dist[source]=0.
    a,b=sheet.layout.edges.T
    lo=sheet.rest[source].min(0)-.017;hi=sheet.rest[source].max(0)+.017
    keep=((sheet.rest[a]>lo)&(sheet.rest[a]<hi)).all(1)
    a,b=a[keep],b[keep]
    lengths=np.linalg.norm(sheet.rest[a]-sheet.rest[b],axis=1)
    for _ in range(16):
        old=dist.copy()
        np.minimum.at(dist,a,old[b]+lengths);np.minimum.at(dist,b,old[a]+lengths)
        if np.array_equal(old,dist):break
    x,y=sheet.knit.T
    grain=.5+.5*np.sin(.58*y+.93*np.sin(.31*x))
    lane=.5+.5*np.sin(1.83*x-.10*y)
    reach=.0024+.0031*grain+.0012*lane
    # Tight twisted shoulders darken the actual boundary. The more open
    # wales live one or two yarn diameters back from it.
    shoulder=.44+.56*np.minimum(1.,dist/.0010)
    weights=(.68+.20*grain)*np.exp(-.5*(dist/reach)**2)*shoulder
    attr=mesh.attributes.get(EDGE) or mesh.attributes.new(EDGE,'FLOAT','POINT')
    attr.data.foreach_set('value',weights.astype('f'));mesh.update()


def edge_yarn(sheet,crack):
    bank=crack['rim']
    # Start from a stable material vertex so a paused hand cannot make
    # the broken ends crawl around a closed contour.
    if not crack.get('notch'):
        bank=np.roll(bank,-np.argmin(bank))
        bank=np.r_[bank,bank[0]]
    neutral=sheet.rest[bank]
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(neutral,axis=0),axis=1))]
    s=np.linspace(0.,arc[-1],max(12,int(arc[-1]/.00085))+1)
    j=np.clip(np.searchsorted(arc,s,side='right')-1,0,len(arc)-2)
    t=np.clip((s-arc[j])/np.maximum(arc[j+1]-arc[j],1e-12),0,1)[:,None]
    pos=(1.-t)*sheet.pos[bank[j]]+t*sheet.pos[bank[j+1]]
    n=normalized((1.-t)*sheet.normals[bank[j]]+t*sheet.normals[bank[j+1]])
    lateral=np.roll(pos,-1,axis=0)-np.roll(pos,1,axis=0)
    if crack.get('notch'):lateral[0]=pos[1]-pos[0];lateral[-1]=pos[-1]-pos[-2]
    tangent=normalized(lateral)
    into=normalized(np.cross(tangent,n))
    phase=crack['id']*1.72
    twist=np.sin(s*1760.+phase)
    drift=.00010+.000065*twist
    center=pos+(.00010+.000045*np.sin(s*1120.))[:,None]*n+drift[:,None]*into
    yield center,(.76+.24*twist+.15*np.sin(s*415.+phase))
    # A tucked strand follows only a few neighbouring loop shoulders.
    # Exposed ends grow in small, unequal groups and turn back along the wale.
    rng=np.random.default_rng(int(bank[0])+crack['id']*71)
    station=.0025
    while station<arc[-1]-.002:
        k=min(len(s)-2,int(np.searchsorted(s,station)))
        span=int(rng.integers(3,8));part=slice(k,min(len(s),k+span))
        v=np.linspace(0,1,len(s[part]))
        if len(v)>2:
            lift=np.sin(np.pi*v)
            strand=center[part]+lift[:,None]*(.00025*n[part]-.00031*into[part])
            yield strand,.08+.26*lift
        for _ in range(1+int(rng.random()<.36)):
            k=min(len(s)-2,k+int(rng.integers(0,3)))
            l=rng.uniform(.0013,.0051);hand=rng.uniform(-.52,.66)
            u=np.linspace(0,1,8)[:,None]
            fiber=center[k]+l*((.30+.35*np.sin(np.pi*u))*u*into[k]
                 +(hand*u+.55*u*u)*tangent[k]+.22*np.sin(np.pi*u)*n[k])
            yield fiber,.055+.54*(1.-u[:,0])**.72
        station+=rng.uniform(.007,.015)
    # Keep a handful of fine courses stretched across a taper while their
    # two material anchors remain within the available yarn length.
    first=sheet.sources[crack['members'][0]]
    axis=1 if first['run_axis']=='wale' else 0
    uv=sheet.knit[bank[:-1]].copy()
    if axis==0:uv[:,0]=(uv[:,0]-uv[0,0]+72.)%144.-72.
    for k in range(1,len(uv)-1,3):
        near=np.flatnonzero((abs(uv[:,axis]-uv[k,axis])<.77)&(abs(np.arange(len(uv))-k)>8))
        if not len(near):continue
        root=sheet.pos[bank[k]]
        sep=sheet.pos[bank[near]]-root
        gap=np.linalg.norm(sep,axis=1);j=int(gap.argmin())
        if not .0016<gap[j]<.0095 or near[j]<=k:continue
        i=int(np.searchsorted(s,min(arc[k],s[-1])));j2=int(near[j])
        if normalized(sep[j])@into[i]<.62:continue
        u=np.linspace(0,1,9)[:,None]
        bow=np.sin(np.pi*u)*(.00035*n[i]+.00080*tangent[i])
        yield root+u*(sheet.pos[bank[j2]]-root)+bow,.17+.18*(2.*u[:,0]-1.)**2


def publish_yarns(obj,sheet):
    paths=[]
    for c in sheet.cracks:
        if c['id'] in sheet.original_yarns and not set(c['members'])&sheet.new_yarn:
            paths.extend((sheet.pos[ids]+offset,r) for ids,offset,r in sheet.original_yarns[c['id']])
        else:paths.extend(edge_yarn(sheet,c))
    publish_paths(obj,sheet,paths,'yarn_ends','loose yarn',radius=.00023)


def knit_at_front(sheet):
    if not sheet.run_image or sheet.plan is None:return
    # The slipped wale ends in two light tails. Along fully freed banks the
    # rolled shoulders retain more yarn than the broad run of unbroken cloth.
    atlas=sheet.run_pixels
    h,w=atlas.shape[:2];scale=w/144.
    plan=sheet.plan
    start=getattr(plan,'painted',0)
    path_uv=sheet.knit[plan.path]
    step=np.diff(path_uv,axis=0)
    step[:,0]=(step[:,0]+72.)%144.-72.
    travel=step.sum(0)
    along=int(np.argmax(abs(travel)))
    across=1-along
    # Retract the ladder into the rolled lips along every newly split
    # material segment. The transition has the same width across old and new
    # atlas resolutions, and follows either knit axis across the cylindrical UV.
    radius=np.empty(2);radius[across]=2.2;radius[along]=.9
    for k in range(max(1,start),plan.reached+1):
        a=path_uv[k-1];delta=step[k-1];b=a+delta
        lo=(np.minimum(a,b)-2.7*radius-[0,sheet.first_row])*scale
        hi=(np.maximum(a,b)+2.7*radius-[0,sheet.first_row])*scale
        rows=np.arange(max(0,int(np.floor(lo[1]))),min(h,int(np.ceil(hi[1]))))
        cols=np.arange(int(np.floor(lo[0])),int(np.ceil(hi[0])))[:w]
        if not len(rows) or not len(cols):continue
        x=((cols[None,:]+.5)/scale-a[0])/radius[0]
        y=((rows[:,None]+.5)/scale+sheet.first_row-a[1])/radius[1]
        dx,dy=delta/radius
        t=np.clip((x*dx+y*dy)/max(dx*dx+dy*dy,1e-12),0.,1.)
        gathered=np.exp(-.5*((x-t*dx)**2+(y-t*dy)**2))
        band=np.ix_(rows,cols%w)
        for direction in (0,1):
            atlas[:,:,direction][band]=np.minimum(atlas[:,:,direction][band],1.-.87*gathered)
        atlas[:,:,2][band]=np.maximum(atlas[:,:,2][band],.23*gathered)
        atlas[:,:,3][band]=np.maximum(atlas[:,:,3][band],gathered)
    if not plan.joining and plan.reached>0 and abs(travel[along])>.5:
        from .run_finish import paint_ladder
        paid=abs(step[:plan.reached,along].sum())
        run=sheet.response.run
        paint_ladder(atlas,sheet.first_row,across,path_uv[plan.reached],np.sign(travel[along]),
                     run*(13.+3.2*min(paid,17.)),(1.4+.15*min(paid,13.))*(.65+.55*run),
                     min(1.,(.58+.033*paid)*(.60+.64*run)))
    plan.painted=plan.reached+1
    sheet.run_image.pixels.foreach_set(atlas.ravel());sheet.run_image.update()


def press(tool,face,point,xy):
    sheet=tool.sim
    hit=np.asarray(point)
    tri=sheet.faces[face]
    ab=np.linalg.lstsq((sheet.pos[tri[1:]]-sheet.pos[tri[0]]).T,hit-sheet.pos[tri[0]],rcond=None)[0]
    center=np.r_[1.-ab.sum(),ab]@sheet.rest[tri]
    bridge=sheet.project_bridge(center)
    fine=False
    if bridge:
        from .open_seam import material_path
        a,b=bridge[-2:]
        path=material_path(sheet,a,b)
        fine=path is not None and np.linalg.norm(np.diff(sheet.rest[path],axis=0),axis=1).max()<.0045
    if not fine:
        from .seam_refine import refine
        face,changed=refine(sheet,face,hit,center)
        if changed:
            topology(tool.obj.data,sheet)
            if sheet.dressed:shade_edge(tool.obj,sheet)
    sheet.pick(face,hit)
    tool.drag_xy=None;tool.start=point.copy();tool.grab_offset=0.
    tool.last=time.perf_counter()
    tool.area.header_text_set(f'已抓住布面　现有 {len(tool.sim.cracks)} 处缺口　横向拉开后松手')
    tool.region.tag_redraw()


def tick(tool,target=None,now=None,*,settle=False):
    s=tool.sim
    now=time.perf_counter() if now is None else now
    dt=np.clip(now-tool.last,.001,.10);tool.last=now
    if target is None and s.grip is not None and tool.drag_xy is not None:target=tool.target_at(tool.drag_xy)
    delta=None if target is None else target-np.asarray(tool.start)
    wet=getattr(tool,'wet',None)
    previous=s.pos if wet is not None else None
    if not s.step(dt,delta,settle=settle):return
    if wet is not None:
        wet.prepare_contact()
        s.contact_pose.seat_wet_seam(s,previous,wet,dt)
    if s.plan is not None and s.plan.reached:
        from .seam_overlap import correct
        if s.changed and getattr(s.plan,'checked_rest',-1)!=s.plan.reached:correct(s,released=True)
        if s.changed or delta is not None:correct(s,released=False)
    if s.grip is not None and hasattr(s,'held_clearance'):
        # Saved cuts deform a separate cloth object. Its long, uncut side
        # faces can dip into the shin even while the actual rim stays clear.
        from .seam_contact import seat_held_faces
        seat_held_faces(s,s.support_normals,s.held_clearance,s.layout,face_needs=getattr(s,'held_face_lifts',None))
    mesh=tool.obj.data
    if s.changed:
        topology(mesh,s)
        if s.parent is not None:
            s.dressed.add(s.parent)
            s.new_yarn.add(s.parent)
        knit_at_front(s)
        shade_edge(tool.obj,s)
        from .controls import fracture_audio
        fracture_audio(tool)
        tool.obj['weft_live_tears']=len(s.cracks)
        tool.area.header_text_set(f'已抓住布面　当前 {len(s.cracks)} 处缺口　松开左键释放')
    else:
        mesh.vertices.foreach_set('co',s.pos.astype('f').ravel());mesh.update()
    publish_yarns(tool.obj,s)
    tool.region.tag_redraw()


def modal(tool,ctx,event,xy):
    if event.type=='LEFTMOUSE':
        if event.value=='PRESS':
            if not tool.pointer_in_view(xy):return {'PASS_THROUGH'}
            hit,p,face,obj=tool.pick_visible(ctx,*tool.pick_ray(xy))
            tool.last_pick=(xy,obj.name if hit else None,face)
            if hit and obj==tool.obj:press(tool,face,p,xy)
        elif event.value=='RELEASE' and tool.sim.grip is not None:tool.release_grip()
        else:return {'PASS_THROUGH'}
    elif event.type=='MOUSEMOVE' and tool.sim.grip is not None:
        if tool.pointer_in_view(xy):
            tool.last_input=time.perf_counter()
            tool.enqueue_drag(xy)
    elif event.type=='TIMER':
        if tool.sim.moving:tick(tool)
        if tool.wet_mode:
            tool.wet_step+=1
            if tool.wet is not None:
                tool.wet.advance();tool.wet_display.publish();tool.region.tag_redraw()
    else:return {'PASS_THROUGH'}
    return {'RUNNING_MODAL'}
