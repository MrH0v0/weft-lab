"""Audit the whole evaluated sock, including ambiguous nearest-face contacts."""
import hashlib
import json
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .posed_contact import world_vertices, correction, add_lookup


def collision_mesh(skin, slot, graph):
    obj=skin.evaluated_get(graph)
    verts=world_vertices(obj)
    mesh=obj.data
    mesh.calc_loop_triangles()
    ts=mesh.loop_triangles
    ids=np.empty((len(ts),3),np.int32)
    ts.foreach_get('vertices',ids.ravel())
    poly=np.empty(len(ts),np.int32)
    ts.foreach_get('polygon_index',poly)
    if slot>=0:
        mask=np.fromiter((t.material_index==slot for t in ts),bool,len(ts))
        ids,poly=ids[mask],poly[mask]
    if not len(ids):
        raise ValueError("所选皮肤材质没有接触面")
    return verts,ids,poly


def collider(skin, slot, graph):
    verts,ids,poly=collision_mesh(skin,slot,graph)
    tree=BVHTree.FromPolygons(verts.tolist(),ids.tolist(),all_triangles=True)
    return tree,verts[ids],poly


def opposing_face(point, tree, hit, normal, distance):
    for other_hit, other_normal, other, _ in tree.find_nearest_range(point, distance+.000002):
        if normal.dot(other_normal)<-.8 and (hit-other_hit).length<.00001:
            return other
    return None


def sample(points, tree, poly, verify_ties, *, details=False):
    gaps=np.empty(len(points))
    ties={}
    for i,row in enumerate(points):
        p=Vector(row)
        q,n,face,d=tree.find_nearest(p)
        gaps[i]=1000*(p-q).dot(n)
        if verify_ties:
            other=opposing_face(p,tree,q,n,d)
            if other is not None:
                ties[i]=[int(poly[face]),int(poly[other])]
    result={'min_mm':float(gaps.min()),'inside':int((gaps<-.1).sum()),
            'worst_vertex':int(gaps.argmin()),'opposing':ties}
    if verify_ties:
        unambiguous=np.ones(len(gaps),bool)
        unambiguous[list(ties)]=False
        result['inside_unambiguous']=int(((gaps<-.1)&unambiguous).sum())
        result['min_unambiguous_mm']=float(gaps[unambiguous].min()) if unambiguous.any() else None
        if details:
            result['penetrating_unambiguous'] = {
                int(i): float(gaps[i]) for i in np.flatnonzero((gaps<-.1)&unambiguous)}
    return result


def scan(scene,sock,policy=None):
    """Yield after each paired sample; restore pose on cancellation or exception."""
    body=sock['weft_contact_skin']
    slot=sock['weft_skin_slot']
    rig=body.find_armature()
    key=sock.data.shape_keys.key_blocks[sock['weft_bound']]
    graph=bpy.context.evaluated_depsgraph_get()
    restore=(scene.frame_current,scene.frame_subframe,key.value,key.mute,rig.data.pose_position)
    report={'vertices':len(sock.data.vertices),'frames':{},'units':'mm','skin':body.name,'slot':slot}
    offsets=[]
    digest=hashlib.sha256()
    raw_text=sock.get('weft_raw_audit')
    if policy and (not raw_text or sock.get('weft_baked_policy')):
        raise ValueError("请先在待修正的袜面副本上完成原始检查")
    if policy:
        raw=json.loads(raw_text.as_string())
    union=set()
    try:
        key.mute=False
        for rest,f in [(True,scene.frame_start)]+[(False,i) for i in range(scene.frame_start,scene.frame_end+1)]:
            rig.data.pose_position='REST' if rest else 'POSE'
            scene.frame_set(f)
            tree,faces,poly=collider(body,slot,graph)
            digest.update(faces.tobytes());digest.update(poly.tobytes())
            pair=[];records={}
            for value,label in [(0,'off')] if rest else [(0,'off'),(1,'on')]:
                key.value=value;bpy.context.view_layer.update()
                ev=sock.evaluated_get(graph)
                p=world_vertices(ev)
                if len(p)!=len(sock.data.vertices):
                    raise ValueError("全袜审查需要保持袜面顶点拓扑")
                digest.update(p.tobytes())
                if policy:
                    if not rest:
                        skip=raw['frames'][str(f)][label]['opposing'] if policy=='SKIP' else {}
                        inv=np.linalg.inv(np.asarray(ev.matrix_world)[:3,:3])
                        pair.append(np.array([correction(Vector(v),tree,faces) if str(i) not in skip else (0,0,0)
                                              for i,v in enumerate(p)],np.float32) @ inv.T)
                else:
                    records[label]=sample(p,tree,poly,True)
                    union.update(records[label]['opposing'])
            if policy and not rest:offsets.append(pair)
            elif rest:report['rest']=records
            else:report['frames'][str(f)]=records
            yield
        if policy:
            if digest.hexdigest()!=raw['geometry_sha256']:
                raise ValueError("动作、网格或检查范围发生变化，请重新检查再确认")
            add_lookup(scene,sock,np.asarray(offsets,np.float32))
            sock['weft_baked_policy']=policy
            if 'weft_target_contact' in scene:del scene['weft_target_contact']
        else:
            report['geometry_sha256']=digest.hexdigest()
            report['conflict_vertices']=sorted(union)
            field='weft_corrected_audit' if sock.get('weft_baked_policy') else 'weft_raw_audit'
            doc=sock.get(field) or bpy.data.texts.new(sock.name+' • '+field)
            doc.clear();doc.write(json.dumps(report,separators=(',',':')))
            sock[field]=doc
            group=sock.vertex_groups.get("Weft • 原始法线冲突")
            if field=='weft_raw_audit':
                sock['weft_conflict_count']=len(union)
                if group:sock.vertex_groups.remove(group)
                if union:
                    sock.vertex_groups.new(name="Weft • 原始法线冲突").add(sorted(union),1,'REPLACE')
            contact={'object':sock.name,'start':scene.frame_start,'vertices':len(sock.data.vertices)}
            for label in ('off','on'):
                for metric,k in [('min_mm','min'),('inside','inside')]:
                    contact[label+'_'+k]=[r[label][metric] for r in report['frames'].values()]
            scene['weft_target_contact']=contact
    finally:
        frame,sub,weight,mute,pose=restore
        rig.data.pose_position=pose
        key.value=weight;key.mute=mute
        scene.frame_set(frame,subframe=sub)
        bpy.context.view_layer.update()


class WEFT_OT_audit(bpy.types.Operator):
    bl_idname="weft.audit_binding"
    bl_label="检查静止及整段全袜"
    mode:bpy.props.EnumProperty(items=[('SCAN','检查',''),('NEAREST','按最近面修正',''),('SKIP','跳过反向重面点',''),
                                       ('MOTION','准备动作贴肤','')])
    def invoke(self,context,event):
        if self.mode=='NEAREST':
            return context.window_manager.invoke_confirm(self,event)
        return self.execute(context)
    def execute(self,context):
        scene=context.scene
        from .controls import live_garment
        if live_garment(scene) is not None:
            return {'CANCELLED'}
        sock=scene.weft_length_object if self.mode=='MOTION' else scene.weft_generated
        if not sock or sock.name not in scene.objects:return {'CANCELLED'}
        def stages():
            if self.mode=='MOTION':
                from .motion_fit import prepare
                start,end=(scene.frame_preview_start,scene.frame_preview_end) if scene.use_preview_range else (scene.frame_start,scene.frame_end)
                yield from prepare(context,sock,start,end)
                return
            if self.mode!='SCAN':
                yield from scan(scene,sock,self.mode)
            yield from scan(scene,sock)
        if scene.get('weft_check_running'):
            self.report({'WARNING'},"正在检查");return {'CANCELLED'}
        self.steps=stages()
        self.scene=scene
        self.i=0
        self.timer=context.window_manager.event_timer_add(.02,window=context.window)
        context.window_manager.modal_handler_add(self)
        scene['weft_check_running']=True
        return {'RUNNING_MODAL'}
    def finish(self,context):
        if self.timer is None:
            return
        context.window_manager.event_timer_remove(self.timer)
        self.timer=None
        self.steps.close()
        self.scene['weft_check_running']=False
        if context.area:context.area.header_text_set(None)
    def modal(self,context,event):
        if event.type=='ESC':
            self.finish(context);return {'CANCELLED'}
        if event.type!='TIMER' or getattr(event,'timer',self.timer)!=self.timer:return {'RUNNING_MODAL'}
        try:
            frame=next(self.steps);self.i+=1
            if context.area:
                label=f'正在准备动作贴肤 · F{frame} · Esc 取消' if self.mode=='MOTION' else f'全袜检查 / 接触处理 {self.i} · Esc 结束'
                context.area.header_text_set(label)
        except StopIteration:
            self.finish(context);return {'FINISHED'}
        except Exception as e:
            self.finish(context);self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def cancel(self,context):
        self.finish(context)

