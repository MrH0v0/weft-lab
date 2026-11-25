"""Bind a clicked rest-surface point using the target character's own leg chain."""
import re

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .binding import bind
from .controls import live_garment
from .posed_contact import world_vertices

REFERENCE_SHIN_M = .40


def viewport_ray(region, rv, xy):
    origin = view3d_utils.region_2d_to_origin_3d(region, rv, xy, clamp=20)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv, xy)
    if rv.view_perspective == 'CAMERA' and not rv.is_perspective:
        eye = rv.view_matrix.inverted().translation
        origin += direction * (eye - origin).dot(direction)
    return origin, direction


def ray_anchor(sock, origin, direction, graph, *, with_hit=False):
    evaluated = sock.evaluated_get(graph)
    mesh = evaluated.data
    mesh.calc_loop_triangles()
    base = sock.data
    # Tessellation can change with pose, but polygon corner IDs must survive.
    if len(mesh.vertices) != len(base.vertices) or len(mesh.polygons) != len(base.polygons):
        raise ValueError("袜面求值后拓扑改变，请选择已应用细分的袜面")
    faces = [t.vertices[:] for t in mesh.loop_triangles]
    tree = BVHTree.FromPolygons(world_vertices(evaluated).tolist(), faces, all_triangles=True)
    hit, _, index, _ = tree.ray_cast(Vector(origin), Vector(direction).normalized())
    if index is None:
        raise ValueError("没有点到所选袜面")
    tri = mesh.loop_triangles[index]
    if tuple(mesh.polygons[tri.polygon_index].vertices) != tuple(base.polygons[tri.polygon_index].vertices):
        raise ValueError("求值后的三角面无法映回原袜面")
    ids = faces[index]
    a,b,c = [np.asarray(evaluated.matrix_world @ mesh.vertices[i].co) for i in ids]
    uv, _, rank, _ = np.linalg.lstsq(np.stack((b-a,c-a),axis=1), np.asarray(hit)-a,rcond=None)
    if rank < 2:
        raise ValueError("抓点三角面退化，请点选相邻位置")
    weights = [1-uv.sum(), *uv]
    local = sum((float(w)*base.vertices[i].co for i,w in zip(ids,weights)), Vector())
    return (local, hit) if with_hit else local


def leg_chain(rig, side):
    """Resolve a unique shin and ankle. Names may carry a Mixamo namespace."""
    native='左' if side=='L' else '右'
    mmd=[rig.data.bones.get(native+suffix) for suffix in ('ひざD','足首D')]
    if all(mmd):return mmd
    names = (('J_Bip_'+side+'_LowerLeg','leg_joint_'+side+'_2',side+'Shin','shin.'+side,
              ('Left' if side=='L' else 'Right')+'Leg'),
             ('J_Bip_'+side+'_Foot','leg_joint_'+side+'_3',side+'Foot','foot.'+side,
              ('Left' if side=='L' else 'Right')+'Foot'))
    points = []
    for aliases in names:
        norm = lambda s: re.sub(r'[^a-z0-9]','',s.split(':')[-1].lower())
        keys = {norm(name) for name in aliases}
        found = [b for b in rig.data.bones if norm(b.name) in keys]
        if len(found) != 1:
            raise ValueError(side+" 腿骨链不唯一或未识别，请指定膝、踝骨")
        points.append(found[0])
    return points


def infer(scene, sock, skin, local):
    if scene.unit_settings.system!='NONE' and abs(scene.unit_settings.scale_length-1)>1e-6:
        raise ValueError("此版本按每 Blender 单位 1 米适配，请先统一场景比例")
    rig = sock.find_armature()
    skin_rig = skin.find_armature()
    # The accepted fitted Lisa mesh is transported by its designated rigged skin.
    if rig is None and sock.get('weft_contact_skin') == skin:
        rig = skin_rig
    if sock == skin or not rig or skin_rig != rig:
        raise ValueError("请选择同一骨架的独立袜面和皮肤")
    anchor = sock.matrix_world @ Vector(local)
    legs = {}
    for side in 'LR':
        configured = [getattr(scene,'weft_'+side+role) for role in ('shin','foot')]
        if scene.weft_override and any(configured):
            bones = [rig.data.bones.get(n) for n in configured]
            if not all(bones):
                raise ValueError(side+" 腿需要同时指定膝、踝骨")
        else:
            bones = leg_chain(rig,side)
        if bones[0] not in bones[1].parent_recursive:
            raise ValueError(side+" 踝骨须位于膝骨的下游")
        legs[side] = [rig.matrix_world @ b.head_local for b in bones]
    distances = {s:(anchor-p[1]).length for s,p in legs.items()}
    side = scene.weft_side
    if side == 'AUTO':
        side=min(distances,key=distances.get)
        if abs(distances['L']-distances['R']) < .08*(legs[side][0]-legs[side][1]).length:
            raise ValueError("抓点与两踝接近，请确认左右腿")
    other = 'R' if side=='L' else 'L'
    z=(legs['L'][0]+legs['R'][0]-legs['L'][1]-legs['R'][1]).normalized()
    x=legs[side][1]-legs[other][1]
    x=(x-z*x.dot(z)).normalized()
    if min(x.length,z.length)<.9:
        raise ValueError("骨链方向退化，请检查骨骼选择")
    ratio=(legs[side][0]-legs[side][1]).length/REFERENCE_SHIN_M
    if ratio<1e-5:
        raise ValueError("选中腿的膝、踝关节重合")
    rotation=np.stack((x,z.cross(x),z),axis=1)
    return side,ratio,anchor,rotation


def materials(scene, skin):
    name=scene.weft_skin_material.strip()
    if not name and len(skin.data.materials)<=1:
        return None
    if name:
        ids={i for i,m in enumerate(skin.data.materials) if m and m.name==name}
    else:
        ids={i for i,m in enumerate(skin.data.materials) if m and 'skin' in m.name.lower()}
    if len(ids)!=1:
        raise ValueError("混合材质皮肤请指定参与接触的皮肤材质")
    return ids


class WEFT_OT_grab(bpy.types.Operator):
    bl_idname="weft.pick_grab"
    bl_label="在袜面点选抓点"
    @classmethod
    def poll(cls, context):
        return live_garment(context.scene) is None

    def invoke(self,context,event):
        if not context.scene.weft_sock or context.area.type!='VIEW_3D':
            self.report({'ERROR'},"请先选择袜面");return {'CANCELLED'}
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set("左键点选袜面，Esc 取消")
        return {'RUNNING_MODAL'}
    def modal(self,context,event):
        if event.type in {'ESC','RIGHTMOUSE'}:
            context.area.header_text_set(None);return {'CANCELLED'}
        if event.type!='LEFTMOUSE' or event.value!='PRESS':
            return {'PASS_THROUGH'}
        reg=next(r for r in context.area.regions if r.type=='WINDOW')
        xy=(event.mouse_x-reg.x,event.mouse_y-reg.y)
        if not (0<=xy[0]<reg.width and 0<=xy[1]<reg.height):
            return {'RUNNING_MODAL'}
        rv=context.space_data.region_3d
        sock=context.scene.weft_sock
        try:
            origin,direction=viewport_ray(reg,rv,xy)
            local,hit=ray_anchor(sock,origin,direction,context.evaluated_depsgraph_get(),with_hit=True)
        except ValueError as e:
            self.report({'WARNING'},str(e));return {'RUNNING_MODAL'}
        context.scene.weft_grab_owner=sock
        context.scene.weft_grab_local=local
        context.scene['weft_grab_world']=list(hit)
        context.area.header_text_set(None);return {'FINISHED'}


class WEFT_OT_prepare(bpy.types.Operator):
    bl_idname="weft.prepare_binding"
    bl_label="生成回放副本"
    bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls, context):
        return context.scene.weft_length_object is None and live_garment(context.scene) is None

    def execute(self,context):
        s=context.scene
        sock,skin=s.weft_sock,s.weft_skin
        if not sock or not skin or s.weft_grab_owner!=sock:
            self.report({'ERROR'},"请选择袜面、皮肤并点选对应抓点");return {'CANCELLED'}
        try:
            ids=materials(s,skin)
            side,ratio,anchor,rotation=infer(s,sock,skin,s.weft_grab_local)
            obj=bind(s,sock,np.asarray(anchor),rotation,ratio,knit_scale=s.weft_knit_scale)
        except ValueError as e:
            s['weft_binding_error']=str(e)
            marker=s.objects.get('Weft • binding ambiguity')
            if marker is None:
                marker=bpy.data.objects.new('Weft • binding ambiguity',None)
                s.collection.objects.link(marker)
            marker.empty_display_type='SPHERE';marker.empty_display_size=.006
            marker.location=s.get('weft_grab_world',sock.matrix_world @ Vector(s.weft_grab_local))
            marker.show_in_front=True;marker['reason']=str(e)
            self.report({'ERROR'},str(e));return {'CANCELLED'}
        obj['weft_contact_skin']=skin
        obj['weft_skin_slot']=next(iter(ids)) if ids else -1
        obj['weft_leg']=side
        obj['weft_bone_scale']=ratio
        obj['weft_grab_local']=list(s.weft_grab_local)
        obj['weft_binding_source_render_hidden']=sock.hide_render
        sock.hide_set(True);sock.hide_render=True
        obj.hide_set(False);obj.hide_render=False
        s.weft_generated=obj
        s['weft_binding_error']=''
        marker=s.objects.get('Weft • binding ambiguity')
        if marker:bpy.data.objects.remove(marker,do_unlink=True)
        if 'weft_target_contact' in s:del s['weft_target_contact']
        context.view_layer.objects.active=obj;obj.select_set(True)
        s.frame_set(s.frame_current)
        return {'FINISHED'}


class WEFT_OT_restore_binding(bpy.types.Operator):
    bl_idname = 'weft.restore_binding'
    bl_label = '切回绑定前原袜'
    bl_options = {'REGISTER','UNDO'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        obj = scene.weft_generated
        return bool(obj and obj.get('weft_binding_source') and scene.weft_length_object is None
                    and live_garment(scene) is None)

    def execute(self, context):
        scene = context.scene
        bound = scene.weft_generated
        source = bound['weft_binding_source']
        bound.hide_set(True)
        bound.hide_render = True
        source.hide_set(False)
        source.hide_render = bound.get('weft_binding_source_render_hidden',False)
        scene.weft_sock = source
        scene.weft_generated = None
        context.view_layer.objects.active = source
        source.select_set(True)
        return {'FINISHED'}
