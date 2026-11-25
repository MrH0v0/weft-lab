"""Reversible rest-mesh subdivision cache for fixed-action replay."""
import bpy


def cache_skin(body,context):
    if body.get('weft_original_skin_mesh'):
        raise ValueError("皮肤已经使用细分缓存")
    mods=list(body.modifiers)
    if [m.type for m in mods]!=['SUBSURF','ARMATURE']:
        raise ValueError("此缓存适用于先细分、后骨骼蒙皮的皮肤")
    sub,arm=mods
    if body.mode!='OBJECT' or body.library or body.override_library or body.data.library:
        raise ValueError("请在对象模式中选择本地皮肤")
    if body.data.shape_keys or body.data.animation_data or body.animation_data:
        raise ValueError("动态皮肤或修改器需要实时求值")
    if not (sub.show_viewport and sub.show_render and arm.show_viewport and arm.object):
        raise ValueError("请启用皮肤细分和骨骼蒙皮")
    if sub.levels!=sub.render_levels or getattr(sub,'use_adaptive_subdivision',False) or context.scene.render.use_simplify:
        raise ValueError("需要相同的视口和渲染细分级别，并关闭简化")
    original=body.data
    try:
        arm.show_viewport=False
        context.view_layer.update()
        graph=context.evaluated_depsgraph_get()
        cached=bpy.data.meshes.new_from_object(body.evaluated_get(graph),preserve_all_data_layers=True,depsgraph=graph)
    finally:
        arm.show_viewport=True
        context.view_layer.update()
    cached.name=original.name+" • Weft rest subdivision"
    try:
        body['weft_original_skin_mesh']=original
        body['weft_cached_sub_uid']=sub.persistent_uid
        sub.show_viewport=False
        sub.show_render=False
        body.data=cached
        context.view_layer.update()
    except Exception:
        body.data=original
        sub.show_render=sub.show_viewport=True
        for key in ('weft_original_skin_mesh','weft_cached_sub_uid'):
            if key in body:del body[key]
        if not cached.users:bpy.data.meshes.remove(cached)
        raise


def restore_skin(body,context):
    original=body.get('weft_original_skin_mesh')
    if not original:raise ValueError("未找到原始皮肤")
    sub=next((m for m in body.modifiers if m.persistent_uid==body['weft_cached_sub_uid']),None)
    if not sub or sub.type!='SUBSURF':raise ValueError("原细分修改器已删除，请检查皮肤修改器栈")
    body.data=original
    sub.show_viewport=sub.show_render=True
    del body['weft_original_skin_mesh'],body['weft_cached_sub_uid']
    context.view_layer.update()


class WEFT_OT_skin_cache(bpy.types.Operator):
    bl_idname="weft.skin_cache"
    bl_label="切换固定回放皮肤细分缓存"
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        from .controls import live_garment
        scene=context.scene
        if scene.get('weft_check_running') or live_garment(scene) is not None:
            return {'CANCELLED'}
        target=scene.weft_length_object or scene.weft_generated
        body=target.get('weft_contact_skin') if target else None
        if not body:return {'CANCELLED'}
        try:
            if body.get('weft_original_skin_mesh'):restore_skin(body,context)
            else:cache_skin(body,context)
        except ValueError as e:
            self.report({'ERROR'},str(e));return {'CANCELLED'}
        return {'FINISHED'}
