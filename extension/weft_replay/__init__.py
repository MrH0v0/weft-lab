"""Weft Lab • cached ankle experiment."""
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import FloatProperty
from .cache import DATA,assets,delta,frames,grip_gap,Section,xyz
from .binding import bind,binding_state,apply_bound
from .scene import create,set_camera
from .stitches import centrelines,StitchPatch
from .semi_binding import WEFT_OT_grab,WEFT_OT_prepare,WEFT_OT_restore_binding
from .contact_audit import WEFT_OT_audit
from .skin_cache import WEFT_OT_skin_cache
from .live_view import WEFT_OT_live,WEFT_OT_live_reset
from .sock_length import (WEFT_OT_apply_length, WEFT_OT_restore_length, clear_contact_cache,
                          update_contact, update_frame_contact)
from . import wet_controls, controls, monitor_ui, lifecycle

_runtimes={}
_target_bindings={}

def line(scene,role,count=257):
    curve=bpy.data.curves.new(role,"CURVE");curve.dimensions="3D";curve.bevel_depth=.00015
    spline=curve.splines.new("POLY");spline.points.add(count-1)
    obj=bpy.data.objects.new(role,curve);scene.collection.objects.link(obj);obj["weft_role"]=role
    obj.hide_render=True;obj.color=(.1,.85,.68,1) if role=="Cloth section" else (.82,.48,.18,1)
    return obj

def runtime(scene):
    pointer=scene.as_pointer()
    existing=_runtimes.get(pointer)
    if existing is not None:return existing
    objects={o.get("weft_role"):o for o in scene.objects if o.get("weft_role")}
    if "Stocking" not in objects or "Leg" not in objects:return None
    for name in ["Skin section","Cloth section"]:
        if name not in objects:objects[name]=line(scene,name)
    for name in ("Loop A","Loop B"):
        if name not in objects:
            obj=line(scene,name,49);obj.data.bevel_depth=.000055;obj.show_in_front=True
            objects[name]=obj
    m=frames()[0]["section"];sr,si=assets()["Leg"][:2];cr,ci=assets()["Stocking"][:2]
    data={"objects":objects,"skin":Section(sr,si,m,False).depths(sr),"section":Section(cr,ci,m,True),"bound":_target_bindings,"stitches":StitchPatch()}
    _runtimes[pointer]=data;return data

def update_reference(scene):
    if not scene.get("weft_replay"):return
    state=runtime(scene)
    if not state:return
    i=max(0,min(len(frames())-1,scene.frame_current-1));f=frames()[i]
    d=delta(i);p=assets()["Stocking"][0]+d;obj=state["objects"]["Stocking"]
    # The built-in position attribute exposes the contiguous float3 buffer.
    # MeshVertex.co instead traverses an RNA property for every vertex.
    positions=obj.data.attributes["position"].data
    positions.foreach_set("vector",p.ravel());obj.data.update()
    # Read back native mesh positions; section results never consume web gap samples.
    positions.foreach_get("vector",p.ravel())
    sec=state["section"];skin=state["skin"];cloth=sec.depths(p);gap=cloth-skin;state["gap"]=gap
    scene["weft_valley"]=float(gap[75]);scene["weft_min"]=float(np.nanmin(gap))
    scene["weft_grip"]=grip_gap(f["probe"])
    for name,uv in zip(("Loop A","Loop B"),centrelines(f)):
        pts=np.ones((len(uv),4),np.float32);pts[:,:3]=state["stitches"].lift(p,uv)
        state["objects"][name].data.splines[0].points.foreach_set("co",pts.ravel())
    axis=((np.arange(257)+.5)/257-.5)*.048
    for name,heights in (("Skin section",skin),("Cloth section",cloth)):
        line_obj=state["objects"][name];pts=np.ones((257,4),np.float32)
        pts[:,:3]=sec.o+axis[:,None]*sec.t+np.nan_to_num(heights[:,None])*sec.n*.001
        line_obj.data.splines[0].points.foreach_set("co",pts.ravel())
    aspect=scene.render.resolution_x/scene.render.resolution_y
    for name in ("main","detail"):set_camera(state["objects"][name],f[name],aspect)

@persistent
def update(scene,*_):
    update_reference(scene)
    targets=[o for o in scene.objects if o.get('weft_bound')]
    if not targets:return
    d=delta(max(0,min(len(frames())-1,scene.frame_current-1)))
    for target in targets:
        key=target.as_pointer()
        if key not in _target_bindings:_target_bindings[key]=binding_state(target)
        apply_bound(target,_target_bindings[key],d)

@persistent
def clear(*_):
    _runtimes.clear()
    _target_bindings.clear()

@persistent
def restore(*_):
    for image in bpy.data.images:
        asset=image.get("weft_asset")
        if asset not in ("coverage.png","knit/knit0001.png") or image.packed_file:
            continue
        path=Path(bpy.path.abspath(image.filepath,library=image.library))
        if path.is_file():
            if asset=="coverage.png" or all((path.parent/frame.name).is_file()
                for frame in (DATA/"knit").glob("knit[0-9][0-9][0-9][0-9].png")):
                continue
        image.filepath=str(DATA/asset)
    for scene in bpy.data.scenes:
        if scene.get("weft_replay"):
            if "weft_check_running" in scene:del scene["weft_check_running"]
            update(scene)

class WEFT_OT_load(bpy.types.Operator):
    bl_idname="weft.load_replay";bl_label="新建脚踝回放场景"
    def execute(self,context):
        create(context);return {"FINISHED"}

class WEFT_OT_camera(bpy.types.Operator):
    bl_idname="weft.switch_camera";bl_label="切换全景 / 线圈近景"
    def execute(self,context):
        s=context.scene;objs=runtime(s)["objects"]
        main=s.objects.get(s.get("weft_model_main","")) or objs["main"]
        detail=s.objects.get(s.get("weft_model_detail","")) or objs["detail"]
        s.camera=detail if s.camera==main else main
        return {"FINISHED"}

class WEFT_OT_bind(bpy.types.Operator):
    bl_idname="weft.bind_garment";bl_label="从游标绑定袜面副本";bl_options={"REGISTER","UNDO"}
    def execute(self,context):
        s=context.scene;target=context.active_object
        if not target:self.report({"ERROR"},"请选择袜面");return {"CANCELLED"}
        try:obj=bind(s,target,np.asarray(s.cursor.location),s.cursor.matrix.to_3x3(),s.weft_scale)
        except ValueError as e:self.report({"ERROR"},str(e));return {"CANCELLED"}
        # The source remains available, but drawing both surfaces would overlap.
        target.hide_set(True);target.hide_render=True
        context.view_layer.objects.active=obj;obj.select_set(True);update(s);return {"FINISHED"}

class WEFT_PT_replay(bpy.types.Panel):
    bl_label="参考样本 · 固定缓存回放";bl_idname="WEFT_PT_replay";bl_space_type="VIEW_3D";bl_region_type="UI";bl_category="Weft Lab"
    bl_options={'DEFAULT_CLOSED'}
    bl_order=10
    def draw(self,context):
        ui=self.layout;s=context.scene
        if controls.live_garment(s) is not None:
            ui.label(text="恢复原袜后再使用参考回放")
            return
        ui.operator("weft.load_replay")
        if not s.get("weft_replay"):return
        f=frames()[max(0,min(s.frame_current-1,len(frames())-1))]
        contact=s.get("weft_target_contact")
        if contact:
            garment=s.objects.get(contact["object"])
            if garment and garment.data.shape_keys:
                key=garment.data.shape_keys.key_blocks.get(garment.get("weft_bound",""))
                if key:
                    box=ui.box()
                    box.label(text="目标袜面 · 固定动作实测")
                    box.prop(key,"value",text="回放形态键")
                    value=0 if key.mute else key.value
                    index=s.frame_current-contact["start"]
                    if value in (0,1) and s.frame_subframe==0 and 0<=index<len(contact["on_min"]):
                        mode="on" if value else "off"
                        box.label(text=f"最小离肤   {contact[mode+'_min'][index]:+.3f} mm")
                        box.label(text=f"穿入超过 0.1 mm   {contact[mode+'_inside'][index]} / {contact['vertices']}")
                    else:
                        box.label(text="此帧或形态键权重尚未采样")
                    box.label(text="更改动作或网格后需重新测量")
        ui.operator("weft.switch_camera")
        ui.label(text=f"物理步 {f['tick']}   /   {f['tick']/120:.3f} s")
        ui.label(text=f"抓点离肤   {s.get('weft_grip',0):.3f} mm")
        ui.label(text=f"参考谷位   {s.get('weft_valley',0):.3f} mm")
        ui.label(text=f"截线最小   {s.get('weft_min',0):.3f} mm")
        ui.label(text=f"圈冠横偏   {f['leanA']:+.2f} / {f['leanB']:+.2f} %")
        ui.label(text=f"停手路程 / 净位移   {f['path']:.2f} / {f['net']:.2f} mm")
        ui.separator();ui.label(text="迁移到当前场景的其他袜面");ui.label(text="游标对齐目标抓点与朝向")
        ui.prop(s,"weft_scale");ui.operator("weft.bind_garment")
        ui.label(text="上方探针与截线属于参考样本")

class WEFT_PT_binding(bpy.types.Panel):
    bl_label="半自动袜面绑定";bl_idname="WEFT_PT_binding"
    bl_space_type="VIEW_3D";bl_region_type="UI";bl_category="Weft Lab"

    bl_options={'DEFAULT_CLOSED'}
    bl_order=-10

    def draw(self,context):
        ui=self.layout;s=context.scene
        if controls.live_garment(s) is not None:
            ui.label(text="结束并恢复原袜后才能换目标")
            return
        if s.weft_length_object is not None:
            ui.operator("weft.restore_length")
            return
        ui.enabled=not s.get('weft_check_running',False)
        ui.prop(s,'weft_sock');ui.prop(s,'weft_skin')
        if s.weft_skin and len(s.weft_skin.data.materials)>1:
            ui.prop_search(s,'weft_skin_material',s.weft_skin.data,'materials',text='皮肤材质')
        ui.operator("weft.pick_grab")
        if s.weft_sock and s.weft_grab_owner==s.weft_sock:ui.label(text="已记录袜面抓点")
        ui.prop(s,'weft_side')
        ui.prop(s,'weft_override')
        rig = s.weft_sock.find_armature() if s.weft_sock else None
        if rig is None and s.weft_skin:
            rig = s.weft_skin.find_armature()
        if s.weft_override and rig is not None:
            bones=rig.data
            for side in 'LR':
                for role,label in [('shin','膝'),('foot','踝')]:
                    ui.prop_search(s,'weft_'+side+role,bones,'bones',text=side+' '+label)
        ui.prop(s,'weft_knit_scale')
        ui.operator("weft.prepare_binding")
        if s.get('weft_binding_error'):
            ui.label(text=s['weft_binding_error'],icon='ERROR')
        obj=s.weft_generated
        if not obj or not obj.get('weft_bound'):return
        if obj.get('weft_binding_source'):
            ui.operator("weft.restore_binding")
        body=obj.get('weft_contact_skin')
        if body and (body.get('weft_original_skin_mesh') or [m.type for m in body.modifiers]==['SUBSURF','ARMATURE']):
            cached=bool(body.get('weft_original_skin_mesh'))
            ui.operator("weft.skin_cache",text="恢复可编辑的原皮肤" if cached else "为固定回放缓存皮肤细分")
            if cached:ui.label(text="编辑皮肤或修改细分前请先恢复")
        ui.label(text=f"{obj.name} · {obj.get('weft_leg','')} 腿 · {obj.get('weft_bone_scale',1):.3f}×")
        ui.operator('weft.audit_binding').mode='SCAN'
        raw=obj.get('weft_raw_audit')
        if raw:
            ui.label(text=f"全段反向重面顶点 {obj.get('weft_conflict_count',0)} 个")
            ui.label(text="逐帧面号和原始读数在文本编辑器")
            ui.label(text=raw.name)
            if not obj.get('weft_baked_policy'):
                ui.prop(s,'weft_contact_policy')
                ui.operator('weft.audit_binding',text="烘焙接触修正").mode=s.weft_contact_policy

classes=(WEFT_OT_load,WEFT_OT_camera,WEFT_OT_bind,WEFT_OT_grab,WEFT_OT_prepare,WEFT_OT_restore_binding,WEFT_OT_audit,WEFT_OT_skin_cache,WEFT_OT_live,WEFT_OT_live_reset,WEFT_OT_apply_length,WEFT_OT_restore_length,WEFT_PT_replay,WEFT_PT_binding)

_semi_props={}
def mesh_only(self,obj):return obj.type=='MESH'
def register():
    for cls in classes:bpy.utils.register_class(cls)
    wet_controls.register()
    controls.register()
    monitor_ui.register()
    bpy.types.Scene.weft_scale=FloatProperty(name="目标 / 参考局部尺寸",default=1,min=.1,max=10)
    for k,label in [('sock','袜面'),('skin','皮肤'),('grab_owner','抓点袜面'),('generated','回放副本')]:
        options={'update':controls.target_changed} if k=='sock' else {}
        _semi_props['weft_'+k]=bpy.props.PointerProperty(type=bpy.types.Object,name=label,poll=mesh_only,**options)
    _semi_props.update(weft_grab_local=bpy.props.FloatVectorProperty(size=3),
                       weft_length_object=bpy.props.PointerProperty(type=bpy.types.Object,poll=mesh_only),
                       weft_cuff_height=FloatProperty(name='袜口高度',description='从小腿到大腿根，应用后生成对应的单腿袜面',default=0.,min=0.,max=1.,subtype='FACTOR'),
                       weft_motion_speed=FloatProperty(name="动作速度",description="从当前帧播放到时间轴末尾，设为 0 时固定姿势",default=1.,min=0.,max=2.),
                       weft_knit_scale=FloatProperty(name="织纹尺寸倍率",description="1.0 使用参考袜的静止织纹尺寸，独立于骨长",default=1.0,min=.1,max=10),
                       weft_contact_policy=bpy.props.EnumProperty(name="冲突点",items=[('SKIP','标记并保留原位','该帧保留原蒙皮与回放位置'),('NEAREST','使用最近面修正','按最近面的法线修正冲突点')],default='SKIP'),
                       weft_skin_material=bpy.props.StringProperty(name='皮肤材质'),
                       weft_override=bpy.props.BoolProperty(name='指定骨骼名称'),
                       weft_side=bpy.props.EnumProperty(name='腿侧',items=[('AUTO','自动识别',''),('L','左腿',''),('R','右腿','')]))
    for side in 'LR':
        for role in ('shin','foot'):_semi_props['weft_'+side+role]=bpy.props.StringProperty()
    for k,v in _semi_props.items():setattr(bpy.types.Scene,k,v)
    bpy.app.handlers.frame_change_pre.append(update)
    bpy.app.handlers.depsgraph_update_post.append(update_contact)
    bpy.app.handlers.frame_change_post.append(update_frame_contact)
    bpy.app.handlers.load_post.append(restore)
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        handlers.append(clear)
        handlers.append(clear_contact_cache)
    lifecycle.register()

def unregister():
    lifecycle.unregister()
    bpy.app.handlers.depsgraph_update_post.remove(update_contact)
    bpy.app.handlers.frame_change_post.remove(update_frame_contact)
    monitor_ui.unregister()
    controls.unregister()
    wet_controls.unregister()
    bpy.app.handlers.load_post.remove(restore)
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        handlers.remove(clear)
        handlers.remove(clear_contact_cache)
    bpy.app.handlers.frame_change_pre.remove(update);clear()
    clear_contact_cache()
    del bpy.types.Scene.weft_scale
    for k in _semi_props:delattr(bpy.types.Scene,k)
    _semi_props.clear()
    for cls in reversed(classes):bpy.utils.unregister_class(cls)

