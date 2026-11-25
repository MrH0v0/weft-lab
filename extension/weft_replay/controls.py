"""Public object settings, interaction controls and event-driven tear audio."""
from dataclasses import replace
from pathlib import Path
import tomllib

import bpy
import numpy as np

with Path(__file__).with_name('blender_manifest.toml').open('rb') as manifest:
    VERSION = tomllib.load(manifest)['version']
_device = None
_sounds = ()
_handles = []


def live_garment(scene):
    from .live_view import ACTIVE
    tool = ACTIVE.get(scene.as_pointer())
    if tool is not None:
        return tool.obj
    return next((obj for obj in scene.objects if obj.get('weft_live_source')), None)


def source_for(scene, active=None):
    live = live_garment(scene)
    if live is not None:
        return live['weft_live_source']
    stocking=scene.weft_length_object
    if stocking is not None and stocking.name in scene.objects:
        return stocking
    generated, sock = scene.weft_generated, scene.weft_sock
    if generated and generated.name in scene.objects:
        original = generated.get('weft_binding_source')
        if sock is None or generated == sock or original == sock:
            return generated
    return sock or (active if active and active.type == 'MESH' else None)


def target_changed(scene, context):
    stocking=scene.weft_length_object
    if stocking is not None and live_garment(scene) is None:
        template=stocking.get('weft_length_source')
        if scene.weft_sock not in (stocking,template,template.get('weft_binding_source') if template else None):
            from .sock_length import restore_length
            restore_length(scene)
    generated = scene.weft_generated
    if generated and generated != scene.weft_sock:
        original = generated.get('weft_binding_source')
        if original != scene.weft_sock:
            scene.weft_generated = None
    scene.weft_grab_owner = None
    scene.weft_skin = None
    scene.weft_skin_material = ''
    scene['weft_binding_error'] = ''
    if 'weft_grab_world' in scene:
        del scene['weft_grab_world']


def appearance(material, source):
    if not material or not material.use_nodes:
        return
    nodes, links = material.node_tree.nodes, material.node_tree.links
    control = nodes.get('Weft Dry Appearance')
    if control:
        control.inputs['Palette'].default_value = source.get('weft_dry_palette', 0)
        control.inputs['Thickness / Opacity'].default_value = source.get('weft_dry_thickness', .64)
    scale = source.get('weft_visual_knit_scale', 1.)
    repeat=nodes.get('Weft drawn yarn repeat')
    if repeat:repeat.outputs[0].default_value=1./scale
    for tex in list(nodes):
        if tex.type != 'TEX_IMAGE' or not tex.image:
            continue
        if not (tex.image.name.startswith('live-loop-profile.png') or
                tex.label == 'Open pore coverage and approximate top relief'):
            continue
        name = 'Weft knit size • '+tex.name
        mapping = nodes.get(name)
        if mapping is None and scale != 1. and tex.inputs['Vector'].is_linked:
            entry = tex.inputs['Vector'].links[0].from_socket
            mapping = nodes.new('ShaderNodeVectorMath')
            mapping.operation = 'SCALE'
            mapping.name = name
            links.new(entry, mapping.inputs[0])
            links.new(mapping.outputs[0], tex.inputs['Vector'])
        if mapping:
            mapping.inputs['Scale'].default_value = 1./scale


def changed(owner, key, value):
    source = owner.id_data
    source[key] = value
    from .live_view import ACTIVE
    tool = next((t for t in ACTIVE.values() if t.obj.get('weft_live_source') == source), None)
    if key in {'weft_dry_palette', 'weft_dry_thickness', 'weft_visual_knit_scale'}:
        for slot in source.material_slots:
            material = slot.material
            if (material is None or not material.use_nodes
                    or not material.node_tree.nodes.get('Weft Dry Appearance')):
                continue
            if slot.link != 'OBJECT' or material.users > 1:
                slot.link = 'OBJECT'
                slot.material = material.copy()
            appearance(slot.material, source)
        if tool:
            appearance(tool.obj.data.materials[0], source)
            if key in {'weft_dry_palette','weft_dry_thickness'}:
                from .live_rims import update_yarn_appearance
                update_yarn_appearance(tool.sim,source)
    if tool:
        fields = {'weft_recovery':'recovery', 'weft_break_extension':'break_extension',
                  'weft_run':'run', 'weft_wrinkles':'wrinkles',
                  'weft_pretension':'pretension', 'weft_fit':'fit'}
        if key in fields:
            tool.sim.response = replace(tool.sim.response, **{fields[key]:value})
        if key == 'weft_tear_volume' and value > 0:
            prepare_audio(tool.obj)
        tool.area.tag_redraw()


def value(key, default):
    return lambda self: self.id_data.get(key, default)


def setter(key):
    return lambda self, val: changed(self, key, val)


class WEFT_PG_fabric(bpy.types.PropertyGroup):
    palette: bpy.props.EnumProperty(name='配色', items=[('BLACK','黑色',''),('SMOKE','烟灰',''),('SKIN','肤色',''),('WHITE','白色','')],
                                   get=value('weft_dry_palette',0), set=setter('weft_dry_palette'))
    thickness: bpy.props.FloatProperty(name='光学厚度', min=0., max=1., get=value('weft_dry_thickness',.64), set=setter('weft_dry_thickness'))
    knit_scale: bpy.props.FloatProperty(name='现场织纹尺寸', description='仅改变显示织纹，独立于骨长和材料力学坐标；固定回放的尺度在绑定前设置',
                                      min=.25,max=4., get=value('weft_visual_knit_scale',1.), set=setter('weft_visual_knit_scale'))
    wrinkles: bpy.props.BoolProperty(name='受拉褶皱', get=value('weft_wrinkles',True), set=setter('weft_wrinkles'))
    recovery: bpy.props.FloatProperty(name='回复速度', min=.3,max=2.,get=value('weft_recovery',1.),set=setter('weft_recovery'))
    break_extension: bpy.props.FloatProperty(name='断裂伸长参数',min=.25,max=2.5,get=value('weft_break_extension',.8),set=setter('weft_break_extension'))
    run: bpy.props.FloatProperty(name='脱圈传播',min=0.,max=1.,get=value('weft_run',.65),set=setter('weft_run'))
    pretension: bpy.props.FloatProperty(name='预张力', subtype='FACTOR', min=0., max=.24,
                                       get=value('weft_pretension',.04), set=setter('weft_pretension'))
    fit: bpy.props.FloatProperty(name='贴合度', subtype='FACTOR', min=0., max=1.,
                                 get=value('weft_fit',.75), set=setter('weft_fit'))
    idle: bpy.props.FloatProperty(name='待机强度',description='固定姿势、松手稳定后的小幅呼吸起伏；1 倍上限 0.12 mm，默认关闭。显示近似，未标定',
                                 min=0.,max=1.,get=value('weft_idle',0.),set=setter('weft_idle'))
    volume: bpy.props.FloatProperty(name='撕裂音量',min=0.,max=1.,get=value('weft_tear_volume',.35),set=setter('weft_tear_volume'))


def prepare_audio(obj):
    """Open the sound at session startup, away from the first tear's mesh update."""
    global _device, _sounds
    if obj['weft_live_source'].get('weft_tear_volume', .35) <= 0:
        return
    try:
        import aud
        if _device is None:
            _device = aud.Device()
        if not _sounds:
            data = Path(__file__).with_name('data')
            _sounds = tuple(aud.Sound(str(data/name)).cache() for name in
                            ('tear.wav','tear-alt.wav','tear-resume.wav'))
        if 'weft_audio_error' in obj:
            del obj['weft_audio_error']
    except (RuntimeError, OSError) as error:
        obj['weft_audio_error'] = str(error)


def fracture_audio(tool):
    """Use one continuous take at the start of each new or resumed tear."""
    events = tool.sim.fracture_events
    count = len(events)
    previous = getattr(tool, 'audio_cursor', 0)
    if count <= previous:
        return
    tool.audio_cursor = count
    tool.obj['weft_fracture_event_count'] = count
    gain = tool.obj['weft_live_source'].get('weft_tear_volume', .35)
    if gain <= 0 or _device is None or not _sounds:
        return
    _handles[:] = [handle for handle in _handles if handle.status]
    try:
        for event in events[previous:]:
            if event['kind']=='first_break':
                cue,level = (event['hole_id']-1)%2,1.
            elif event.get('resumed'):
                cue,level = 2,.65
            else:
                continue
            handle = _device.play(_sounds[cue])
            handle.volume = gain*level
            _handles.append(handle)
            tool.obj['weft_tear_sound_triggers'] = tool.obj.get('weft_tear_sound_triggers', 0)+1
    except (RuntimeError, OSError) as error:
        tool.obj['weft_audio_error'] = str(error)


def idle(tool, now):
    sim, source = tool.sim, tool.obj['weft_live_source']
    strength = source.get('weft_idle', 0.)
    previous = getattr(tool, 'idle_offset', None)
    if sim.pose is not None or sim.grip is not None or sim.moving:
        tool.idle_offset = None
        return False
    if not strength and previous is None:
        return False
    from .knit_patch import surface_normals
    normals = surface_normals(sim.rest, sim.faces)
    height = sim.rest[:,2]
    envelope = np.sin(np.pi*np.clip((height-height.min())/max(np.ptp(height),1e-6),0,1))**2
    offset = strength*.00012*(.5-.5*np.cos((now-tool.started)*2*np.pi/3.6))*envelope[:,None]*normals
    sim.pos[:] = sim.rest+sim.opening+sim.seating+offset
    if sim.patch is not None:sim.constrain_skin()
    sim.tie_fronts(sim.pos)
    sim.step_number += 1
    sim.elapsed += max(0.,now-tool.last)
    tool.idle_offset = offset if strength else None
    tool.publish_positions()
    tool.capture_monitor()
    return True


def stop_audio():
    global _device, _sounds
    for handle in _handles:
        handle.stop()
    _handles.clear()
    _device = None
    _sounds = ()


class WEFT_OT_control(bpy.types.Operator):
    bl_idname = 'weft.live_control'
    bl_label = '实时交互控制'
    action: bpy.props.EnumProperty(items=[(s,s,'') for s in ('PAUSE','STEP','FINISH')])
    def execute(self, context):
        from .live_view import ACTIVE
        from . import monitor_ui
        tool = ACTIVE.get(context.scene.as_pointer())
        if tool is None:
            return {'CANCELLED'}
        if self.action == 'PAUSE': monitor_ui.pause(tool)
        elif self.action == 'STEP': monitor_ui.single_step(tool, context)
        else: tool.finish(context)
        return {'FINISHED'}


class WEFT_PT_session(bpy.types.Panel):
    bl_idname='WEFT_PT_session'; bl_label='Weft Lab · '+VERSION
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category='Weft Lab'; bl_order=-20
    def draw(self,context):
        from .live_view import ACTIVE
        s,ui=context.scene,self.layout
        source=source_for(s,context.active_object)
        tool=ACTIVE.get(s.as_pointer())
        frozen=next((o for o in s.objects if o.get('weft_live_source')),None)
        ui.label(text='当前袜面：'+(source.name if source else '请先选择袜面'))
        if tool:
            mode=('湿态' if tool.wet_mode else '干态')+' · '+('随骨架动作' if tool.speed else '固定姿势')
            pose=tool.committed_pose()[0]
        else:
            mode='封存快照' if frozen else '准备 / 固定回放'
            pose=frozen.get('weft_live_frame',s.frame_current) if frozen else s.frame_current
        ui.label(text=f'F{pose} · {mode}')
        if source and source.get('weft_contact_skin'):
            ui.label(text='皮肤：'+source['weft_contact_skin'].name)
        if source and tool is None and frozen is None:
            from .sock_length import CATEGORIES
            cuff=ui.box()
            cuff.label(text='单腿长筒袜')
            cuff.prop(s,'weft_cuff_height',slider=True)
            cuff.operator('weft.apply_length')
            choices=cuff.column(align=True)
            for start in (0,2):
                row=choices.row(align=True)
                for height,label in CATEGORIES[start:start+2]:
                    row.operator('weft.apply_length',text=label).height=height
            if s.weft_length_object is not None:
                cuff.operator('weft.restore_length')
                from .motion_fit import CACHE
                cuff.enabled=not s.get('weft_check_running',False)
                body=source.get('weft_contact_skin')
                if body and (body.get('weft_original_skin_mesh') or [m.type for m in body.modifiers]==['SUBSURF','ARMATURE']):
                    label='恢复皮肤原细分' if body.get('weft_original_skin_mesh') else '缓存皮肤静态细分'
                    cuff.operator('weft.skin_cache',text=label)
                label='重做动作贴肤准备' if source.get(CACHE) else '准备动作贴肤'
                cuff.operator('weft.audit_binding',text=label).mode='MOTION'
        if tool:
            row=ui.row(align=True)
            row.operator('weft.live_control',text='续播' if tool.monitor_pause is not None else '暂停').action='PAUSE'
            if tool.monitor_pause is not None: row.operator('weft.live_control',text='单步').action='STEP'
            ui.operator('weft.live_control',text='结束并封存当前袜面').action='FINISH'
            ui.label(text=f'破口 {len(tool.sim.cracks)} · Space 暂停 · → 单步')
            ui.label(text='拉两孔间的布桥可贯通，拉孔尖可延长' if getattr(tool,'continued',False) else '左键抓放 · W 加水（湿态） · Shift 左键取样')
        elif frozen:
            if frozen.get('weft_live_tears',0) and frozen.get('weft_snapshot_mode','dry')=='dry' and not frozen.get('weft_live_wet'):
                ui.operator('weft.live_grab',text='接着撕当前缺口').resume=True
            else:ui.label(text='已封存当前袜面')
        else:
            row=ui.row(align=True)
            row.operator('weft.live_grab',text='干态抓取')
            row.operator('weft.live_grab',text='湿态抓取 / 泼水').wet_mode=True
            ui.prop(s,'weft_motion_speed')
            ui.operator('weft.live_grab',text='随骨架动作抓取').motion=True
        if frozen: ui.operator('weft.live_reset')
        ui.label(text='保存会结束实时求解；时间轴仅重放已绑定缓存')
        if source and (tool or frozen is None):
            box=ui.box(); box.label(text='袜面参数 · 原精度')
            for name in ('palette','thickness','knit_scale','wrinkles','recovery','break_extension',
                         'run','pretension','fit','idle','volume'):
                box.prop(source.weft_fabric,name)


CLASSES=(WEFT_PG_fabric,WEFT_OT_control,WEFT_PT_session)
def register():
    for cls in CLASSES: bpy.utils.register_class(cls)
    bpy.types.Object.weft_fabric=bpy.props.PointerProperty(type=WEFT_PG_fabric)
def unregister():
    stop_audio()
    del bpy.types.Object.weft_fabric
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)
