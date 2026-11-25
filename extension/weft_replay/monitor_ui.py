"""05 viewport controls. Register these classes from the development launcher or 06."""
import time
from math import isfinite

import blf
import bpy
import gpu
import numpy as np
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from .live_view import ACTIVE
from .monitor import INVENTORY, Monitor
from .wet_display import make_object
from .wet_film import DT
from .wet_scene import TRIANGLES


MAPS = {
    'OFF': ('袜面外观', '', ()),
    'WALE': ('伸长率 · 纵向', '≤ −25%　0%　≥ +100%', ('wale_pct', 'course_pct', 'single_course_pct', 'patch_course_pct')),
    'COURSE': ('伸长率 · 横向', '≤ −25%　0%　≥ +100%', ('wale_pct', 'course_pct', 'single_course_pct', 'patch_course_pct')),
    'GAP': ('离肤／损伤', '≤ −0.10　0　≥ +15 mm　紫色＝孔边', ('gap_mm', 'slip_mm', 'yarn_transfer_mm')),
    'OUTER': ('浸润 · 袜外膜', '0 → ≥ 0.50 mm', ('outer_mm',)),
    'KNIT': ('浸润 · 织层', '0 → 100%', ('knit_pct',)),
    'SKIN': ('浸润 · 皮肤侧', '0 → ≥ 0.50 mm', ('skin_mm', 'skin_anchor_mm')),
}

HISTORY = {
    'wale_pct': ('纵向', (1., .58, .19, 1.)),
    'course_pct': ('横向', (.35, .65, 1., 1.)),
    'single_course_pct': ('单边', (.85, .69, 1., 1.)),
    'patch_course_pct': ('六边', (.29, .94, .85, 1.)),
    'gap_mm': ('间隙', (.35, .65, 1., 1.)),
    'slip_mm': ('滑移', (.29, .94, .85, 1.)),
    'yarn_transfer_mm': ('过纱', (1., .58, .19, 1.)),
    'outer_mm': ('外膜', (.35, .65, 1., 1.)),
    'skin_mm': ('最近皮肤', (.72, .62, 1., 1.)),
    'skin_anchor_mm': ('皮肤锚点', (1., .58, .19, 1.)),
    'knit_pct': ('织层', (.29, .94, .85, 1.)),
}

NEUTRAL = (.23, .25, .32)
COMPRESSION = (.20, .48, 1.)
EXTENSION = (1., .45, .12)
PENETRATION = (1., .29, .13)
LIFT = (.20, .83, .98)
WATER = (.23, .72, 1.)
KNIT = (.21, .91, .59)
SKIN = (.49, .43, 1.)
INVALID = (.31, .32, .35)
FILM_RANGE_MM = .5
SKIN_DETAIL_MM = .001
SOCK_LAYERS = ('OUTER', 'KNIT')
MATERIAL_PREVIEW_SHADING = {'MATERIAL', 'RENDERED'}
SOCK_LAYER_NAME = 'Weft • sock layer map'
MONITOR_OWNER = 'weft_monitor_owner'


def remove_sock_layer(obj):
    mesh = obj.data
    material = mesh.materials[0]
    bpy.data.objects.remove(obj, do_unlink=True)
    if not mesh.users:
        bpy.data.meshes.remove(mesh)
    if not material.users:
        bpy.data.materials.remove(material)


class SockLayerDisplay:
    """Let scene depth occlude the wet map in shaded viewports."""
    color_name = 'Weft layer color'

    def __init__(self, scene, owner):
        material = bpy.data.materials.new('Weft • layer map')
        material.use_nodes = True
        material.surface_render_method = 'BLENDED'
        material.use_transparency_overlap = False
        nodes, link = material.node_tree.nodes, material.node_tree.links.new
        nodes.clear()
        color = nodes.new('ShaderNodeVertexColor')
        color.layer_name = self.color_name
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Strength'].default_value = 1.3
        link(color.outputs['Color'], emission.inputs['Color'])
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs[0].default_value = .76
        link(nodes.new('ShaderNodeBsdfTransparent').outputs[0], mix.inputs[1])
        link(emission.outputs[0], mix.inputs[2])
        link(mix.outputs[0], nodes.new('ShaderNodeOutputMaterial').inputs['Surface'])
        self.obj = make_object(scene, SOCK_LAYER_NAME, np.empty((0, 3)),
                               np.empty((0, 3), dtype=np.int32), material, owner)
        self.obj[MONITOR_OWNER] = owner
        self.obj.hide_render = True
        self.obj.hide_select = True
        self.faces = None

    def update(self, xyz, faces, rgba):
        mesh = self.obj.data
        if not np.array_equal(self.faces, faces):
            mesh.clear_geometry()
            mesh.from_pydata(xyz.tolist(), [], faces.tolist())
            self.faces = faces
        else:
            mesh.vertices.foreach_set('co', np.ascontiguousarray(xyz, dtype=np.float32).ravel())
        colors = mesh.color_attributes.get(self.color_name)
        if colors is None:
            colors = mesh.color_attributes.new(name=self.color_name, type='FLOAT_COLOR', domain='POINT')
        colors.data.foreach_set('color', np.ascontiguousarray(rgba, dtype=np.float32).ravel())
        mesh.update()
        if self.obj.hide_get():
            self.obj.hide_set(False)

    def remove(self):
        remove_sock_layer(self.obj)


def refresh_sock_layer(tool, scene):
    if tool.monitor is None:
        return
    field = scene.weft_monitor_map
    preview = tool.sock_layer_display
    data = None
    if field in SOCK_LAYERS and tool.monitor_view.shading.type in MATERIAL_PREVIEW_SHADING:
        data = map_data(tool.monitor, field)
    if data is None or not len(data[1]):
        if preview is not None and not preview.obj.hide_get():
            preview.obj.hide_set(True)
        return
    xyz, faces, rgba = data
    if preview is None:
        tool.sock_layer_display = preview = SockLayerDisplay(scene, tool.obj)
    # Small depth separation for coplanar triangles. Blender's material pass
    # handles foreground occlusion before blending the diagnostic color.
    eye = np.asarray(tool.rv.view_matrix.inverted().col[2].xyz)
    preview.update(xyz + .00003*eye, faces, rgba)


def attach(tool):
    tool.publish_pending_surface()
    tool.monitor = Monitor(tool.sim, tool.obj['weft_live_source'].name,
                           getattr(tool.sim, 'monitor_contact', None), tool.committed_pose())
    tool.monitor.capture(tool.wet)
    tool.monitor_view = tool.area.spaces.active
    tool.sock_layer_display = None
    if tool.wet_mode:
        bpy.msgbus.subscribe_rna(key=(bpy.types.View3DShading, 'type'), owner=tool,
                                 args=(tool, bpy.context.scene), notify=refresh_sock_layer)
        refresh_sock_layer(tool, bpy.context.scene)
    tool.monitor_handlers = [
        bpy.types.SpaceView3D.draw_handler_add(draw_surface, (tool,), 'WINDOW', 'POST_VIEW'),
        bpy.types.SpaceView3D.draw_handler_add(draw_inset, (tool,), 'WINDOW', 'POST_PIXEL'),
    ]
    tool.area.tag_redraw()


def detach(tool):
    bpy.msgbus.clear_by_owner(tool)
    if tool.sock_layer_display is not None:
        tool.sock_layer_display.remove()
    for handler in tool.monitor_handlers:
        bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
    del tool.sim.monitor_interpolate
    del tool.sim.monitor_interpolate_faces
    tool.monitor = None


def pause(tool):
    now = time.perf_counter()
    if tool.monitor_pause is None:
        tool.monitor_pause = now
        tool.refresh_monitor_snapshot()
    else:
        wait = now-tool.monitor_pause
        tool.last += wait
        tool.started += wait
        tool.monitor_pause = None
        release_if_pending(tool)
    tool.area.tag_redraw()


def release_if_pending(tool, *, commit=True):
    if tool.monitor_released:
        tool.monitor_released = False
        tool.release_grip(commit=commit)


def single_step(tool, context):
    if tool.monitor_pause is None:
        return True
    release_if_pending(tool, commit=False)
    dt = DT if tool.wet_mode else (1/45 if tool.sim.pose is not None else 1/90)
    now = tool.last+dt
    tool.monitor_pause += dt
    xy = tool.held_cursor()
    if tool.sim.pose is not None:
        return tool.advance_motion(context, xy, now=now,
                                   pointer_changed=tool.wet_mode and tool.wet_pointer_changed)
    if tool.sim.grip is None and not tool.sim.moving:
        tool.sim.step(dt)
        tool.last = now
        tool.capture_monitor()
    else:
        tool.tick(tool.target_at(xy) if xy is not None else None, now)
    return True


def pick_at(tool, context, xy):
    tool.publish_pending_surface()
    origin, direction = tool.pick_ray(xy)
    hit, location, face, obj = tool.pick_visible(context, origin, direction)
    if hit and obj == tool.obj and face >= 0:
        from .live_contact import barycentric

        tri = np.array([tool.obj.data.vertices[i].co for i in tool.obj.data.polygons[face].vertices])
        if np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0])) > 1e-12:
            tool.monitor.select(face, barycentric(tri, np.asarray(location)))
            tool.area.tag_redraw()
            return
    tool.monitor.anchor = None
    tool.monitor.edges = []
    tool.monitor.history.clear()
    tool.monitor.capture(tool.wet)
    tool.area.tag_redraw()


def event(tool, context, e, xy):
    if e.type == 'LEFTMOUSE' and e.value == 'RELEASE' and tool.monitor_skip_release:
        tool.monitor_skip_release = False
        return {'RUNNING_MODAL'}
    in_view = tool.pointer_in_view(xy) if e.type in {'LEFTMOUSE', 'MOUSEMOVE', 'W', 'SPACE', 'RIGHT_ARROW'} else False
    if in_view and e.value == 'PRESS':
        if e.type == 'SPACE':
            pause(tool)
            return {'RUNNING_MODAL'}
        if e.type == 'RIGHT_ARROW' and tool.monitor_pause is not None:
            return {'RUNNING_MODAL'} if single_step(tool, context) else {'FINISHED'}
        if e.type == 'LEFTMOUSE' and e.shift and getattr(tool,'monitor',None) is not None:
            pick_at(tool, context, xy)
            tool.monitor_skip_release = True
            return {'RUNNING_MODAL'}
    if tool.monitor_pause is not None:
        if e.type == 'LEFTMOUSE' and e.value == 'RELEASE' and tool.sim.grip is not None:
            tool.monitor_released = True
        if (e.type == 'TIMER' or
                (e.type == 'MOUSEMOVE' and tool.sim.grip is not None) or
                (in_view and e.type in {'LEFTMOUSE', 'W'})):
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}
    return None


def colors(values, lower, upper, cool, warm=None):
    values = np.asarray(values)
    finite = np.isfinite(values)
    v = np.where(finite, values, 0.)
    t = np.clip(v/upper, 0., 1.)[:, None]
    rgb = np.array(NEUTRAL)*(1-t) + np.array(cool)*t
    if warm is not None:
        t = np.clip(v/lower, 0., 1.)[:, None]
        rgb = rgb*(1-t) + np.array(warm)*t
    rgb[~finite] = INVALID
    return np.column_stack((rgb, np.full(len(rgb), .86)))


def map_data(monitor, field, *, skin_detail=False):
    """Coordinates in display space. Numeric readings remain on the solver geometry."""
    sim = monitor.sim
    if field in {'WALE', 'COURSE'}:
        strain = monitor.surface.stretch_map()[:, int(field=='WALE')]
        xyz = sim.display_positions()[sim.display_faces()].reshape(-1, 3)
        return xyz, None, np.repeat(colors(strain, -25, 100, EXTENSION, COMPRESSION), 3, axis=0)
    if field == 'GAP' and (gap := monitor.gaps()) is not None:
        return sim.display_positions(), sim.display_faces(), colors(gap, -.1, 15, LIFT, PENETRATION)
    if field in {'OUTER', 'KNIT', 'SKIN'} and monitor.wet_fields is not None:
        group = monitor.wet_fields['skin' if field == 'SKIN' else 'sock']
        tris = TRIANGLES[group['valid'][TRIANGLES].all(1)]
        if field == 'SKIN':
            upper = SKIN_DETAIL_MM if skin_detail else FILM_RANGE_MM
            return group['xyz_m'], tris, colors(group['film_mm'], 0., upper, SKIN)
        if field == 'OUTER':
            return group['xyz_m'], tris, colors(group['outer_mm'], 0., FILM_RANGE_MM, WATER)
        return group['xyz_m'], tris, colors(group['saturation'], 0., 1., KNIT)
    return None


def draw_surface(tool):
    if bpy.context.region != tool.region:
        return
    monitor = tool.monitor
    scene = bpy.context.scene
    field = scene.weft_monitor_map
    if tool.advancing:
        return
    if field in SOCK_LAYERS and tool.monitor_view.shading.type in MATERIAL_PREVIEW_SHADING:
        return
    data = map_data(monitor, field, skin_detail=scene.weft_monitor_skin_detail)
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)
    try:
        if data is not None:
            xyz, tris, rgba = data
            if len(xyz) and (tris is None or len(tris)):
                # View bias keeps the map just above the selected textile or
                # wet lattice. Depth still hides the far side of the cloth.
                eye = np.asarray(tool.rv.view_matrix.inverted().col[2].xyz)
                gpu.state.depth_test_set('NONE' if field == 'SKIN' else 'LESS_EQUAL')
                shader = gpu.shader.from_builtin('SMOOTH_COLOR')
                shader.bind()
                indices = None if tris is None else tris.astype(np.int32, copy=False)
                # The GPU vertex format is F32. NumPy buffers must match it.
                position = np.ascontiguousarray(xyz + .00015*eye, dtype=np.float32)
                vertex_color = np.ascontiguousarray(rgba, dtype=np.float32)
                batch_for_shader(shader, 'TRIS', {'pos': position, 'color': vertex_color}, indices=indices).draw(shader)
        gpu.state.depth_test_set('LESS_EQUAL')
        if field == 'GAP':
            for _, rails, _ in monitor.sim.damage_geometry():
                for rail in rails:
                    line(rail, (.94, .33, 1., 1.), .00035*eye if data is not None else 0.)
        if field in {'WALE', 'COURSE'}:
            position = monitor.sim.display_positions()
            for edge, length in zip(monitor.edges, monitor.edge_lengths()):
                if length is not None:
                    line([a.point(position) for a in edge], (.22, 1., .90, 1.), .00025*eye)
    finally:
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set('NONE')


def line(points, color, offset=0.):
    if len(points) < 2:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    shader.uniform_float('color', color)
    position = np.asarray(np.asarray(points)+offset, dtype=np.float32)
    if position.shape[1] == 2:
        position = np.column_stack((position, np.zeros(len(position), dtype=np.float32)))
    batch_for_shader(shader, 'LINE_STRIP', {'pos': position}).draw(shader)


def text(x, y, value, size=13, rgba=(.92, .94, 1., 1.)):
    blf.size(0, size)
    blf.color(0, *rgba)
    blf.position(0, x, y, 0)
    blf.draw(0, value)


def graph(rows, keys, width):
    rows = [row for row in rows if row['time_s'] >= rows[-1]['time_s']-6]
    series = [[r['probe'].get(key) for r in rows] for key in keys]
    values = [v for data in series for v in data if v is not None and isfinite(v)]
    if not values:
        text(22, 58, 'No probe history')
        return
    low, high = min(0., min(values)), max(.001, max(values))
    span = max(rows[-1]['time_s']-rows[0]['time_s'], .001)
    x = [44+(width-53)*(r['time_s']-rows[0]['time_s'])/span for r in rows]
    line([(44, 38), (44, 131), (width-9, 131)], (.48, .55, .64, .8))
    zero = 38-93*low/(high-low)
    line([(44, zero), (width-9, zero)], (.48, .55, .64, .8))
    text(15, 125, f'{high:.2g}', 10)
    text(15, 37, f'{low:.2g}', 10)
    text(46, 21, f'{rows[0]["time_s"]:.3f} s', 10)
    text(width-63, 21, f'{rows[-1]["time_s"]:.3f} s', 10)
    label = {'wale_pct':'Wale', 'course_pct':'Course', 'single_course_pct':'Edge', 'patch_course_pct':'Six',
             'gap_mm':'Gap', 'slip_mm':'Slip', 'yarn_transfer_mm':'Yarn', 'outer_mm':'Outer',
             'skin_mm':'Near', 'skin_anchor_mm':'Anchor', 'knit_pct':'Knit'}
    for i, (key, data) in enumerate(zip(keys, series)):
        color = HISTORY[key][1]
        text(44+52*i, 140, label[key], 11, color)
        run = []
        for px, v in zip(x, data):
            if v is None:
                line(run, color)
                run.clear()
            else:
                run.append((px, 38+93*(v-low)/(high-low)))
        line(run, color)


def draw_inset(tool):
    if bpy.context.region != tool.region or tool.advancing:
        return
    monitor = tool.monitor
    r = monitor.row
    probe = r['probe']
    scene = bpy.context.scene
    markers = [
        ('displayed_xyz_m', 9, (.21, 1., .83, 1.)),
        ('skin_anchor_xyz_m', 6, (1., .58, .19, 1.)),
    ]
    if scene.weft_monitor_map == 'SKIN':
        markers.append(('nearest_skin_xyz_m', 4, HISTORY['skin_mm'][1]))
    for key, radius, color in markers:
        if key in probe:
            center = location_3d_to_region_2d(tool.region, tool.rv, Vector(probe[key]))
            if center is not None:
                angle = np.linspace(0., 2*np.pi, 33)
                line(np.column_stack((np.cos(angle), np.sin(angle)))*radius+np.asarray(center), color)
    if not scene.weft_monitor_history:
        return
    field = scene.weft_monitor_map
    keys = MAPS[field][2] or MAPS['GAP'][2]
    width = min(304, tool.region.width-24)
    if width < 180:
        return
    gpu.state.blend_set('ALPHA')
    try:
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        shader.bind()
        shader.uniform_float('color', (.035, .047, .069, .93))
        batch_for_shader(shader, 'TRI_FAN', {'pos': [(10, 10, 0), (width, 10, 0), (width, 176, 0), (10, 176, 0)]}).draw(shader)
        units = '%' if keys[0].endswith('pct') else 'mm'
        text(21, 155, f'Step {r["step"]}   ·   {r["time_s"]:.3f} s   ·   {units}')
        if monitor.history:
            graph(list(monitor.history), keys, width)
        else:
            text(22, 64, 'Shift + click the stocking')
    finally:
        gpu.state.blend_set('NONE')


def measure(layout, label, value, unit, digits=3):
    suffix = '暂无数据' if value is None else f'{value:+.{digits}f} {unit}'
    layout.label(text=f'{label}　{suffix}')


def wet_probe(layout, row):
    data, probe = row['wet'], row['probe']
    box = layout.box()
    box.label(text='浸润探针 · 青圈袜面')
    for label, key, unit, factor, digits in (
        ('袜外膜', 'outer_mm', 'mm', 1., 6), ('织层饱和度', 'knit_pct', '%', 1., 3),
        ('最近皮肤膜 · 紫圈', 'skin_mm', 'µm', 1000., 4), ('皮肤锚点膜 · 橙圈', 'skin_anchor_mm', 'µm', 1000., 4),
    ):
        value = probe.get(key)
        measure(box, label, None if value is None else factor*value, unit, digits)
    if data is not None:
        box.label(text='本步两侧交换')
        measure(box, '织层 → 皮肤', 1000*data['knit_to_skin_ml'], 'µL', 6)
        measure(box, '皮肤 → 织层', 1000*data['skin_to_knit_ml'], 'µL', 6)


class WEFT_OT_monitor(bpy.types.Operator):
    bl_idname = 'weft.monitor'
    bl_label = '监测控制'
    action: bpy.props.EnumProperty(items=[(s, s, '') for s in ('ATTACH', 'PAUSE', 'STEP')])

    def execute(self, context):
        tool = ACTIVE.get(context.scene.as_pointer())
        if tool is None:
            return {'CANCELLED'}
        if self.action == 'ATTACH' and getattr(tool, 'monitor', None) is None:
            if getattr(tool,'continued',False):return {'CANCELLED'}
            attach(tool)
        elif self.action == 'PAUSE':
            pause(tool)
        elif self.action == 'STEP':
            if not single_step(tool, context):
                return {'CANCELLED'}
        return {'FINISHED'}


class WEFT_PT_monitor(bpy.types.Panel):
    bl_label = '目标袜面 · 实时监测'
    bl_idname = 'WEFT_PT_monitor'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Weft Lab'
    bl_order = -5
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        s = context.scene
        ui = self.layout
        tool = ACTIVE.get(s.as_pointer())
        if tool is None:
            ui.label(text='先从上方启动目标袜面的实时交互')
            return
        if getattr(tool,'continued',False):
            ui.label(text=f'当前网格自由边界　{len(tool.sim.cracks)} 处断口')
            for h in tool.sim.fracture_readings()['holes']:
                ui.label(text=f'孔 {h["id"]}　{h["gap_mm"]:.1f} mm　{h["boundary_vertices"]} 个边缘顶点')
            return
        monitor = getattr(tool, 'monitor', None)
        if monitor is None:
            ui.operator('weft.monitor', text='监测当前袜面').action = 'ATTACH'
            return
        r = monitor.row
        p = r['probe']
        ui.label(text='实时目标角色')
        ui.label(text=r['object'])
        ui.label(text=f'物理步 {r["step"]}　{r["time_s"]:.4f} s　姿势帧 {r["pose_frame"][0]}')
        controls = ui.row(align=True)
        paused = tool.monitor_pause is not None
        controls.operator('weft.monitor', text='续播' if paused else '暂停').action = 'PAUSE'
        if paused:
            controls.operator('weft.monitor', text='单步').action = 'STEP'
        ui.label(text='Shift + 左键取样 · Space 暂停 · → 单步')
        ui.prop(s, 'weft_monitor_map', text='物理图')
        field = s.weft_monitor_map
        if MAPS[field][1]:
            legend = '0 → ≥ 1 µm' if field == 'SKIN' and s.weft_monitor_skin_detail else MAPS[field][1]
            ui.label(text=legend)
            if field in {'OUTER', 'KNIT', 'SKIN'} and monitor.wet_fields is None:
                ui.label(text='当前物理图　暂无数据')
        if field == 'SKIN':
            ui.prop(s, 'weft_monitor_skin_detail', text='放大皮肤微膜色标')
        ui.prop(s, 'weft_monitor_history', text='探针历史')
        if tool.wet_mode:
            wet_probe(ui, r)
        ui.label(text='参考构形 · 启用监测时的贴合姿势')
        if not p:
            ui.label(text='探针失效 · 暂无数据' if monitor.anchor else '点选袜面 · 暂无数据')
        else:
            ui.label(text=f'材料 U {p["uv"][0]:.6f}　纵向 V {p["uv"][1]:.3f}')
            ui.label(text=f'三角面 {p["face"]}　织圈 {p["cell"][0]}, {p["cell"][1]}')
            if p['cell_failed']:
                ui.label(text='线圈已失效 · 局部伸长率暂无数据')
            for bank in p['banks']:
                ui.label(text=f'孔 {bank["hole_id"]}　{bank["side"]} 侧')
            for label, key, unit in (
                ('纵向伸长率', 'wale_pct', '%'), ('横向伸长率', 'course_pct', '%'),
                ('单边横伸', 'single_course_pct', '%'), ('青色六边加权横伸', 'patch_course_pct', '%'),
                ('切向滑移', 'slip_mm', 'mm'), ('最近皮肤间隙', 'gap_mm', 'mm'), ('圈间过纱', 'yarn_transfer_mm', 'mm'),
            ):
                measure(ui, label, p.get(key), unit)
            if p.get('contact', {}).get('opposing_polygon') is not None:
                ui.label(text='最近皮肤面存在反向法线')
        whole = monitor.whole_sock()
        measure(ui, '全袜最小间隙', whole and whole['min_mm'], 'mm')
        ui.label(text='穿入 > 0.1 mm　'+(str(whole['inside']) if whole else '暂无数据'))
        ui.label(text=f'破口 {len(r["holes"])}　失效线圈 {r["failed_cells"]}')
        if r['wet'] is not None:
            box = ui.box()
            box.label(text='当前步液量收支')
            for key, label in zip(INVENTORY, ('待入射', '空中', '袜外膜', '织层', '皮肤侧', '流出', '蒸发')):
                if key == 'skin_ml':
                    measure(box, label, 1000*r['wet'][key], 'µL', 6)
                else:
                    measure(box, label, r['wet'][key], 'mL', 6)
            box.label(text=f'误差　{r["wet"]["balance_error_ml"]:+.2e} mL')


CLASSES = (WEFT_OT_monitor, WEFT_PT_monitor)


def redraw_map(scene, context):
    tool = ACTIVE.get(scene.as_pointer())
    if tool is not None and tool.wet_mode and getattr(tool, 'monitor', None) is not None:
        refresh_sock_layer(tool, scene)
    if context.area:
        context.area.tag_redraw()


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.weft_monitor_map = bpy.props.EnumProperty(
        items=[(key, item[0], item[1]) for key, item in MAPS.items()], update=redraw_map)
    bpy.types.Scene.weft_monitor_skin_detail = bpy.props.BoolProperty(default=False, update=redraw_map)
    bpy.types.Scene.weft_monitor_history = bpy.props.BoolProperty(default=True, update=redraw_map)


def unregister():
    for tool in list(ACTIVE.values()):
        if getattr(tool, 'monitor', None) is not None:
            detach(tool)
    del bpy.types.Scene.weft_monitor_history
    del bpy.types.Scene.weft_monitor_skin_detail
    del bpy.types.Scene.weft_monitor_map
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
