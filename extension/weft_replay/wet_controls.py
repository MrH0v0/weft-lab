"""04's standalone settings and viewport entry. Registration can move into 06."""
import bpy

from .wet_film import Liquid


def parameters(scene):
    p = scene.weft_wet
    return Liquid(volume=p.volume, radius=p.radius_mm/1000, viscosity=p.viscosity,
                  tension=p.tension, uptake=p.uptake, drying=p.drying,
                  wet_gain=p.wet_gain, color=tuple(p.color), opacity=p.opacity)


class WEFT_PG_wet(bpy.types.PropertyGroup):
    color: bpy.props.FloatVectorProperty(name='液色', subtype='COLOR', size=3, min=0., max=1., default=(.19, .59, .82))
    opacity: bpy.props.FloatProperty(name='不透明度', subtype='FACTOR', min=0., max=1., default=.8)
    volume: bpy.props.FloatProperty(name='水量 (mL)', min=.01, max=5., default=2.)
    radius_mm: bpy.props.FloatProperty(name='水片半宽／加水半径 (mm)', min=3., max=30., default=12.)
    viscosity: bpy.props.FloatProperty(name='黏度 (×)', min=.3, max=4., default=.6)
    tension: bpy.props.FloatProperty(name='表面张力系数', subtype='FACTOR', min=0., max=1., default=.65)
    uptake: bpy.props.FloatProperty(name='浸透速度 (×)', min=0., max=3., default=1.)
    drying: bpy.props.FloatProperty(name='干燥速度 (×)', min=0., max=4., default=1.)
    wet_gain: bpy.props.FloatProperty(name='湿摩擦增益', min=0., max=1., default=1.)
    direction: bpy.props.EnumProperty(name='来水方向', items=[('A', '＋横向', ''), ('B', '－横向', '')])
    side: bpy.props.EnumProperty(name='输入', items=[('SHEET', '泼水', ''), ('SKIN', '皮肤侧加水', ''),
                                                     ('OUTER', '袜外加水', ''), ('DRY', '干态标记', '')])
    moving: bpy.props.BoolProperty(name='跟随骨架动作', default=False)


class WEFT_PT_wet(bpy.types.Panel):
    bl_idname = 'WEFT_PT_wet'
    bl_label = '泼水与干湿参数'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Weft Lab'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = -7

    def draw(self, context):
        from .live_view import ACTIVE

        p = context.scene.weft_wet
        layout = self.layout
        tool = ACTIVE.get(context.scene.as_pointer())
        if tool and tool.wet is not None:
            layout.label(text='当前液体参数已固定；重置后应用新值')
            layout.enabled=False
        layout.prop(p, 'side')
        if p.side == 'SHEET':
            layout.prop(p, 'direction', expand=True)
        for key in ('volume', 'radius_mm', 'viscosity', 'tension', 'uptake', 'drying', 'color', 'opacity', 'wet_gain'):
            layout.prop(p, key)
        row = layout.row()
        row.enabled = tool is None
        row.prop(p, 'moving')


def register():
    bpy.utils.register_class(WEFT_PG_wet)
    bpy.types.Scene.weft_wet = bpy.props.PointerProperty(type=WEFT_PG_wet)
    bpy.utils.register_class(WEFT_PT_wet)


def unregister():
    bpy.utils.unregister_class(WEFT_PT_wet)
    del bpy.types.Scene.weft_wet
    bpy.utils.unregister_class(WEFT_PG_wet)
