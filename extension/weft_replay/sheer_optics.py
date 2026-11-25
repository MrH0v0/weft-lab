"""A smooth, dry nylon finish over the existing knit and force-driven gauges."""
import math

import bpy


FINISH = 'Weft • Sheer nylon optics'
THREAD = 'Weft • Yarn profile'
GROUP = FINISH + ' · sheer palette'
REPEAT_LABEL = 'Fine yarn repeat · visual only'
VISUAL_REPEAT = 1.23


def math_socket(tree, operation, *values):
    node = tree.nodes.new('ShaderNodeMath')
    node.operation = operation
    for socket, value in zip(node.inputs, values):
        if isinstance(value, (float, int)):
            socket.default_value = value
        else:
            tree.links.new(value, socket)
    return node.outputs[0]


def optical_group():
    group = bpy.data.node_groups.get(GROUP)
    if group is not None:
        return group
    group = bpy.data.node_groups.new(GROUP, 'ShaderNodeTree')
    for name, default in (('Optical thickness', .64), ('Loop stock', .5),
                          ('Facing', 1.), ('Yarn coverage', .315), ('Toe closure', 0.)):
        socket = group.interface.new_socket(name=name,in_out='INPUT',socket_type='NodeSocketFloat')
        socket.default_value = default
    dye = group.interface.new_socket(name='Yarn dye',in_out='INPUT',socket_type='NodeSocketColor')
    dye.default_value = (.014,.011,.016,1.)
    for name in ('Coverage','Roughness','Reflection'):
        group.interface.new_socket(name=name,in_out='OUTPUT',socket_type='NodeSocketFloat')
    group.interface.new_socket(name='Fibre color',in_out='OUTPUT',socket_type='NodeSocketColor')
    entry = group.nodes.new('NodeGroupInput')
    exit = group.nodes.new('NodeGroupOutput')

    def calc(op, *values):
        return math_socket(group, op, *values)

    # The gauge is the mean area density. The mipmapped yarn profile varies
    # the coverage around that mean so pores retain contrast near the camera.
    stock = calc('MULTIPLY_ADD',entry.outputs['Loop stock'],.69,.655)
    luminance = group.nodes.new('ShaderNodeRGBToBW')
    group.links.new(entry.outputs['Yarn dye'],luminance.inputs[0])
    light_yarn = calc('MINIMUM',1.,calc('MAXIMUM',0.,calc('MULTIPLY_ADD',luminance.outputs[0],2.90,-.12)))
    thickness = calc('MULTIPLY_ADD',entry.outputs['Optical thickness'],3.27,.21)
    thickness = calc('MULTIPLY',thickness,calc('MULTIPLY_ADD',light_yarn,-.81,1.))
    edge = calc('SUBTRACT',1.,entry.outputs['Facing'])
    path = calc('MULTIPLY_ADD',calc('MULTIPLY',edge,edge),.30,1.)
    toe = calc('MULTIPLY_ADD',entry.outputs['Toe closure'],.35,1.)
    depth = calc('MULTIPLY',calc('MULTIPLY',stock,thickness),calc('MULTIPLY',path,toe))
    transmitted = calc('POWER',math.e,calc('MULTIPLY',depth,-1.))
    veil = calc('SUBTRACT',1.,transmitted)
    yarn = calc('SUBTRACT',entry.outputs['Yarn coverage'],.315)
    contrast = calc('MULTIPLY_ADD',entry.outputs['Facing'],.21,.105)
    fine = calc('MULTIPLY',yarn,calc('MULTIPLY_ADD',light_yarn,.31,contrast))
    coverage = calc('MINIMUM',.98,calc('MAXIMUM',0.,calc('ADD',veil,fine)))

    # Keep the broad surface matte. The remaining silk glint varies with
    # yarn coverage rather than brightening the spaces in the knit.
    roughness = calc('MULTIPLY_ADD',entry.outputs['Yarn coverage'],.10,
                     calc('MULTIPLY_ADD',light_yarn,.12,.57))
    silk = calc('MULTIPLY_ADD',entry.outputs['Yarn coverage'],.144,.065)
    reflection = calc('MULTIPLY',silk,calc('MULTIPLY_ADD',light_yarn,-.48,1.))
    tint = group.nodes.new('ShaderNodeVectorMath')
    tint.operation = 'MULTIPLY'
    tint.inputs[1].default_value = (1.04,.94,.83)
    group.links.new(entry.outputs['Yarn dye'],tint.inputs[0])
    dye_relief = group.nodes.new('ShaderNodeVectorMath')
    dye_relief.operation = 'SCALE'
    group.links.new(tint.outputs[0],dye_relief.inputs[0])
    # In pale yarn the adjacent pores and shadowed sides of each thread
    # still differ even when the underlying skin is close to the dye color.
    shade = calc('MULTIPLY_ADD',yarn,-.96,.78)
    scale = calc('MULTIPLY_ADD',light_yarn,calc('SUBTRACT',shade,1.),1.)
    group.links.new(scale,dye_relief.inputs['Scale'])
    for name, value in (('Coverage',coverage),('Roughness',roughness),('Reflection',reflection),
                        ('Fibre color',dye_relief.outputs[0])):
        group.links.new(value,exit.inputs[name])
    return group


def apply_sheer_finish(material):
    """Use the same dry finish on the authored Lisa template and generic stocking."""
    tree = material.node_tree
    if tree is None:
        return False
    nodes, link = tree.nodes, tree.links.new
    existing = nodes.get(FINISH)
    shader = nodes.get('Principled BSDF')
    settings = nodes.get('Weft Dry Appearance')
    if existing is not None and shader is not None and settings is not None:
        group = optical_group()
        if existing.node_tree != group:
            wires = {socket.name: socket.links[0].from_socket
                     for socket in existing.inputs if socket.is_linked}
            existing.node_tree = group
            for name, source in wires.items():
                link(source, existing.inputs[name])
            link(settings.outputs['Yarn color'],existing.inputs['Yarn dye'])
            link_optics(tree, shader, existing)
            repeat = next(node for node in nodes if node.label == REPEAT_LABEL)
            repeat.inputs['Scale'].default_value = VISUAL_REPEAT
            tune_fibre(tree, shader, shader.inputs['Normal'].links[0].from_node)
        return True
    gauge = next((node for node in nodes if node.type=='TEX_IMAGE' and
                  node.label=='Per-loop head, neck and shear'),None)
    thread = next((node for node in nodes if node.type=='TEX_IMAGE' and node.image and
                   (node.label==THREAD or node.image.name.startswith('live-loop-profile.png'))),None)
    if not (shader and settings and gauge and thread and thread.inputs['Vector'].is_linked):
        return False

    material.surface_render_method = 'BLENDED'
    material.use_transparency_overlap = False
    material.use_transparent_shadow = False
    repeat = nodes.new('ShaderNodeVectorMath')
    repeat.operation = 'SCALE'
    repeat.label = REPEAT_LABEL
    repeat.inputs['Scale'].default_value = VISUAL_REPEAT
    link(thread.inputs['Vector'].links[0].from_socket, repeat.inputs[0])
    link(repeat.outputs['Vector'],thread.inputs['Vector'])
    channels = nodes.new('ShaderNodeSeparateColor')
    link(thread.outputs['Color'],channels.inputs['Color'])

    geometry = nodes.new('ShaderNodeNewGeometry')
    angle = nodes.new('ShaderNodeVectorMath')
    angle.operation = 'DOT_PRODUCT'
    link(geometry.outputs['Normal'],angle.inputs[0])
    link(geometry.outputs['Incoming'],angle.inputs[1])
    facing = math_socket(tree,'ABSOLUTE',angle.outputs['Value'])
    toe = nodes.new('ShaderNodeAttribute')
    toe.attribute_name = 'Weft Toe Finish'
    optics = nodes.new('ShaderNodeGroup')
    optics.node_tree = optical_group()
    optics.name = FINISH
    optics.label = 'Sheer nylon · dry'
    optics.location = (shader.location.x-310,shader.location.y+70)
    link(settings.outputs['Thickness / Opacity'],optics.inputs['Optical thickness'])
    link(gauge.outputs['Alpha'],optics.inputs['Loop stock'])
    link(facing,optics.inputs['Facing'])
    link(channels.outputs['Blue'],optics.inputs['Yarn coverage'])
    link(toe.outputs['Fac'],optics.inputs['Toe closure'])

    link(settings.outputs['Yarn color'],optics.inputs['Yarn dye'])
    link_optics(tree, shader, optics)

    tangent = nodes.new('ShaderNodeTangent')
    tangent.direction_type, tangent.uv_map = 'UV_MAP','Knit'
    link(tangent.outputs['Tangent'],shader.inputs['Tangent'])
    bump = nodes.new('ShaderNodeBump')
    link(channels.outputs['Green'],bump.inputs['Height'])
    link(bump.outputs['Normal'],shader.inputs['Normal'])
    tune_fibre(tree, shader, bump)
    return True


def link_optics(tree, shader, optics):
    for output, name in (('Coverage','Alpha'), ('Roughness','Roughness'),
                         ('Reflection','Specular IOR Level'), ('Fibre color','Base Color')):
        tree.links.new(optics.outputs[output], shader.inputs[name])


def tune_fibre(tree, shader, bump):
    bump.inputs['Strength'].default_value = .24
    bump.inputs['Distance'].default_value = .000024
    bump.inputs['Filter Width'].default_value = .45
    for name, value in (('Anisotropic',.46),('Coat Weight',0.),('Sheen Weight',.041),
                        ('Sheen Roughness',.77),('Specular Tint',(.91,.86,.79,1.))):
        socket = shader.inputs[name]
        for wire in list(socket.links):
            tree.links.remove(wire)
        socket.default_value = value
