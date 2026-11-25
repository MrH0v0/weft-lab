"""Dry stocking appearance driven by the knit chart and local loop gauges."""
import math

import bpy
import numpy as np


PALETTES = (
    (.014, .011, .016, 1),  # Black
    (.20, .205, .20, 1),   # Smoke
    (.51, .35, .27, 1),    # Skin
    (.79, .77, .72, 1),    # White
)
TOE_FINISH_ATTRIBUTE = "Weft Toe Finish"


def controls(material, palette, thickness):
    tree = bpy.data.node_groups.get("Weft • Dry stocking controls")
    if tree is None:
        tree = bpy.data.node_groups.new("Weft • Dry stocking controls", "ShaderNodeTree")
        dye = tree.interface.new_socket(name="Palette", in_out="INPUT", socket_type="NodeSocketFloat")
        dye.default_value, dye.min_value, dye.max_value = 0, 0, 3
        dye.description = "0 黑色 · 1 烟灰 · 2 肤色 · 3 白色"
        gauge = tree.interface.new_socket(name="Thickness / Opacity", in_out="INPUT", socket_type="NodeSocketFloat")
        gauge.default_value, gauge.min_value, gauge.max_value = .64, 0, 1
        gauge.description = "纱线的光学厚度，受拉时随有效纱线面密度降低"
        tree.interface.new_socket(name="Yarn color", in_out="OUTPUT", socket_type="NodeSocketColor")
        tree.interface.new_socket(name="Thickness / Opacity", in_out="OUTPUT", socket_type="NodeSocketFloat")
        entry = tree.nodes.new("NodeGroupInput")
        entry.location = (-590, 20)
        quantize = tree.nodes.new("ShaderNodeMath")
        quantize.operation = "ROUND"
        quantize.location = (-400, 20)
        index = tree.nodes.new("ShaderNodeMath")
        index.operation = "MULTIPLY_ADD"
        index.inputs[1].default_value = .25
        index.inputs[2].default_value = .125
        index.location = (-220, 20)
        colors = tree.nodes.new("ShaderNodeValToRGB")
        colors.location = (-40, 20)
        ramp = colors.color_ramp
        ramp.interpolation = "CONSTANT"
        ramp.elements[1].position = .75
        ramp.elements.new(.25)
        ramp.elements.new(.50)
        for stop, color in zip(ramp.elements, PALETTES):
            stop.color = color
        exit_node = tree.nodes.new("NodeGroupOutput")
        exit_node.location = (250, 20)
        tree.links.new(entry.outputs["Palette"], quantize.inputs[0])
        tree.links.new(quantize.outputs[0], index.inputs[0])
        tree.links.new(index.outputs[0], colors.inputs[0])
        tree.links.new(colors.outputs[0], exit_node.inputs["Yarn color"])
        tree.links.new(entry.outputs["Thickness / Opacity"], exit_node.inputs["Thickness / Opacity"])
    node = material.node_tree.nodes.new("ShaderNodeGroup")
    node.node_tree = tree
    node.name = "Weft Dry Appearance"
    node.label = "干袜 · 配色 / 厚度"
    node.inputs["Palette"].default_value = palette
    node.inputs["Thickness / Opacity"].default_value = thickness
    return node


def shade_dry_fabric(material, shader, channels, lookup, head, neck, row, math_op, palette, thickness):
    nodes = material.node_tree.nodes
    link = material.node_tree.links.new
    settings = controls(material, palette, thickness)
    link(settings.outputs["Yarn color"], shader.inputs["Base Color"])
    geometry = nodes.new("ShaderNodeNewGeometry")

    def view_dot(vector):
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        link(vector, dot.inputs[0])
        link(geometry.outputs["Incoming"], dot.inputs[1])
        return math_op("ABSOLUTE", dot.outputs["Value"])

    def saturate(value):
        return math_op("MINIMUM", 1, math_op("MAXIMUM", 0, value))

    facing = view_dot(geometry.outputs["Normal"])
    density = math_op("MULTIPLY", 2, lookup.outputs["Alpha"])
    # Attenuate by yarn per deformed area with a shallow grazing-angle term.
    grazing = math_op("SUBTRACT", 1.11, math_op("MULTIPLY", .14, facing))
    toe_detail = saturate(math_op("DIVIDE", math_op("ADD", row, 23), 13))
    toe_finish = nodes.new("ShaderNodeAttribute")
    toe_finish.attribute_name = TOE_FINISH_ATTRIBUTE
    finish = saturate(toe_finish.outputs["Fac"])
    toe_detail = math_op("MULTIPLY", toe_detail, math_op("SUBTRACT", 1, math_op("MULTIPLY", .82, finish)))
    closure = math_op("SUBTRACT", 1.19, math_op("MULTIPLY", .19, toe_detail))
    closure = math_op("ADD", closure, math_op("MULTIPLY", .22, finish))
    depth = math_op("ADD", .20, math_op("MULTIPLY", 2.76, settings.outputs["Thickness / Opacity"]))
    extinction = math_op("MULTIPLY", math_op("MULTIPLY", density, grazing), math_op("MULTIPLY", depth, closure))
    alpha = math_op("SUBTRACT", 1, math_op("POWER", math.e, math_op("MULTIPLY", -1, extinction)))
    # The mipmapped stitch image adds pore contrast over the mean coverage.
    structure = math_op("SUBTRACT", channels.outputs["Blue"], .315)
    detail_strength = math_op("ADD", .055, math_op("MULTIPLY", .35, toe_detail))
    alpha = math_op("ADD", alpha, math_op("MULTIPLY", structure, detail_strength))
    link(math_op("MINIMUM", .98, math_op("MAXIMUM", 0, alpha)), shader.inputs["Alpha"])

    tangent = nodes.new("ShaderNodeTangent")
    tangent.direction_type = "UV_MAP"
    tangent.uv_map = "Knit"
    link(tangent.outputs["Tangent"], shader.inputs["Tangent"])
    alignment = saturate(math_op("ABSOLUTE", math_op("SUBTRACT", head, neck)))
    directional = math_op("MULTIPLY", alignment, math_op("SUBTRACT", 1, view_dot(tangent.outputs["Tangent"])))
    compression = saturate(math_op("MULTIPLY", 2, math_op("SUBTRACT", density, 1)))
    working_fibre = math_op("MAXIMUM", compression, alignment)
    fold_reflection = math_op("SUBTRACT", 1, math_op("MULTIPLY", .76, working_fibre))
    fold_reflection = math_op("MULTIPLY", fold_reflection,
                              math_op("SUBTRACT", 1, math_op("MULTIPLY", .60, finish)))
    roughness = math_op("ADD", .65, math_op("MULTIPLY", .19, working_fibre))
    link(math_op("ADD", roughness, math_op("MULTIPLY", .10, finish)), shader.inputs["Roughness"])
    specular = math_op("ADD", .17, math_op("MULTIPLY", .05, directional))
    sheen = math_op("ADD", .10, math_op("MULTIPLY", .05, directional))
    link(math_op("MULTIPLY", specular, fold_reflection), shader.inputs["Specular IOR Level"])
    link(math_op("MULTIPLY", sheen, fold_reflection), shader.inputs["Sheen Weight"])
    link(math_op("ADD", .25, math_op("MULTIPLY", .40, alignment)), shader.inputs["Anisotropic"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = .000045
    link(math_op("ADD", .085, math_op("MULTIPLY", .16, toe_detail)), bump.inputs["Strength"])
    link(channels.outputs["Green"], bump.inputs["Height"])
    link(bump.outputs["Normal"], shader.inputs["Normal"])


def loop_density(loops):
    """Relative projected yarn stock per unit area of a deformed stitch."""
    shear = np.clip(loops.rest_angle + loops.lean/.32, -.99, .99)
    angle_ratio = np.sqrt((1-shear**2) / np.maximum(1-loops.rest_angle**2, .01))
    area = loops.stretch.prod(axis=1)*angle_ratio
    stock = loops.stock/np.maximum(loops.rest_length, 1e-9)
    return np.clip(stock/np.maximum(area, .20), .30, 1.98)


def optical_density(density, displacement_m):
    """Increase dry coverage contrast around relaxed density for light pulls."""
    offset = density - 1
    falloff = np.maximum(0, 1 - np.abs(offset)/.70)
    # Once the patch seats, use its ordinary yarn coverage and eliminate the
    # extra contrast on the smaller strains left by the skin-conforming fit.
    departure = np.clip(displacement_m/.004, 0, 1)
    amount = departure**2 * (3 - 2*departure)
    # The compact response leaves rest and the strong-deformation limits anchored.
    # A gain below three keeps the curve monotone through the shoulder.
    return density + 2.65*amount*offset*falloff**2


def patch_optical_density(sim):
    ids = sim.patch.ids
    settled = sim.rest[ids] + sim.opening[ids] + sim.seating[ids]
    departure = np.linalg.norm(sim.pos[ids]-settled, axis=1).max(initial=0)
    density = optical_density(loop_density(sim.patch.loops), departure)
    # Let a stretched stitch admit more light while keeping its intact yarn
    # visible. Retain a soft shoulder without flattening the tensile gradient.
    deficit = np.maximum(0., .92-density)
    return density + .42*deficit**2 / (deficit + .055)
