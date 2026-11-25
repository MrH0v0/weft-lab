"""Bind the approved dry stocking finish to live loop gauges."""

import bpy
import numpy as np
from .sheer_optics import apply_sheer_finish


TEMPLATE = 'weft_live_finish'
GAUGE = 'Per-loop head, neck and shear'
COURSES = 144


def gauge_image(uv, name='Weft • local loop gauges'):
    first = int(np.floor(uv[:, 1].min()))
    height = int(np.floor(uv[:, 1].max())) - first + 1
    pixels = np.empty((height, COURSES, 4), np.float32)
    image = bpy.data.images.new(name, COURSES, height, float_buffer=True, is_data=True)
    image.alpha_mode = 'CHANNEL_PACKED'
    return first, pixels, image


def set_gauge(tree, lookup, uv, image, first):
    nodes, link = tree.nodes, tree.links.new
    split = nodes.new('ShaderNodeSeparateXYZ')
    link(uv, split.inputs[0])
    col = nodes.new('ShaderNodeMath')
    col.operation = 'DIVIDE'
    col.inputs[1].default_value = COURSES
    link(split.outputs['X'], col.inputs[0])
    rows = nodes.new('ShaderNodeMath')
    rows.operation = 'SUBTRACT'
    rows.inputs[1].default_value = first
    link(split.outputs['Y'], rows.inputs[0])
    height = nodes.new('ShaderNodeMath')
    height.operation = 'DIVIDE'
    height.inputs[1].default_value = image.size[1]
    link(rows.outputs[0], height.inputs[0])
    xyz = nodes.new('ShaderNodeCombineXYZ')
    link(col.outputs[0], xyz.inputs[0])
    link(height.outputs[0], xyz.inputs[1])
    lookup.image = image
    lookup.extension = 'REPEAT'
    link(xyz.outputs[0], lookup.inputs['Vector'])


def smooth_preview_alpha(material):
    tree=material.node_tree
    shader=tree.nodes.get('Principled BSDF')
    if material.surface_render_method!='DITHERED' or shader is None or not shader.inputs['Alpha'].is_linked:
        return
    alpha=shader.inputs['Alpha']
    # Keep pore contrast and tint when replacing the hashed coverage pass.
    # Calibrate the transmittance curve against the locked side-lit material.
    transmittance=tree.nodes.new('ShaderNodeMath')
    transmittance.operation='SUBTRACT'
    transmittance.inputs[0].default_value=1.
    transmittance.use_clamp=True
    tree.links.new(alpha.links[0].from_socket,transmittance.inputs[1])
    depth=tree.nodes.new('ShaderNodeMath')
    depth.label='Blended yarn optical depth'
    depth.operation='POWER'
    depth.inputs[1].default_value=2.8
    tree.links.new(transmittance.outputs[0],depth.inputs[0])
    coverage=tree.nodes.new('ShaderNodeMath')
    coverage.operation='SUBTRACT'
    coverage.inputs[0].default_value=1.
    tree.links.new(depth.outputs[0],coverage.inputs[1])
    tree.links.new(coverage.outputs[0],alpha)
    material.surface_render_method='BLENDED'
    material.use_transparency_overlap=False


def apply_template(template, image, first, palette, thickness, *, smooth_alpha=True):
    material = template.copy()
    material.name = 'Weft • approved open-pore finish'
    tree = material.node_tree
    lookups = [node for node in tree.nodes if node.type == 'TEX_IMAGE' and node.label == GAUGE]
    if len(lookups) != 1:
        raise ValueError('定稿袜材质应包含一个线圈量表入口')
    # The supplied template is the dry material. The sole baked animation
    # is the sequence address on the gauge image being replaced here.
    tree.animation_data_clear()
    knit = tree.nodes.new('ShaderNodeUVMap')
    knit.uv_map = 'Knit'
    set_gauge(tree, lookups[0], knit.outputs['UV'], image, first)
    settings = tree.nodes['Weft Dry Appearance'].inputs
    settings['Palette'].default_value = palette
    settings['Thickness / Opacity'].default_value = thickness
    if smooth_alpha and not apply_sheer_finish(material):
        smooth_preview_alpha(material)
    material[TEMPLATE] = True
    return material
