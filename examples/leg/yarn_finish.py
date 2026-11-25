# SPDX-License-Identifier: MIT
"""Bake periodic fibre normals and an unoriented tangent field for the demo yarn."""
import math

import bpy
import numpy as np


def _packed_map(name, channels):
    length = channels.shape[0]
    image = bpy.data.images.new(name, length, length, alpha=True, is_data=True)
    image.alpha_mode = "CHANNEL_PACKED"
    image.pixels.foreach_set(np.ascontiguousarray(channels, dtype=np.float32).ravel())
    image.update()
    image.pack()
    return image


def bake_maps(path, height, radii, size=1024):
    """Resolve the upper yarn at crossings and lay the outer filaments along its arc."""
    crown = np.full((size, size), -np.inf, np.float32)
    relief = np.zeros_like(crown)
    flow = np.zeros((size, size, 2), np.float32)
    flow[:, :, 0] = 1
    coverage = np.zeros_like(crown)
    roughness = np.full_like(crown, 0.53)
    path = np.asarray(path, dtype=float)
    radius = np.asarray(radii, dtype=float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    for i, (a, b) in enumerate(zip(path, path[1:])):
        delta = (b-a) / radius
        square = np.dot(delta, delta)
        if square < 1e-12:
            continue
        theta = math.atan2(b[1]-a[1], b[0]-a[0])
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                offset = np.array([tx, ty])
                lo = np.maximum(0, np.floor((np.minimum(a,b)+offset-radius)*size).astype(int))
                hi = np.minimum(size, np.ceil((np.maximum(a,b)+offset+radius)*size).astype(int))
                if np.any(hi<=lo):
                    continue
                sl = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
                yy,xx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
                dx = ((xx+.5)/size-a[0]-tx)/radius[0]
                dy = ((yy+.5)/size-a[1]-ty)/radius[1]
                t = np.clip((dx*delta[0]+dy*delta[1])/square, 0, 1)
                ox,oy = dx-t*delta[0], dy-t*delta[1]
                d = np.hypot(ox,oy)
                signed = np.copysign(d, delta[0]*oy-delta[1]*ox)
                mask = np.clip((1-d)*min(radius)*size+.5,0,1)
                cap = np.sqrt(np.maximum(1-d*d,0))
                z = height[i]*(1-t)+height[i+1]*t + 0.000064*cap
                upper = (mask>0) & (z > crown[sl])
                if not upper.any():
                    continue
                along = arc[i]+t*(arc[i+1]-arc[i])
                phase = 2*np.pi*(3.7*signed+.65*along)
                filament = .5+.5*np.cos(phase)
                crown[sl][upper] = z[upper]
                coverage[sl][upper] = mask[upper]
                relief[sl][upper] = (mask*cap*(.000030+.0000042*filament))[upper]
                twist = theta+.105*np.sin(phase)
                flow[sl][upper,0] = np.cos(2*twist[upper])
                flow[sl][upper,1] = np.sin(2*twist[upper])
                roughness[sl][upper] = (.34+.13*(1-cap)+.055*(1-filament))[upper]
    # Metric slopes keep the normal strength independent of map resolution.
    nx = -(np.roll(relief,-1,1)-np.roll(relief,1,1))*size/(2*.0023)
    ny = -(np.roll(relief,-1,0)-np.roll(relief,1,0))*size/(2*.0017)
    normal = np.stack((nx,ny,np.ones_like(nx)),axis=-1)
    normal /= np.linalg.norm(normal,axis=-1,keepdims=True)
    tangent_map = np.dstack((.5+.5*flow,roughness,coverage))
    normal_map = np.dstack((.5+.5*normal,np.ones_like(nx)))
    return (_packed_map("Weft - fibre flow roughness coverage",tangent_map),
            _packed_map("Weft - tangent fibre normal",normal_map))


def add_yarn_reflection(material, maps):
    """Place a dielectric anisotropic lobe only on the yarn, in the live knit chart."""
    tree = material.node_tree
    nodes, link = tree.nodes, tree.links.new
    atlas = next(n for n in nodes if n.label == "Open pore coverage and approximate top relief")
    address = atlas.inputs["Vector"].links[0].from_socket
    mapping = nodes.new("ShaderNodeVectorMath")
    mapping.operation = "SCALE"
    mapping.name = "Weft knit size • " + atlas.name
    mapping.inputs["Scale"].default_value = 1
    link(address,mapping.inputs[0])
    vector = mapping.outputs[0]
    link(vector,atlas.inputs["Vector"])
    started = set(nodes)
    def image_node(image):
        n = nodes.new("ShaderNodeTexImage")
        n.image, n.extension, n.interpolation = image, "REPEAT", "Linear"
        link(vector,n.inputs["Vector"])
        return n

    def calc(op, *values):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for target, val in zip(n.inputs, values):
            if isinstance(val, (int,float)):
                target.default_value = val
            else:
                link(val,target)
        return n.outputs[0]

    flow, normal = map(image_node,maps)
    flow.label, normal.label = "Fibre axis, roughness, coverage", "Outer filament relief"
    split = nodes.new("ShaderNodeSeparateColor")
    link(flow.outputs["Color"],split.inputs[0])
    angle = calc("ARCTAN2",calc("MULTIPLY_ADD",split.outputs[1],2,-1),
                 calc("MULTIPLY_ADD",split.outputs[0],2,-1))
    turns = calc("MULTIPLY",angle,1/(4*math.pi))
    tangent = nodes.new("ShaderNodeTangent")
    tangent.direction_type, tangent.uv_map = "UV_MAP", "Knit"
    nm = nodes.new("ShaderNodeNormalMap")
    nm.uv_map = "Knit"
    nm.inputs["Strength"].default_value = .52
    link(normal.outputs["Color"],nm.inputs["Color"])
    gloss = nodes.new("ShaderNodeBsdfAnisotropic")
    gloss.label = "Nylon filaments"
    gloss.inputs["Color"].default_value = (.94,.94,.99,1)
    gloss.inputs["Anisotropy"].default_value = .76
    link(split.outputs[2],gloss.inputs["Roughness"])
    link(tangent.outputs[0],gloss.inputs["Tangent"])
    link(turns,gloss.inputs["Rotation"])
    link(nm.outputs[0],gloss.inputs["Normal"])
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.53
    link(nm.outputs[0],fresnel.inputs["Normal"])
    reflected = calc("MULTIPLY",flow.outputs["Alpha"],calc("MULTIPLY_ADD",fresnel.outputs[0],.86,.038))
    out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
    base = out.inputs["Surface"].links[0].from_socket
    mix = nodes.new("ShaderNodeMixShader")
    mix.label = "Fibre reflection in the knit"
    link(reflected,mix.inputs[0])
    link(base,mix.inputs[1])
    link(gloss.outputs[0],mix.inputs[2])
    link(mix.outputs[0],out.inputs["Surface"])
    frame = nodes.new("NodeFrame")
    frame.label = "Nylon • baked filament reflection"
    frame.location = (out.location.x - 860, out.location.y - 620)
    helpers = [n for n in nodes if n not in started and n != frame]
    for i, n in enumerate(helpers):
        n.parent = frame
        n.location = ((i//3)*200, -(i%3)*230)
    mapping.location = (atlas.location.x-220, atlas.location.y)
    finish_leg_yarn(material)


def finish_leg_yarn(material):
    """Apply the photographed sample finish to a material with baked yarn maps."""
    tree = material.node_tree
    nodes, link = tree.nodes, tree.links.new
    glossy = next(n for n in nodes if n.label == "Nylon filaments")
    reflection = next(n for n in nodes if n.label == "Fibre reflection in the knit")
    factor = reflection.inputs[0].links[0].from_socket
    budget = nodes.new("ShaderNodeMath")
    budget.operation = "MULTIPLY"
    budget.label = "Filament reflection budget"
    budget.inputs[1].default_value = .235
    link(factor, budget.inputs[0])
    link(budget.outputs[0], reflection.inputs[0])
    glossy.inputs["Color"].default_value = (.72, .66, .62, 1.)
    glossy.inputs["Anisotropy"].default_value = .54
    glossy.inputs["Normal"].links[0].from_node.inputs["Strength"].default_value = .45
    source = glossy.inputs["Roughness"].links[0].from_socket
    roughness = nodes.new("ShaderNodeMath")
    roughness.operation = "MULTIPLY_ADD"
    roughness.inputs[1].default_value = .75
    roughness.inputs[2].default_value = .21
    link(source, roughness.inputs[0])
    link(roughness.outputs[0], glossy.inputs["Roughness"])

    # Use the same mipmapped coverage field as the live gauge shader. Pores
    # open where the simulated head and neck diverge under tension.
    profile = next(n for n in nodes if n.type == "TEX_IMAGE" and n.image
                   and n.image.name.startswith("live-loop-profile"))
    channels = nodes.new("ShaderNodeSeparateColor")
    link(profile.outputs["Color"], channels.inputs[0])
    pores = nodes.new("ShaderNodeMapRange")
    pores.label = "Transmission between yarns"
    pores.interpolation_type = "SMOOTHERSTEP"
    pores.inputs["From Min"].default_value = .07
    pores.inputs["From Max"].default_value = .83
    pores.inputs["To Min"].default_value = .64
    link(channels.outputs["Blue"], pores.inputs["Value"])
    surface = next(n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
    mix = nodes.new("ShaderNodeMixShader")
    link(pores.outputs[0], mix.inputs[0])
    link(nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    link(reflection.outputs[0], mix.inputs[2])
    link(mix.outputs[0], surface.inputs["Surface"])
