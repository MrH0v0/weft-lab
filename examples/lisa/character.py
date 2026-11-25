# SPDX-License-Identifier: MIT
"""Import the official Lisa PMX as editable Blender surfaces and an armature."""

from collections import defaultdict
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from third_party import pmx_reader


def to_world(pmx):
    """Switch handedness and keep the PMX origin and unit length."""
    return np.asarray(pmx)[..., [0, 2, 1]] * (-1, -1, 1)


def insert_attribute(mesh, name, domain, data_type, values):
    attribute = mesh.attributes.new(name, data_type, domain)
    attribute.data.foreach_set("value", values)


def source_weights(vertex):
    source = vertex.weight
    if source.type == 0:
        factors = [1.0]
    elif source.type == 1:
        first = source.weights[0]
        factors = [first, 1.0 - first]
    elif source.type == 2:
        factors = source.weights
    else:
        raise ValueError(f"Unsupported PMX weighting mode {source.type}")
    weights = defaultdict(float)
    for bone, value in zip(source.bones, factors, strict=True):
        if bone >= 0 and value > 0:
            weights[bone] += value
    total = sum(weights.values())
    assert total > 0
    return {bone: value / total for bone, value in weights.items()}


def pack_textures(model):
    images = []
    for texture in model.textures:
        path = Path(texture.path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}. Keep the original Texture folder next to the PMX.")
        image = bpy.data.images.load(str(path), check_existing=True)
        image.pack()
        image.filepath = "//Texture/" + path.name
        images.append(image)
    return images


def outfit_finish(material, shader, name, painted_color):
    tree = material.node_tree
    link = tree.links.new
    if name in {"皮肤", "新脸"}:
        shader.inputs["Roughness"].default_value = .51
        shader.inputs["Specular IOR Level"].default_value = .31
        shader.inputs["Subsurface Weight"].default_value = .065
        shader.inputs["Subsurface Scale"].default_value = .005
    elif name == "头发":
        direction = tree.nodes.new("ShaderNodeTangent")
        direction.direction_type, direction.uv_map = "UV_MAP", "PMX UV"
        link(direction.outputs[0], shader.inputs["Tangent"])
        shader.inputs["Anisotropic"].default_value = .46
        shader.inputs["Roughness"].default_value = .44
        shader.inputs["Specular IOR Level"].default_value = .34
    elif name in {"眼睛", "眼白"}:
        shader.inputs["Roughness"].default_value = .27
        shader.inputs["Specular IOR Level"].default_value = .38
    elif name in {"衣服", "帽子", "手饰", "肩饰", "腰带", "鞋子", "裙子"}:
        shader.inputs["Sheen Weight"].default_value = .14
        shader.inputs["Sheen Roughness"].default_value = .55
        shader.inputs["Specular IOR Level"].default_value = .32
        # The original painted red/blue difference isolates gold from the
        # purple cloth and white lining, even when they share a material slot.
        color = tree.nodes.new("ShaderNodeSeparateColor")
        link(painted_color, color.inputs[0])
        gold = tree.nodes.new("ShaderNodeMath")
        gold.operation = "SUBTRACT"
        link(color.outputs["Red"],gold.inputs[0])
        link(color.outputs["Blue"],gold.inputs[1])
        mask = tree.nodes.new("ShaderNodeMapRange")
        mask.interpolation_type = "SMOOTHERSTEP"
        mask.clamp = True
        mask.inputs["From Min"].default_value = .045
        mask.inputs["From Max"].default_value = .32
        mask.inputs["To Max"].default_value = .73
        link(gold.outputs[0],mask.inputs[0])
        link(mask.outputs[0],shader.inputs["Metallic"])
        rough = tree.nodes.new("ShaderNodeMath")
        rough.operation = "MULTIPLY_ADD"
        link(mask.outputs[0],rough.inputs[0])
        rough.inputs[1].default_value = -.27
        rough.inputs[2].default_value = .48 if name == "鞋子" else .57
        link(rough.outputs[0],shader.inputs["Roughness"])


def source_materials(model, images):
    materials = []
    for index, original in enumerate(model.materials):
        mat = bpy.data.materials.new("Lisa | " + original.name)
        mat["pmx_material_index"] = index
        mat.diffuse_color = original.diffuse
        mat.use_backface_culling = not original.is_double_sided
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = original.diffuse
        shader.inputs["Roughness"].default_value = 0.84
        shader.inputs["Specular IOR Level"].default_value = 0.18
        shader.inputs["Alpha"].default_value = original.diffuse[3]
        if original.texture >= 0:
            image = mat.node_tree.nodes.new("ShaderNodeTexImage")
            image.image = images[original.texture]
            image.label = image.image.name
            image.location = (-530, 220)
            tint = mat.node_tree.nodes.new("ShaderNodeMixRGB")
            tint.blend_type = "MULTIPLY"
            tint.inputs[0].default_value = 1.0
            tint.inputs[2].default_value = original.diffuse
            tint.location = (-270, 230)
            mat.node_tree.links.new(image.outputs["Color"], tint.inputs[1])
            mat.node_tree.links.new(tint.outputs["Color"], shader.inputs["Base Color"])
            outfit_finish(mat, shader, original.name, tint.outputs["Color"])
            if original.name == "表情":
                mat.surface_render_method = "DITHERED"
                mat.node_tree.links.new(image.outputs["Alpha"], shader.inputs["Alpha"])
            if original.name == "新脸":
                # Keep the original atlas dominant to soften the hard stripe
                # across the nose under physically based lighting.
                emission = mat.node_tree.nodes.new("ShaderNodeEmission")
                emission.label = "Original painted face color"
                emission.location = (0, -130)
                emission.inputs["Strength"].default_value = 0.82
                mix = mat.node_tree.nodes.new("ShaderNodeMixShader")
                mix.label = "Face lighting"
                mix.location = (340, 200)
                mix.inputs[0].default_value = 0.55
                output = mat.node_tree.nodes.get("Material Output")
                output.location = (560, 200)
                mat.node_tree.links.new(tint.outputs["Color"], emission.inputs["Color"])
                mat.node_tree.links.new(shader.outputs[0], mix.inputs[1])
                mat.node_tree.links.new(emission.outputs[0], mix.inputs[2])
                mat.node_tree.links.new(mix.outputs[0], output.inputs["Surface"])
        materials.append(mat)

    skin = bpy.data.materials.new("Lisa | Right Leg Test Skin")
    skin.diffuse_color = (0.55, 0.34, 0.27, 1.0)
    skin.use_nodes = True
    shader = skin.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = skin.diffuse_color
    shader.inputs["Roughness"].default_value = 0.54
    shader.inputs["Specular IOR Level"].default_value = 0.30
    shader.inputs["Subsurface Weight"].default_value = .075
    shader.inputs["Subsurface Scale"].default_value = .005
    variation = skin.node_tree.nodes.new("ShaderNodeTexNoise")
    variation.inputs["Scale"].default_value = 4.0
    variation.inputs["Detail"].default_value = 2.0
    tint = skin.node_tree.nodes.new("ShaderNodeValToRGB")
    tint.color_ramp.elements[0].color = (.50, .298, .235, 1)
    tint.color_ramp.elements[1].color = (.60, .375, .303, 1)
    skin.node_tree.links.new(variation.outputs["Fac"], tint.inputs[0])
    skin.node_tree.links.new(tint.outputs[0], shader.inputs["Base Color"])
    pore = skin.node_tree.nodes.new("ShaderNodeTexNoise")
    pore.inputs["Scale"].default_value = 360.
    pore.inputs["Detail"].default_value = 1.6
    bump = skin.node_tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .11
    bump.inputs["Distance"].default_value = .000014
    skin.node_tree.links.new(pore.outputs["Fac"],bump.inputs["Height"])
    skin.node_tree.links.new(bump.outputs[0],shader.inputs["Normal"])
    return materials, skin


def add_rig(model, parent):
    data = bpy.data.armatures.new("Lisa 1.4 PMX Skeleton")
    rig = bpy.data.objects.new("Lisa Armature", data)
    parent.objects.link(rig)
    rig["pmx_coordinates"] = "(-X, -Z, Y), in PMX units before object scale"
    data.display_type = "STICK"
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    heads = to_world([bone.location for bone in model.bones])
    offsets = []
    for bone, head in zip(model.bones, heads, strict=True):
        connection = bone.displayConnection
        if isinstance(connection, int):
            offset = heads[connection] - head if connection >= 0 else np.zeros(3)
        else:
            offset = to_world(connection)
        offsets.append(offset)
    leg_grants = (68, 69, 70, 72, 73, 74)
    for index in leg_grants:
        offsets[index] = offsets[model.bones[index].additionalTransform[0]]

    weighted_bones = {bone for vertex in model.vertices for bone in source_weights(vertex)}
    for index, (bone, head, offset) in enumerate(zip(model.bones, heads, offsets, strict=True)):
        edit = data.edit_bones.new(bone.name)
        assert edit.name == bone.name
        edit.head = head
        if np.linalg.norm(offset) < 1.0e-5:
            offset = (0, 0, 0.12)
        edit.tail = head + offset
        edit.use_deform = index in weighted_bones
    for bone in model.bones:
        if bone.parent >= 0:
            data.edit_bones[bone.name].parent = data.edit_bones[model.bones[bone.parent].name]
    bpy.ops.object.mode_set(mode="OBJECT")
    for index, bone in enumerate(model.bones):
        data.bones[bone.name]["pmx_bone_index"] = index
    for index in leg_grants:
        bone = model.bones[index]
        target, factor = bone.additionalTransform
        constraint = rig.pose.bones[bone.name].constraints.new("COPY_ROTATION")
        constraint.name = "PMX leg rotation inheritance"
        constraint.target = rig
        constraint.subtarget = model.bones[target].name
        constraint.target_space = "LOCAL"
        constraint.owner_space = "LOCAL"
        constraint.mix_mode = "ADD"
        constraint.influence = factor
    rig.select_set(False)
    return rig


def build_surface(name, part, rig, positions, triangles, weights, ids, materials,
                  face_materials, normals, uv_corners):
    data = bpy.data.meshes.new(name)
    data.from_pydata(to_world(positions).tolist(), [], triangles.tolist(), shade_flat=False)
    data.update()
    assert not data.validate(verbose=True), f"Unexpected invalid geometry in {name}"
    for material in materials:
        data.materials.append(material)
    data.polygons.foreach_set("material_index", face_materials)
    uv = data.uv_layers.new(name="PMX UV")
    uv.uv.foreach_set("vector", np.asarray(uv_corners, dtype=np.float32).ravel())
    data.normals_split_custom_set_from_vertices(to_world(normals).tolist())
    insert_attribute(data, "pmx_vertex_index", "POINT", "INT", ids)

    obj = bpy.data.objects.new(name, data)
    part.objects.link(obj)
    obj.parent = rig
    groups = {}
    bone_names = {bone["pmx_bone_index"]: bone.name for bone in rig.data.bones}
    for index, point_weights in enumerate(weights):
        for bone, value in point_weights.items():
            if bone not in groups:
                groups[bone] = obj.vertex_groups.new(name=bone_names[bone])
            groups[bone].add([index], value, "REPLACE")
    modifier = obj.modifiers.new("Lisa PMX Weights", "ARMATURE")
    modifier.object = rig
    return obj


def original_mesh(name, part, rig, faces, groups, positions, normals, uv, weights, materials):
    ids, local_faces = np.unique(faces, return_inverse=True)
    used_materials, local_groups = np.unique(groups, return_inverse=True)
    return build_surface(
        name, part, rig, positions[ids], local_faces.reshape(faces.shape),
        [weights[i] for i in ids], ids, [materials[i] for i in used_materials],
        local_groups, normals[ids], uv[faces],
    ), ids


def vertex_morphs(obj, source_ids, model):
    cache = {}

    def offsets(index):
        if index in cache:
            return cache[index]
        morph = model.morphs[index]
        result = defaultdict(lambda: np.zeros(3))
        if isinstance(morph, pmx_reader.VertexMorph):
            for item in morph.offsets:
                result[item.index] += to_world(item.offset)
        elif isinstance(morph, pmx_reader.GroupMorph):
            for item in morph.offsets:
                for vertex, delta in offsets(item.morph).items():
                    result[vertex] += item.factor * delta
        cache[index] = result
        return result

    lookup = {source: idx for idx, source in enumerate(source_ids)}
    for i, morph in enumerate(model.morphs):
        relevant = {lookup[k]: delta for k, delta in offsets(i).items() if k in lookup}
        if not relevant:
            continue
        if obj.data.shape_keys is None:
            obj.shape_key_add(name="Basis", from_mix=False)
        key = obj.shape_key_add(name=morph.name, from_mix=False)
        key.value = 0.0
        for local, delta in relevant.items():
            key.data[local].co += Vector(delta)
