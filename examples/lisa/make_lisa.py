# SPDX-License-Identifier: MIT
"""Create a playable local Weft Lab scene from the official Lisa 1.4 PMX."""

import argparse
import hashlib
import importlib
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import character
from third_party import pmx_reader


LISA_14_HASH = "0b8db0b0c73ab23357f7c5d4e3cc2507eb8a1d32ca871b0760d25ba68ddb6632"
METRES_PER_UNIT = 0.085


def import_character(model, scene):
    parts = scene.collection
    rig = character.add_rig(model, parts)
    rig.scale = (METRES_PER_UNIT,) * 3
    points = np.asarray([v.co for v in model.vertices])
    normals = np.asarray([v.normal for v in model.vertices])
    uv = np.asarray([v.uv for v in model.vertices]) * (1, -1) + (0, 1)
    weights = [character.source_weights(v) for v in model.vertices]
    materials, skin_material = character.source_materials(model, character.pack_textures(model))
    faces = np.asarray(model.faces, dtype=np.int32)
    slots = np.repeat(np.arange(len(materials)), [m.vertex_count // 3 for m in model.materials])
    tights = next(i for i, m in enumerate(model.materials) if m.name == "丝袜")
    right_leg = (slots == tights) & (points[faces, 0] < 0).all(1) & (points[faces, 1] <= 10.0).all(1)

    def mesh(name, selection):
        return character.original_mesh(
            name, parts, rig, faces[selection], slots[selection], points, normals, uv, weights, materials
        )

    body, vertex_ids = mesh("Lisa · 原装服饰", ~right_leg)
    character.vertex_morphs(body, vertex_ids, model)
    skin, _ = mesh("Lisa · 右腿皮肤", right_leg)
    skin.data.materials.clear()
    skin.data.materials.append(skin_material)
    for polygon in skin.data.polygons:
        polygon.material_index = 0
    detail = skin.modifiers.new("袜面采样密度", "SUBSURF")
    detail.subdivision_type = "SIMPLE"
    detail.levels = detail.render_levels = 1
    return rig, skin


def fit_stocking(scene, skin, apply_length):
    template = skin.copy()
    template.name = "Lisa · 原袜"
    scene.collection.objects.link(template)
    template.material_slots[0].link = "OBJECT"
    mat = bpy.data.materials.new("Lisa · 原袜底色")
    mat.diffuse_color = (0.065, 0.046, 0.052, 1)
    template.material_slots[0].material = mat
    template["weft_contact_skin"] = skin
    template["weft_skin_slot"] = 0
    library = Path(__file__).with_name("weft-yarn.blend")
    with bpy.data.libraries.load(str(library), link=False) as (_, loaded):
        loaded.materials = ["Weft Lab • Dry yarn"]
    template["weft_live_finish"] = loaded.materials[0]
    template["weft_dry_palette"] = 0
    template["weft_dry_thickness"] = 0.53
    template["weft_visual_knit_scale"] = 1.12
    scene.weft_sock = template
    scene.weft_skin = skin
    scene.weft_side = "R"
    bpy.context.view_layer.update()
    sock = apply_length(bpy.context, template, skin, 0, 0.78)
    sock.name = "Lisa · 可抓取右袜"
    return sock


def knee_motion(scene, rig):
    knee = rig.pose.bones["右ひざ"]
    knee.rotation_mode = "XYZ"
    for frame, angle in ((1, 0.0), (25, 0.18), (45, 0.38), (65, 0.18), (85, 0.0)):
        knee.rotation_euler.x = angle
        knee.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
    rig.animation_data.action.name = "Weft Lab · 屈膝"
    scene.frame_start, scene.frame_end = 1, 85
    scene.render.fps = 30
    scene.frame_set(1)


def rest_pose(rig):
    def aim(joint, child, direction):
        bone = rig.pose.bones[joint]
        current = rig.pose.bones[child].head - bone.head
        swing = current.rotation_difference(Vector(direction).normalized())
        axes = bone.matrix.to_quaternion()
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = bone.rotation_quaternion @ axes.inverted() @ swing @ axes
        bpy.context.view_layer.update()

    aim("左腕", "左ひじ", (-0.30, -0.05, -0.95))
    aim("左ひじ", "左手首", (-0.13, 0.30, -0.94))
    aim("右腕", "右ひじ", (0.77, 0.13, -0.62))
    aim("右ひじ", "右手首", (-0.68, 0.53, -0.42))
    head = rig.pose.bones["頭"]
    basis = head.matrix.to_quaternion()
    head.rotation_mode = "QUATERNION"
    turn = Quaternion((0, 0, 1), -0.15) @ Quaternion((0, 1, 0), -0.055)
    head.rotation_quaternion = basis.inverted() @ turn @ basis
    bpy.context.view_layer.update()


def studio(scene):
    # A curved sweep receives the shoe contact shadows without a horizon line.
    from math import cos, pi, sin
    profile = [(4.0, -0.023), (-0.76, -0.023)]
    for i in range(1, 25):
        angle = pi * i / 48
        profile.append((-0.76 - 0.85 * sin(angle), -0.023 + 0.85 * (1 - cos(angle))))
    profile.append((-1.61, 4.5))
    points = [(x, y, z) for y, z in profile for x in (-6.0, 6.0)]
    faces = [(i + 1, i, i + 2, i + 3) for i in range(0, len(points) - 2, 2)]
    mesh = bpy.data.meshes.new("弧形背景")
    mesh.from_pydata(points, [], faces)
    for face in mesh.polygons:
        face.use_smooth = True
    backdrop = bpy.data.objects.new(mesh.name, mesh)
    scene.collection.objects.link(backdrop)
    backdrop.hide_select = True
    material = bpy.data.materials.new("摄影棚 · 雾紫")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.039, 0.027, 0.070, 1)
    shader.inputs["Roughness"].default_value = 0.9
    mesh.materials.append(material)
    world = bpy.data.worlds.new("摄影棚 · 环境光")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.31, 0.35, 0.47, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.16
    scene.world = world
    for name, location, target, watts, size, tint in (
        ("柔光箱", (1.22, 0.43, 2.18), (0, 0, 0.92), 145, 1.24, (1.0, 0.82, 0.67)),
        ("补光", (-1.05, 1.23, 1.67), (0, 0, 0.96), 48, 1.60, (0.75, 0.83, 1.0)),
        ("轮廓光", (-0.66, -0.84, 1.69), (0, 0, 0.91), 95, 0.67, (0.69, 0.68, 1.0)),
        ("背景光", (0.82, -0.42, 1.12), (0.47, -1.59, 0.95), 44, 0.95, (0.56, 0.46, 0.83)),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.size, light.color = watts, "DISK", size, tint
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    for name, location, tint, watts, width in (
        ("冷侧条光", (-0.46,0.38,0.83), (0.73,0.85,1.0), 13, 0.18),
        ("暖侧条光", (0.51,-0.12,0.87), (1.0,0.78,0.59), 16, 0.14),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.color = watts, "RECTANGLE", tint
        light.size, light.size_y = width, 0.92
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0.055,0.015,0.53))-obj.location).to_track_quat("-Z","Y").to_euler()
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 64
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


def setup_view(scene, rig):
    knee = rig.matrix_world @ rig.data.bones["右ひざ"].head_local
    hip = rig.matrix_world @ rig.data.bones["右足"].head_local
    target = knee.lerp(hip, 0.24)
    eye = Vector((0.65, 1.0, 0.12)).normalized()
    rotation = (-eye).to_track_quat("-Z", "Y")
    camera = bpy.data.cameras.new("丽莎 · 右腿")
    camera.type = "ORTHO"
    camera.ortho_scale = 0.96
    camera.clip_start = 0.005
    scene.camera = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(scene.camera)
    scene.camera.location = target + eye * 1.6
    scene.camera.rotation_euler = rotation.to_euler()
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.region_3d.view_perspective = "ORTHO"
                space.region_3d.view_rotation = rotation
                space.region_3d.view_location = target
                space.region_3d.view_distance = 0.97
                space.clip_start = 0.005
                space.overlay.show_overlays = False
                space.shading.type = "MATERIAL"
                space.shading.use_scene_lights = True
                space.shading.use_scene_world = True
                space.show_region_ui = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmx", type=Path, help="丽莎.pmx from the original 1.4 distribution")
    parser.add_argument("output", type=Path, help="A new .blend file for local use")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    pmx = args.pmx.resolve(strict=True)
    output = args.output.resolve()
    if not bpy.app.background or bpy.data.filepath:
        raise RuntimeError("Run in a fresh background Blender process, as shown in README.md.")
    if output.suffix.lower() != ".blend" or output.exists():
        raise ValueError("Choose a new .blend output path. No existing file will be overwritten.")
    if hashlib.sha256(pmx.read_bytes()).hexdigest() != LISA_14_HASH:
        raise ValueError("This recipe expects the official Lisa 1.4 PMX. See the model link in README.md.")
    addon = next((a.module for a in bpy.context.preferences.addons if a.module.split(".")[-1] == "weft_replay"), None)
    if addon is None:
        raise RuntimeError("Install and enable Weft Lab in Blender preferences first.")
    apply_length = importlib.import_module(addon + ".sock_length").apply_length
    model = pmx_reader.load(str(pmx))
    scene = bpy.data.scenes.new("Weft Lab · 丽莎")
    bpy.context.window.scene = scene
    for previous in list(bpy.data.scenes):
        if previous != scene:
            bpy.data.scenes.remove(previous)
    bpy.data.orphans_purge(do_recursive=True)
    scene.unit_settings.system = "METRIC"
    rig, skin = import_character(model, scene)
    fit_stocking(scene, skin, apply_length)
    knee_motion(scene, rig)
    rest_pose(rig)
    studio(scene)
    setup_view(scene, rig)
    text = bpy.data.texts.new("Weft Lab · 演示说明")
    text.use_fake_user = True
    text.write("N → Weft Lab → 干态抓取，或湿态抓取 / 泼水。左键拖动袜面，湿态按 W 加水。\n"
               "《原神》丽莎 © miHoYo，模型编辑：神帝宇。角色来自本机的官方 1.4 PMX。\n"
               "生成的工程包含原模型，使用和分享请遵守原包《使用规则.txt》。\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.use_preferences_save = False
    saved = bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
    if saved != {"FINISHED"} or not output.is_file():
        raise RuntimeError(f"Blender could not save {output}")
    print(f"Created {output}", flush=True)


if __name__ == "__main__":
    main()
