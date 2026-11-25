# SPDX-License-Identifier: MIT
"""Build the Weft Lab dry/wet leg study in a clean, add-on-enabled Blender."""
import argparse
import importlib
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

# Hand-authored transverse sections in metres, progressing from toes to thigh.
# Columns are centre height, forward position, lateral offset and the two radii.
SECTIONS = np.array([
    (.027, .212, -.009, .010, .005), (.029, .202, -.008, .031, .012),
    (.032, .175, -.006, .045, .021), (.037, .136, -.001, .047, .027),
    (.044, .096, .001, .040, .031), (.055, .055, .003, .035, .034),
    (.068, .021, .004, .034, .041), (.084, .000, .005, .035, .043),
    (.108,-.004, .005, .035, .038), (.145,-.005, .005, .035, .036),
    (.195,-.010, .004, .041, .045), (.255,-.017, .003, .050, .059),
    (.317,-.022, .000, .056, .070), (.362,-.017,-.002, .056, .067),
    (.406,-.008,-.003, .050, .059), (.453, .004,-.004, .046, .050),
    (.490, .012,-.005, .051, .052), (.514, .014,-.005, .053, .055),
    (.547, .011,-.006, .052, .059), (.605, .005,-.009, .059, .068),
    (.676,-.004,-.011, .069, .079), (.756,-.014,-.013, .078, .087),
    (.835,-.020,-.014, .084, .094), (.911,-.020,-.013, .088, .096),
    (.946,-.018,-.011, .084, .092),
], dtype=float)
RIGHT_X = .078


def leg_surface():
    """Loft a smooth, closed fitting last around its curved ankle centreline."""
    centres = SECTIONS[:, [2, 1, 0]]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centres, axis=0), axis=1))]
    samples = np.linspace(0, distance[-1], 137)
    span = np.minimum(np.searchsorted(distance, samples, side='right')-1, len(distance)-2)
    step = distance[span+1]-distance[span]
    t = ((samples-distance[span])/step)[:, None]
    slope = np.gradient(SECTIONS, distance, axis=0)
    rows = ((2*t**3-3*t*t+1)*SECTIONS[span] + (t**3-2*t*t+t)*step[:,None]*slope[span]
            + (-2*t**3+3*t*t)*SECTIONS[span+1] + (t**3-t*t)*step[:,None]*slope[span+1])
    z, forward, side, width, depth = rows.T
    dy, dz = np.gradient(forward), np.gradient(z)
    norm = np.hypot(dy,dz)
    n = 80
    angle = 2*np.pi*np.arange(n)/n
    cs, sn = np.cos(angle), np.sin(angle)

    def landmark(height, vertical, azimuth, spread):
        return np.exp(-((z[:,None]-height)/vertical)**2
                      +(np.cos(angle-azimuth)-1)/spread**2)

    # The kneecap faces forward. The two calf heads sit at different heights.
    ridge = (.0055*landmark(.506,.034,np.pi/2,.43)
             +.0050*landmark(.312,.066,3.82,.55)
             +.0042*landmark(.349,.066,5.50,.59)
             +.0030*landmark(.263,.115,1.27,.22)
             +.0044*landmark(.097,.021,3.15,.35)
             +.0038*landmark(.087,.018,.03,.36))
    pts = np.stack((RIGHT_X+side[:,None]+(width[:,None]+ridge)*cs,
                    forward[:,None]+(depth[:,None]+ridge)*sn*(dz/norm)[:,None],
                    z[:,None]-(depth[:,None]+ridge)*sn*(dy/norm)[:,None]),axis=-1)
    rings = len(samples)
    vertices = pts.reshape(-1,3).tolist()
    toe, thigh = len(vertices), len(vertices)+1
    vertices += [pts[0].mean(0).tolist(),pts[-1].mean(0).tolist()]
    faces = []
    for i in range(rings-1):
        for j in range(n):
            a,b = i*n+j,i*n+(j+1)%n
            faces.append((a,b,b+n,a+n))
    for j in range(n):
        faces.append((toe,(j+1)%n,j))
        faces.append((thigh,(rings-1)*n+j,(rings-1)*n+(j+1)%n))
    mesh = bpy.data.meshes.new("Leg form • original section loft")
    mesh.from_pydata(vertices,[],faces,shade_flat=False)
    mesh.update()
    return mesh


def make_rig(scene):
    data=bpy.data.armatures.new("Weft • fitting rig")
    rig=bpy.data.objects.new(data.name,data)
    scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active=rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    # The binding tool locates both leg chains, even on a one-leg fitting form.
    for side,x in (('R',RIGHT_X),('L',-RIGHT_X)):
        chain=(('thigh',(.0,-.012,.907),(-.005,.014,.509)),
               ('shin',(-.005,.014,.509),(.005,-.004,.092)),
               ('foot',(.005,-.004,.092),(-.002,.173,.033)))
        parent=None
        for label,start,end in chain:
            b=data.edit_bones.new(label+'.'+side)
            b.head=(x+start[0],start[1],start[2])
            b.tail=(x+end[0],end[1],end[2])
            if parent:
                b.parent=parent
                b.use_connect=True
            b.use_deform=side=='R'
            parent=b
    bpy.ops.object.mode_set(mode='OBJECT')
    rig.select_set(False)
    return rig


def skin_finish():
    mat=bpy.data.materials.new("Weft • warm skin")
    mat.use_nodes=True
    nodes,link=mat.node_tree.nodes,mat.node_tree.links.new
    shader=nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value=.53
    shader.inputs["Specular IOR Level"].default_value=.24
    shader.inputs["Subsurface Weight"].default_value=.13
    shader.inputs["Subsurface Scale"].default_value=.00115
    uv=nodes.new("ShaderNodeTexCoord")
    cloud=nodes.new("ShaderNodeTexNoise")
    cloud.inputs["Scale"].default_value=36.
    cloud.inputs["Detail"].default_value=2.8
    link(uv.outputs["Object"],cloud.inputs[0])
    ramp=nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color=(.342,.174,.143,1)
    ramp.color_ramp.elements[1].color=(.568,.355,.280,1)
    link(cloud.outputs["Fac"],ramp.inputs[0])
    pore=nodes.new("ShaderNodeTexNoise")
    pore.inputs["Scale"].default_value=1600.
    pore.inputs["Detail"].default_value=2.4
    link(uv.outputs["Object"],pore.inputs[0])
    bump=nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value=.30
    bump.inputs["Distance"].default_value=.000065
    link(pore.outputs["Fac"],bump.inputs["Height"])
    follicle = nodes.new("ShaderNodeTexVoronoi")
    follicle.inputs["Scale"].default_value = 790.
    link(uv.outputs["Object"], follicle.inputs["Vector"])
    pores = nodes.new("ShaderNodeMapRange")
    pores.interpolation_type = "SMOOTHERSTEP"
    pores.inputs["From Min"].default_value = .085
    pores.inputs["From Max"].default_value = .35
    pores.inputs["To Min"].default_value = .89
    link(follicle.outputs["Distance"], pores.inputs["Value"])
    dye = nodes.new("ShaderNodeVectorMath")
    dye.operation = "SCALE"
    link(ramp.outputs[0], dye.inputs[0])
    link(pores.outputs[0], dye.inputs["Scale"])
    link(dye.outputs[0], shader.inputs["Base Color"])
    rough = nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY_ADD"
    rough.inputs[1].default_value = .19
    rough.inputs[2].default_value = .45
    link(pore.outputs["Fac"], rough.inputs[0])
    link(rough.outputs[0], shader.inputs["Roughness"])
    skin_relief = nodes.new("ShaderNodeBump")
    skin_relief.inputs["Strength"].default_value = .28
    skin_relief.inputs["Distance"].default_value = .00010
    link(pores.outputs[0], skin_relief.inputs["Height"])
    link(bump.outputs[0], skin_relief.inputs["Normal"])
    link(skin_relief.outputs[0], shader.inputs["Normal"])
    return mat


def bind(scene,rig,addon):
    mesh=leg_surface()
    skin=bpy.data.objects.new("右腿 · 自研试穿腿模",mesh)
    scene.collection.objects.link(skin)
    mesh.materials.append(skin_finish())
    skin.parent=rig
    groups=[skin.vertex_groups.new(name=k+".R") for k in ("foot","shin","thigh")]
    for v in mesh.vertices:
        def smooth(z,lo,hi):
            u=max(0.,min(1.,(z-lo)/(hi-lo)))
            return u*u*(3-2*u)
        leg=smooth(v.co.z,.070,.129)
        thigh=smooth(v.co.z,.465,.555)
        for group,w in zip(groups,(1-leg,leg*(1-thigh),thigh)):
            if w>0:group.add([v.index],w,'REPLACE')
    skin.modifiers.new("试穿骨架","ARMATURE").object=rig
    source=skin.copy()
    source.name="试样 · 原袜"
    scene.collection.objects.link(source)
    source.material_slots[0].link='OBJECT'
    base=bpy.data.materials.new("试样 · 原袜底色")
    base.diffuse_color=(.052,.041,.046,1)
    source.material_slots[0].material=base
    with bpy.data.libraries.load(str(Path(__file__).with_name("weft-yarn.blend"))) as (_,dest):
        dest.materials=["Weft Lab • Dry yarn"]
    source["weft_live_finish"]=dest.materials[0]
    # The downloadable scene owns the appended data. Copies drop Blender's
    # reuse-on-append metadata pointing to the source library on this machine.
    for collection in (bpy.data.materials,bpy.data.node_groups,bpy.data.images):
        for block in list(collection):
            if block.library_weak_reference:
                name=block.name
                local=block.copy()
                block.user_remap(local)
                collection.remove(block)
                local.name=name
    source["weft_contact_skin"],source["weft_skin_slot"]=skin,0
    source["weft_dry_palette"],source["weft_dry_thickness"]=0,.53
    source["weft_visual_knit_scale"]=1.12
    scene.weft_sock,scene.weft_skin,scene.weft_side=source,skin,'R'
    bpy.context.view_layer.update()
    sock=importlib.import_module(addon+'.sock_length').apply_length(bpy.context,source,skin,0,.78)
    sock.name="Weft · 可抓取织物"
    return sock


def backdrop_mesh():
    """Continuous floor-to-wall sweep for both the front and profile cameras."""
    from math import cos, pi, sin
    sectors = 112
    radial = [(1.22, -.008)]
    for i in range(1, 27):
        angle = .5*pi*i/26
        radial.append((1.22+.86*sin(angle), -.008+.86*(1-cos(angle))))
    radial.append((2.08, 3.5))
    directions = [(cos(2*pi*j/sectors), sin(2*pi*j/sectors)) for j in range(sectors)]
    vertices = [(0., 0., -.008)]
    vertices.extend((radius*x, radius*y, z) for radius, z in radial for x, y in directions)
    faces = [(0, 1+j, 1+(j+1)%sectors) for j in range(sectors)]
    for ring in range(len(radial)-1):
        start = 1+ring*sectors
        for j in range(sectors):
            a, b = start+j, start+(j+1)%sectors
            faces.append((a, a+sectors, b+sectors, b))
    mesh = bpy.data.meshes.new("摄影棚 • 环形无缝背景")
    mesh.from_pydata(vertices, [], faces, shade_flat=False)
    mesh.update()
    return mesh


def studio(scene):
    mesh=backdrop_mesh()
    background=bpy.data.objects.new(mesh.name,mesh)
    scene.collection.objects.link(background)
    background.hide_select=True
    # An orthographic pick starts outside the studio. Keep the wraparound set
    # visible to render cameras and leave viewport rays free to hit the knit.
    background.hide_set(True)
    mat=bpy.data.materials.new("摄影棚 • 暖灰")
    mat.use_nodes=True
    diffuse=mat.node_tree.nodes.get("Principled BSDF")
    diffuse.inputs["Base Color"].default_value=(.059,.057,.053,1)
    diffuse.inputs["Roughness"].default_value=.94
    mesh.materials.append(mat)
    world=bpy.data.worlds.new("摄影棚 • 环境光")
    world.use_nodes=True
    world.node_tree.nodes["Background"].inputs[0].default_value=(.37,.40,.44,1)
    world.node_tree.nodes["Background"].inputs[1].default_value=.22
    scene.world=world
    lights=[
        ("柔光箱",(.56,.62,1.40),(.055,.025,.45),77,.88,1.20,(1.,.88,.80)),
        ("宽补光",(-.84,.73,1.0),(.08,0,.46),14,.99,1.30,(.85,.91,1.)),
        ("轮廓光",(-.64,-.65,1.16),(.08,0,.45),19,.72,.90,(.94,.96,1.)),
        ("背景光",(.55,-.25,.90),(.47,-1.59,.75),25,1.,1.,(.84,.86,.90)),
        ("冷条光",(-.54,.30,.81),(.08,0,.47),1.75,.47,.87,(.85,.93,1.)),
        ("暖条光",(.57,-.18,.88),(.08,0,.47),3.1,.59,1.03,(1.,.89,.77)),
    ]
    for name,location,target,watts,w,h,tint in lights:
        lamp=bpy.data.lights.new(name,'AREA')
        lamp.energy,lamp.color=watts,tint
        lamp.shape='RECTANGLE'
        lamp.size=w
        lamp.size_y=h
        obj=bpy.data.objects.new(name,lamp)
        scene.collection.objects.link(obj)
        obj.location=location
        obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
    scene.render.engine='BLENDER_EEVEE'
    scene.eevee.taa_render_samples=64
    scene.view_settings.view_transform='AgX'
    scene.view_settings.look='AgX - Medium High Contrast'


def setup(scene,rig):
    knee=rig.pose.bones["shin.R"]
    knee.rotation_mode='XYZ'
    for frame,angle in ((1,0.),(25,.15),(45,.30),(65,.15),(85,0.)):
        knee.rotation_euler.x=angle
        knee.keyframe_insert(data_path="rotation_euler",index=0,frame=frame)
    rig.animation_data.action.name="试穿腿模 · 屈膝"
    scene.frame_start,scene.frame_end=1,85
    scene.render.fps=30
    scene.frame_set(1)
    target=Vector((RIGHT_X,.021,.457))
    eye=Vector((.60,1.,.075)).normalized()
    facing=(-eye).to_track_quat('-Z','Y')
    lens=bpy.data.cameras.new("织物近景")
    lens.type='ORTHO'
    lens.ortho_scale=.64
    lens.clip_start=.005
    cam=bpy.data.objects.new(lens.name,lens)
    scene.collection.objects.link(cam)
    cam.location=target+eye*1.5
    cam.rotation_euler=facing.to_euler()
    scene.camera=cam
    scene.render.resolution_x,scene.render.resolution_y=1920,1080
    scene.render.image_settings.file_format='PNG'
    scene.render.filepath='//renders/weft-leg'
    scene.weft_wet.moving=False
    # Keep the travelling sheet clear enough to see the knit through the water.
    scene.weft_wet.color=(.984,.994,1.)
    scene.weft_wet.opacity=.49
    for screen in bpy.data.screens:
        for area in screen.areas:
            for editor in area.spaces:
                if editor.type=='FILE_BROWSER' and editor.params:
                    portable_path(editor.params,'directory',b'//')
            if area.type!='VIEW_3D':continue
            space=area.spaces.active
            space.region_3d.view_perspective='ORTHO'
            space.region_3d.view_location=target
            space.region_3d.view_rotation=facing
            space.region_3d.view_distance=.80
            space.clip_start=.005
            space.overlay.show_overlays=False
            space.shading.type='MATERIAL'
            space.shading.use_scene_lights=True
            space.shading.use_scene_world=True
            space.show_region_ui=True


def portable_path(owner,field,value):
    # Blender saves the entire fixed-length buffer, including unused bytes.
    padding='//'+'packed/'*85+'resource'
    setattr(owner,field,padding.encode() if isinstance(value,bytes) else padding)
    setattr(owner,field,value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    output=parser.parse_args(sys.argv[sys.argv.index('--')+1:]).output.resolve()
    if output.suffix.lower()!='.blend' or output.exists():
        raise ValueError('Choose an unused .blend filename.')
    if not bpy.app.background or bpy.data.filepath:
        raise RuntimeError('Start in a fresh background Blender process.')
    addon=next((a.module for a in bpy.context.preferences.addons if a.module.split('.')[-1]=='weft_replay'),None)
    if addon is None:raise RuntimeError('Install and enable Weft Lab first.')
    scene=bpy.data.scenes.new("Weft Lab · 干湿针织试样")
    bpy.context.window.scene=scene
    for previous in list(bpy.data.scenes):
        if previous!=scene:bpy.data.scenes.remove(previous)
    bpy.data.orphans_purge(do_recursive=True)
    scene.unit_settings.system='METRIC'
    rig=make_rig(scene)
    bind(scene,rig,addon)
    studio(scene)
    setup(scene,rig)
    note=bpy.data.texts.new("Weft Lab · 试穿说明")
    note.use_fake_user=True
    note.write("N → Weft Lab → 干态抓取，或湿态抓取 / 泼水。左键拖动，湿态按 W 加水。\n"
               "两种实验之间使用「结束实验并恢复原袜」。试穿腿模、针织材质与场景由 Weft Lab 制作。\n")
    output.parent.mkdir(parents=True,exist_ok=True)
    bpy.context.preferences.use_preferences_save=False
    bpy.context.preferences.filepaths.save_version=0
    for image in bpy.data.images:
        if not image.packed_files:continue
        location='//packed/'+Path(bpy.path.native_pathsep(image.filepath)).name
        portable_path(image,'filepath_raw',location)
        for packed in image.packed_files:
            portable_path(packed,'filepath',location)
    result=bpy.ops.wm.save_as_mainfile(filepath=str(output),compress=True,relative_remap=False)
    if result!={'FINISHED'}:raise RuntimeError('Could not save the leg study.')
    print('LEG MODEL',output.name,flush=True)


if __name__=='__main__':
    main()
