import bpy
import numpy as np
from mathutils import Matrix,Vector
from .cache import DATA,assets,frames,xyz

def mesh_object(scene,name,points,faces):
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(points.tolist(),[],faces.tolist());mesh.update()
    mesh.polygons.foreach_set("use_smooth",np.ones(len(faces),dtype=bool))
    obj=bpy.data.objects.new(name,mesh);scene.collection.objects.link(obj);return obj

def add_uv(mesh,name,uv):
    layer=mesh.uv_layers.new(name=name)
    indices=np.empty(len(mesh.loops),np.int32);mesh.loops.foreach_get("vertex_index",indices)
    layer.uv.foreach_set("vector",np.ascontiguousarray(uv[indices],dtype=np.float32).ravel())

def surface(name,color):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    bs=m.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value=(*color,1);bs.inputs["Roughness"].default_value=.63
    return m

def knit_material():
    m=surface("Weft • knit / 黑色针织",(.034,.032,.037))
    tree=m.node_tree;nodes=tree.nodes;link=tree.links.new
    bs=nodes.get("Principled BSDF");bs.inputs["Roughness"].default_value=.64
    bs.inputs["Specular IOR Level"].default_value=.21
    uv=nodes.new("ShaderNodeUVMap");uv.uv_map="Weft.Material"
    atlas=nodes.new("ShaderNodeUVMap");atlas.uv_map="Weft.Portable"
    base=nodes.new("ShaderNodeTexImage");base.image=bpy.data.images.load(str(DATA/"coverage.png"),check_existing=True)
    base.image["weft_asset"]="coverage.png"
    link(atlas.outputs[0],base.inputs[0])
    bounds=frames()[0]["bounds"];sub=nodes.new("ShaderNodeVectorMath");sub.operation="SUBTRACT"
    link(uv.outputs[0],sub.inputs[0]);sub.inputs[1].default_value=(bounds[0],bounds[1],0)
    scale=nodes.new("ShaderNodeVectorMath");scale.operation="MULTIPLY"
    link(sub.outputs[0],scale.inputs[0]);scale.inputs[1].default_value=(1/(bounds[2]-bounds[0]),1/(bounds[3]-bounds[1]),1)
    tex=nodes.new("ShaderNodeTexImage");tex.label="逐帧圈形 · UV 空间"
    image=bpy.data.images.load(str(DATA/"knit"/"knit0001.png"),check_existing=True)
    image["weft_asset"]="knit/knit0001.png"
    image.source="SEQUENCE";image.colorspace_settings.name="Non-Color";tex.image=image;tex.extension="CLIP"
    tex.image_user.frame_duration=len(frames());tex.image_user.use_auto_refresh=True
    link(scale.outputs[0],tex.inputs[0])
    channels=nodes.new("ShaderNodeSeparateColor");link(tex.outputs["Color"],channels.inputs[0])
    relief=nodes.new("ShaderNodeMath");relief.operation="MULTIPLY"
    link(channels.outputs[1],relief.inputs[0]);link(channels.outputs[2],relief.inputs[1])
    bump=nodes.new("ShaderNodeBump");bump.inputs["Strength"].default_value=.22;bump.inputs["Distance"].default_value=.00006
    link(relief.outputs[0],bump.inputs["Height"]);link(bump.outputs[0],bs.inputs["Normal"])
    blend=nodes.new("ShaderNodeMixRGB");link(tex.outputs["Alpha"],blend.inputs[0]);link(base.outputs["Alpha"],blend.inputs[1]);link(channels.outputs[0],blend.inputs[2])
    link(blend.outputs[0],bs.inputs["Alpha"])
    return m

def set_camera(obj,camera,aspect):
    pos=Vector(xyz(camera["position"]));target=Vector(xyz(camera["target"]))
    forward=(target-pos).normalized();right=forward.cross(Vector(xyz(camera["up"]))).normalized()
    obj.location=pos;obj.rotation_euler=Matrix((right,right.cross(forward),-forward)).transposed().to_euler()
    obj.data.ortho_scale=camera["height"]*max(1,aspect)

def create(context):
    scene=bpy.data.scenes.new("Weft Lab • 脚踝回放")
    scene["weft_replay"]=1;scene.render.fps=30;scene.frame_start=1;scene.frame_end=len(frames())
    scene.render.resolution_x=720;scene.render.resolution_y=480;scene.render.resolution_percentage=100
    scene.render.engine="CYCLES";scene.cycles.samples=32;scene.cycles.use_denoising=True
    scene.render.use_lock_interface=True;scene.view_settings.view_transform="Standard"
    scene.world=bpy.data.worlds.new("Weft • studio");scene.world.use_nodes=True
    world=scene.world.node_tree.nodes.get("Background");world.inputs[0].default_value=(.28,.32,.31,1);world.inputs[1].default_value=.8
    for key,(verts,faces,uv,uv1) in assets().items():
        obj=mesh_object(scene,"Weft • "+key,verts,faces);obj["weft_role"]=key
        if key=="Stocking":
            add_uv(obj.data,"Weft.Material",uv)
            uv1=uv1.copy();uv1[:,1]=1-uv1[:,1];add_uv(obj.data,"Weft.Portable",uv1)
            obj.data.materials.append(knit_material())
        else:obj.data.materials.append(surface("Weft • skin",(.55,.30,.20)))
    for name in ("main","detail"):
        cam=bpy.data.cameras.new("Weft • "+name);cam.type="ORTHO";cam.clip_start=.001
        obj=bpy.data.objects.new(cam.name,cam);scene.collection.objects.link(obj);obj["weft_role"]=name
        set_camera(obj,frames()[0][name],1.5)
        if name=="main":scene.camera=obj
    for i,(p,power,size) in enumerate([((.4,-.55,.67),12,.45),((-.4,-.18,.32),6,.36)]):
        lamp=bpy.data.lights.new("Weft • softbox","AREA");lamp.energy=power;lamp.shape="DISK";lamp.size=size
        obj=bpy.data.objects.new(lamp.name,lamp);scene.collection.objects.link(obj);obj.location=p
        obj.rotation_euler=(Vector(xyz(frames()[0]["anchor"]))-obj.location).to_track_quat('-Z','Y').to_euler()
    for name,f in (("开始拉起",31),("停手",34),("松手",43),("松手后 0.40 s",55)):
        scene.timeline_markers.new(name,frame=f)
    context.window.scene=scene
    for area in context.screen.areas:
        if area.type=="VIEW_3D":area.spaces.active.region_3d.view_perspective="CAMERA"
    scene.frame_set(1);return scene

