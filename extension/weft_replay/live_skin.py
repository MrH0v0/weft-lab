"""Display the same transient pressure-deformed skin used for cloth contact."""
import bpy
import numpy as np


class SkinDisplay:
    def __init__(self, ctx, skin):
        self.skin = skin
        self.hidden = skin.hide_get(),skin.hide_render
        graph = ctx.evaluated_depsgraph_get()
        source = skin.evaluated_get(graph)
        mesh = bpy.data.meshes.new_from_object(source,preserve_all_data_layers=True,depsgraph=graph)
        self.obj = bpy.data.objects.new(skin.name+" • contact response",mesh)
        ctx.scene.collection.objects.link(self.obj)
        self.obj["weft_skin_response_source"] = skin
        self.obj["weft_skin_response_hidden"] = self.hidden[0]
        self.obj["weft_skin_response_render_hidden"] = self.hidden[1]
        skin.hide_set(True)
        skin.hide_render = True

    def publish(self, pose):
        self.obj.matrix_world = pose.skin_transform
        inverse = np.asarray(pose.skin_transform.inverted())
        coords = pose.skin_vertices@inverse[:3,:3].T+inverse[:3,3]
        self.obj.data.vertices.foreach_set("co",coords.astype('f').ravel())
        self.obj.data.update()


def publish_skin(ctx, sim):
    pose = sim.pose
    if pose is None or not pose.responds_to_pressure:
        return
    display = getattr(sim,"skin_display",None)
    if display is None:
        if not pose.skin_deformed_vertices:
            return
        display = sim.skin_display = SkinDisplay(ctx,pose.skin)
    display.publish(pose)


def remove_skin_display(obj):
    skin = obj.get("weft_skin_response_source")
    if skin is not None:
        skin.hide_set(obj["weft_skin_response_hidden"])
        skin.hide_render = obj["weft_skin_response_render_hidden"]
    mesh = obj.data
    bpy.data.objects.remove(obj,do_unlink=True)
    if not mesh.users:
        bpy.data.meshes.remove(mesh)


def restore_skin(sim):
    display = getattr(sim,"skin_display",None)
    if display is not None:
        remove_skin_display(display.obj)
        sim.skin_display = None
