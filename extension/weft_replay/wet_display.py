"""Liquid overlays and mesh attributes for the same committed wet step."""
from itertools import chain

import bpy
import numpy as np
from mathutils import Vector

from .live_finish import TEMPLATE
from .splash_optics import CONTACT_DEPTH_M, PackedFaces, attached_film, impact_beads, ligaments, merge
from .wet_film import DT, GRAVITY, PULSE
from .wet_scene import CENTER, OUTER_MM, SATURATION, SHAPE, SIZE, SKIN_MM, TRIANGLES


CONTACT_DEPTH_MM = 1000 * CONTACT_DEPTH_M


def write_field(mesh, name, values, domain='POINT'):
    attr = mesh.attributes.get(name) or mesh.attributes.new(name, 'FLOAT', domain)
    attr.data.foreach_set('value', np.asarray(values, dtype='f'))


def water_shader(name, liquid, *, clear_optics=False):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    if clear_optics:
        return clear_water_shader(mat, liquid)
    mat.surface_render_method = 'BLENDED'
    mat.use_transparency_overlap = False
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*liquid.color, 1.)
    shader.inputs['IOR'].default_value = 1.333
    shader.inputs['Roughness'].default_value = .18
    shader.inputs['Coat Weight'].default_value = .55
    shader.inputs['Emission Color'].default_value = (*liquid.color, 1.)
    shader.inputs['Emission Strength'].default_value = .23
    field = nodes.new('ShaderNodeAttribute')
    field.attribute_name = 'weft_liquid_visibility'
    alpha = nodes.new('ShaderNodeMath')
    alpha.operation = 'MULTIPLY'
    alpha.inputs[1].default_value = liquid.opacity
    links.new(field.outputs['Fac'], alpha.inputs[0])
    links.new(alpha.outputs[0], shader.inputs['Alpha'])
    return mat


def clear_water_shader(mat, liquid):
    mat.surface_render_method = 'DITHERED'
    mat.use_raytrace_refraction = True
    nodes, link = mat.node_tree.nodes, mat.node_tree.links.new
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['Color'].default_value = (*liquid.color, 1.)
    glass.inputs['IOR'].default_value = 1.333
    roughness = nodes.new('ShaderNodeAttribute')
    roughness.attribute_name = 'liquid_surface_roughness'
    link(roughness.outputs['Fac'], glass.inputs['Roughness'])
    field = nodes.new('ShaderNodeAttribute')
    field.attribute_name = 'weft_liquid_visibility'
    strength = nodes.new('ShaderNodeMath')
    strength.operation = 'MULTIPLY'
    strength.inputs[1].default_value = .88 * liquid.opacity
    link(field.outputs['Fac'], strength.inputs[0])
    mix = nodes.new('ShaderNodeMixShader')
    link(strength.outputs[0], mix.inputs[0])
    link(nodes.new('ShaderNodeBsdfTransparent').outputs[0], mix.inputs[1])
    link(glass.outputs[0], mix.inputs[2])
    link(mix.outputs[0], out.inputs['Surface'])
    out.inputs['Thickness'].default_value = .001
    return mat


def attached_water_shader(liquid):
    """A single blended interface for layers too thin to refract the full sock twice."""
    mat=bpy.data.materials.new('Weft • wet surface sheen')
    mat.use_nodes=True
    mat.surface_render_method='BLENDED'
    mat.use_transparency_overlap=False
    mat.use_transparent_shadow=False
    nodes,link=mat.node_tree.nodes,mat.node_tree.links.new
    shader=nodes.get('Principled BSDF')
    # A thin layer on yarn reflects the environment. Its small absorption
    # tint leaves the dyed knit visible through the centre of the water.
    shader.inputs['Base Color'].default_value=(*(c*.09 for c in liquid.color),1.)
    shader.inputs['IOR'].default_value=1.333
    shader.inputs['Coat Weight'].default_value=.16
    shader.inputs['Coat Roughness'].default_value=.11
    rough=nodes.new('ShaderNodeAttribute');rough.attribute_name='liquid_surface_roughness'
    link(rough.outputs['Fac'],shader.inputs['Roughness'])
    visibility=nodes.new('ShaderNodeAttribute');visibility.attribute_name='weft_liquid_visibility'
    fresnel=nodes.new('ShaderNodeFresnel');fresnel.inputs['IOR'].default_value=1.333
    response=nodes.new('ShaderNodeMath');response.operation='MULTIPLY_ADD'
    link(fresnel.outputs[0],response.inputs[0])
    response.inputs[1].default_value=.74*liquid.opacity
    response.inputs[2].default_value=.09*liquid.opacity
    alpha=nodes.new('ShaderNodeMath');alpha.operation='MULTIPLY'
    link(visibility.outputs['Fac'],alpha.inputs[0]);link(response.outputs[0],alpha.inputs[1])
    link(alpha.outputs[0],shader.inputs['Alpha'])
    return mat


def shade_wet_knit(material):
    """Add saturation-dependent darkening and roughness to 01's live shader."""
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    sat = nodes.new('ShaderNodeAttribute')
    sat.attribute_name = SATURATION
    for name, floor in (('Base Color', (.012, .017, .023, 1)), ('Roughness', .29)):
        socket = bsdf.inputs[name]
        blend = nodes.new('ShaderNodeMixRGB' if name == 'Base Color' else 'ShaderNodeMix')
        if name == 'Base Color':
            fac, a, b = blend.inputs
            output = blend.outputs[0]
        else:
            fac, a, b = blend.inputs[0], blend.inputs[2], blend.inputs[3]
            output = blend.outputs[0]
        if socket.is_linked:
            links.new(socket.links[0].from_socket, a)
        else:
            a.default_value = socket.default_value
        b.default_value = floor
        links.new(sat.outputs['Fac'], fac)
        links.new(output, socket)


def shade_approved_knit(material):
    """Use the same dye response as the approved bake, from current mesh fields."""
    tree = material.node_tree
    nodes, link = tree.nodes, tree.links.new
    bsdf = nodes['Principled BSDF']

    def field(name):
        node = nodes.new('ShaderNodeAttribute')
        node.attribute_name = name
        return node.outputs['Fac']

    def math(op, *args):
        node = nodes.new('ShaderNodeMath')
        node.operation = op
        for socket, value in zip(node.inputs, args):
            if isinstance(value, (float, int)):
                socket.default_value = value
            else:
                link(value, socket)
        return node.outputs[0]

    def original(name):
        channel = bsdf.inputs[name]
        return channel.links[0].from_socket if channel.is_linked else channel.default_value

    sat = field(SATURATION)
    contact = math('MINIMUM', 1., math('DIVIDE', field(OUTER_MM), CONTACT_DEPTH_MM))
    damp = math('MAXIMUM', sat, math('MULTIPLY', .7, contact))
    color = nodes.new('ShaderNodeVectorMath')
    color.operation = 'SCALE'
    color.label = 'Live wet dye response'
    link(original('Base Color'), color.inputs[0])
    link(math('MULTIPLY_ADD', damp, -.59, 1.), color.inputs['Scale'])
    link(color.outputs[0], bsdf.inputs['Base Color'])
    link(math('MULTIPLY',original('Alpha'),math('MULTIPLY_ADD',damp,-.15,1.)),bsdf.inputs['Alpha'])
    link(math('MULTIPLY_ADD', damp, -.19, original('Roughness')), bsdf.inputs['Roughness'])
    for name, loss in (('Specular IOR Level', .36), ('Sheen Weight', .70)):
        link(math('MULTIPLY', original(name), math('MULTIPLY_ADD', damp, -loss, 1.)), bsdf.inputs[name])
    link(math('MULTIPLY_ADD', contact, .17 * .86, math('MULTIPLY', sat, .12 * .86)), bsdf.inputs['Coat Weight'])
    link(math('MULTIPLY_ADD', contact, -.22, math('MULTIPLY_ADD', sat, -.10, .47)), bsdf.inputs['Coat Roughness'])
    bsdf.inputs['Coat IOR'].default_value = 1.333
    link(original('Normal'), bsdf.inputs['Coat Normal'])


def make_object(scene, name, points, faces, mat, owner):
    data = bpy.data.meshes.new(name)
    data.from_pydata(points.tolist(), [], faces.tolist())
    data.materials.append(mat)
    data.polygons.foreach_set('use_smooth', np.ones(len(faces), bool))
    obj = bpy.data.objects.new(name, data)
    obj['weft_wet_owner'] = owner
    scene.collection.objects.link(obj)
    return obj


def make_mark(scene, name, rgb, owner):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*rgb, 1.)
    material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*rgb, 1.)
    shader.inputs['Emission Color'].default_value = (*rgb, 1.)
    shader.inputs['Emission Strength'].default_value = .8
    theta = np.arange(32) * (2*np.pi/32)
    xy = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(32)))
    points = np.vstack((xy*.00146, xy*.00111))
    i = np.arange(32)
    j = (i + 1) % 32
    faces = np.column_stack((i, j, j+32, i+32))
    return make_object(scene, name, points, faces, material, owner)


def write_points(obj, xyz):
    obj.data.vertices.foreach_set('co', np.asarray(xyz, dtype='f').ravel())
    obj.data.update()


def normals(points, fallback):
    grid = points.reshape(*SHAPE, 3)
    y, x = np.gradient(grid, axis=(0, 1))
    n = np.cross(x, y).reshape(-1, 3)
    length = np.linalg.norm(n, axis=1)
    good = (length > 1e-10) & (n @ fallback > 0.)
    n[good] /= length[good, None]
    n[~good] = fallback
    return n


def remove_wet_objects(scene, owner):
    for obj in list(scene.objects):
        if obj.get('weft_wet_owner') != owner:
            continue
        mesh = obj.data
        mat = mesh.materials[0]
        bpy.data.objects.remove(obj, do_unlink=True)
        if not mesh.users:
            bpy.data.meshes.remove(mesh)
        if not mat.users:
            bpy.data.materials.remove(mat)


class WetDisplay:
    def __init__(self, scene, obj, surface):
        self.scene, self.obj, self.surface = scene, obj, surface
        material = obj.data.materials[0]
        self.approved = bool(material.get(TEMPLATE))
        self.material = water_shader('Weft • liquid', surface.film.liquid, clear_optics=self.approved)
        surface_material=attached_water_shader(surface.film.liquid) if self.approved else self.material
        self.inner_faces=TRIANGLES[surface.skin_valid[TRIANGLES].all(axis=1)]
        self.inner_active=None
        self.outer, self.skin = [make_object(scene, name, points, TRIANGLES[valid[TRIANGLES].all(axis=1)],
                                              surface_material, obj)
                                 for name, points, valid in (
                                     ('Weft • outer film', surface.sock_points(), surface.sock_valid),
                                     ('Weft • skin film', surface.skin_points(), surface.skin_valid))]
        self.sheets = []
        self.published_faces={}
        if self.approved:
            write_field(self.skin.data, 'liquid_surface_roughness', np.full(len(surface.skin_valid), .115))
        self.rows = 25
        i, j = np.meshgrid(np.arange(16), np.arange(self.rows-1))
        a = (17*j+i).ravel()
        self.sheet_faces = np.column_stack((a, a+1, a+18, a+17))
        self.sock_mark = make_mark(scene, 'Weft • sock marker', (.22, .95, .45), obj)
        self.skin_mark = make_mark(scene, 'Weft • skin marker', (1., .39, .08), obj)
        if not material.get('weft_wet_shader'):
            (shade_approved_knit if self.approved else shade_wet_knit)(material)
            material['weft_wet_shader'] = True

    def publish(self):
        wet = self.surface
        f = wet.film
        skin, origin, normal, _ = wet.layout()
        sock = wet.sock_points()
        skin_n, sock_n = normals(skin, normal), normals(sock, normal)
        outer = f.depth(f.outer, f.area)
        inner = f.depth(f.skin, f.skin_area)
        saturation = f.knit / np.maximum(f.capacity, 1e-12)
        if self.approved:
            self.publish_optics(sock, outer, normal)
        else:
            self.surface_visual(self.outer, sock, sock_n, outer, .4, wet.sock_valid)
        if self.approved:self.publish_inner(skin,skin_n,inner)
        else:self.surface_visual(self.skin, skin, skin_n, inner, .023, wet.skin_valid)
        for attr, values in ((OUTER_MM, 1000*outer), (SATURATION, saturation)):
            write_field(self.obj.data, attr, np.sum(values[wet.shader_ids]*wet.shader_weights, axis=1))
        self.obj.data.update()
        marker_normal = Vector(normal)
        space=bpy.context.space_data
        show_marks=space is None or space.type!='VIEW_3D' or space.overlay.show_overlays
        for obj, point, n in ((self.sock_mark, sock[CENTER+SIZE], sock_n[CENTER+SIZE]),
                              (self.skin_mark, skin[CENTER-SIZE], skin_n[CENTER-SIZE])):
            obj.hide_render=True
            if obj.hide_get()==show_marks:obj.hide_set(not show_marks)
            obj.location = point + n*.00038
            obj.rotation_euler = marker_normal.to_track_quat('Z', 'Y').to_euler()
        if self.approved:
            return
        while len(self.sheets) < len(f.pours):
            sheet = make_object(self.scene, 'Weft • incoming water sheet', np.zeros((17*self.rows, 3)),
                                self.sheet_faces, self.material, self.obj)
            self.sheets.append(sheet)
        for i, sheet in enumerate(self.sheets):
            if i >= len(f.pours):
                sheet.hide_render = True
                sheet.hide_set(True)
                continue
            p = f.pours[i]
            end = np.minimum(p.flight, np.maximum(0., p.age - p.delay))
            start = np.maximum(0., p.age - p.delay - PULSE)
            active = end > start
            age = np.linspace(np.minimum(start, end), end, self.rows)
            xyz = p.points(age)
            emission_age = np.clip((p.age - p.delay - age) / PULSE, 0., 1.)
            flux = (1-np.cos(2*np.pi*emission_age)) / PULSE
            strip = np.linalg.norm(np.gradient(xyz, axis=1), axis=2)
            speed = np.linalg.norm(p.velocity+age[..., None]*GRAVITY, axis=2)
            thickness = p.volume*1e-6*p.shares*flux / np.maximum(1e-7, strip*speed)
            opacity = np.clip(np.sqrt(thickness/.0014), 0., .98)*active
            sheet.hide_set(not active.any())
            sheet.hide_render = not active.any()
            write_points(sheet, xyz.reshape(-1, 3))
            write_field(sheet.data, 'weft_liquid_visibility', opacity.ravel())
            sheet.data.update()

    def publish_inner(self,skin,normals,depth):
        # Cells outside the inner liquid footprint must not contribute a
        # refractive square or write depth in front of the visible knit.
        alpha=-np.expm1(-np.maximum(depth,0.)/.000023)
        active=alpha[self.inner_faces].max(axis=1)>1./255.
        obj=self.skin;obj.hide_render=not active.any();obj.hide_set(obj.hide_render)
        if obj.hide_render:return
        points=skin+normals*(.000035+np.minimum(depth*.8,.00125))[:,None]
        mesh=obj.data
        if self.inner_active is None or not np.array_equal(active,self.inner_active):
            mesh.clear_geometry()
            mesh.from_pydata(points.tolist(),[],self.inner_faces[active].tolist())
            mesh.polygons.foreach_set('use_smooth',np.ones(int(active.sum()),bool))
            self.inner_active=active
        else:mesh.vertices.foreach_set('co',points.astype('f').ravel())
        write_field(mesh,'weft_liquid_visibility',alpha)
        write_field(mesh,SKIN_MM,depth*1000)
        write_field(mesh,'liquid_surface_roughness',np.full(len(depth),.13))
        mesh.update()

    def publish_optics(self, sock, depth, normal):
        wet = self.surface
        f = wet.film
        pours = f.pours
        free = float(np.sum(np.maximum(0., depth-CONTACT_DEPTH_M) * f.area))
        spray_volume = 0.
        while len(self.sheets) < len(pours):
            sheet = make_object(self.scene, 'Weft • incoming water', np.empty((0, 3)),
                                np.empty((0, 4), dtype=np.int32), self.material, self.obj)
            self.sheets.append(sheet)
        for i, sheet in enumerate(self.sheets):
            model = None
            if i < len(pours):
                pour = pours[i]
                spray, volume = impact_beads(pour, normal, free-spray_volume)
                spray_volume += volume
                model = merge((ligaments(pour), spray))
            self.publish_model(sheet, model)
        model = attached_film(sock.reshape(*SHAPE, 3), depth.reshape(SHAPE), f.area.reshape(SHAPE),
                              normal, f.step*DT, spray_volume=spray_volume, valid=wet.sock_valid.reshape(SHAPE),packed=True)
        self.publish_model(self.outer, model)

    def publish_model(self, obj, model):
        obj.hide_render = model is None
        obj.hide_set(obj.hide_render)
        if model is None:
            return
        points, faces, alpha, rough = model
        data = obj.data
        if isinstance(faces,PackedFaces):
            sizes,corners=faces.sizes,faces.corners
        else:
            sizes=np.fromiter(map(len,faces),dtype=np.int32,count=len(faces))
            corners=np.fromiter(chain.from_iterable(faces),dtype=np.int32,count=int(sizes.sum()))
        old=self.published_faces.get(obj)
        changed=(old is None or len(points)!=len(data.vertices) or not np.array_equal(sizes,old[0])
                 or not np.array_equal(corners,old[1]))
        if changed:
            data.clear_geometry()
            data.vertices.add(len(points))
            data.loops.add(len(corners))
            corner_vertex=data.attributes.get('.corner_vert')
            if corner_vertex is None:data.loops.foreach_set('vertex_index',corners)
            else:corner_vertex.data.foreach_set('value',corners)
            data.polygons.add(len(sizes))
            starts=np.r_[0,sizes[:-1].cumsum(dtype=np.int32)].astype(np.int32,copy=False)
            data.polygons.foreach_set('loop_start',starts)
            data.polygons.foreach_set('use_smooth',np.ones(len(sizes),bool))
            self.published_faces[obj]=sizes,corners
        data.vertices.foreach_set('co',np.asarray(points,dtype='f').ravel())
        write_field(data, 'weft_liquid_visibility', alpha, 'FACE')
        write_field(data, 'liquid_surface_roughness', .76 * rough, 'FACE')
        data.update(calc_edges=changed)

    @staticmethod
    def surface_visual(obj, points, n, depth, scale_mm, valid):
        offset = np.minimum(depth*.8, .00125) + .000045
        alpha = np.clip(np.sqrt(depth*1000/scale_mm), 0., .97)*valid
        write_points(obj, points + n*offset[:, None])
        write_field(obj.data, 'weft_liquid_visibility', alpha)
        write_field(obj.data, SKIN_MM if 'skin' in obj.name else OUTER_MM, depth*1000)
        obj.hide_render = not bool(alpha.any())
        obj.hide_set(obj.hide_render)
