"""Reuse Blender's Knit UV tangents outside the part of the posed cloth that moved."""
import bpy
import numpy as np

FIELD='Weft live knit tangent'


def bind(material,sim):
    tree=material.node_tree
    wires=[(wire.from_socket,wire.to_socket) for n in tree.nodes if n.type=='TANGENT'
           and n.direction_type=='UV_MAP' and n.uv_map=='Knit' for wire in n.outputs[0].links]
    if not wires:return
    field=tree.nodes.new('ShaderNodeAttribute');field.attribute_name=FIELD
    transform=tree.nodes.new('ShaderNodeVectorTransform')
    transform.vector_type='VECTOR';transform.convert_from='OBJECT';transform.convert_to='WORLD'
    tree.links.new(field.outputs['Vector'],transform.inputs[0])
    for _,dest in wires:tree.links.new(transform.outputs[0],dest)
    sim.knit_shader=(tree,wires,field,transform)
    sim.knit_domain=None


def discard_domain(sim):
    domain=getattr(sim,'knit_domain',None)
    if domain is not None:
        bpy.data.meshes.remove(domain[-1])
        sim.knit_domain=None


def finish(sim,mesh):
    if not getattr(sim,'knit_shader',None):return
    tree,wires,field,transform=sim.knit_shader
    for src,dst in wires:tree.links.new(src,dst)
    tree.nodes.remove(transform);tree.nodes.remove(field)
    discard_domain(sim)
    attr=mesh.attributes.get(FIELD)
    if attr is not None:mesh.attributes.remove(attr)
    sim.knit_shader=None


def read_tangents(mesh):
    mesh.calc_tangents(uvmap='Knit')
    value=np.empty((len(mesh.loops),3),dtype='f')
    mesh.loops.foreach_get('tangent',value.ravel())
    mesh.free_tangents()
    return value


def initialize(mesh,sim):
    if not getattr(sim,'knit_shader',None):return
    discard_domain(sim)
    # This is also the exact value used for corners outside the moving patch.
    sim.knit_base=np.empty((len(mesh.vertices),3),'f')
    mesh.vertices.foreach_get('co',sim.knit_base.ravel())
    sim.knit_rest_tangents=read_tangents(mesh)
    sim.knit_tangents=sim.knit_rest_tangents.copy()
    field=mesh.attributes.get(FIELD) or mesh.attributes.new(FIELD,'FLOAT_VECTOR','CORNER')
    field.data.foreach_set('vector',sim.knit_tangents.ravel())


def make_domain(mesh,sim,coords,moved):
    previous=sim.knit_domain
    if previous is not None:
        slots=previous[-2]
        sim.knit_tangents[slots]=sim.knit_rest_tangents[slots]
    discard_domain(sim)
    faces=sim.display_faces()
    support=moved.copy()
    if sim.grip is not None:support[sim.drag_ids]=True
    # Face normals change on incident faces. A corner tangent uses the entire
    # smooth fan of its vertex, including faces at the outer edge of the patch.
    first=support[faces].any(1)
    affected=np.zeros(len(coords),bool);affected[faces[first]]=True
    subset=np.flatnonzero(affected[faces].any(1))
    ids,inverse=np.unique(faces[subset],return_inverse=True)
    local=inverse.reshape(-1,3).astype('i')
    temp=bpy.data.meshes.new('Weft • tangent neighbourhood')
    temp.vertices.add(len(ids));temp.vertices.foreach_set('co',coords[ids].ravel())
    temp.loops.add(local.size)
    temp.attributes['.corner_vert'].data.foreach_set('value',local.ravel())
    temp.polygons.add(len(local))
    temp.polygons.foreach_set('loop_start',np.arange(0,local.size,3,dtype='i'))
    temp.polygons.foreach_set('use_smooth',np.ones(len(local),bool))
    temp.update(calc_edges=True)
    uv=np.empty((len(mesh.loops),2),'f')
    mesh.uv_layers['Knit'].uv.foreach_get('vector',uv.ravel())
    temp.uv_layers.new(name='Knit').uv.foreach_set('vector',uv.reshape(-1,3,2)[subset].ravel())
    temp.update()
    targets=np.flatnonzero(affected[faces[subset]].ravel())
    slots=(subset[:,None]*3+np.arange(3))[targets//3,targets%3]
    sim.knit_domain=(support,ids,targets,slots,temp)
    return sim.knit_domain


def update(mesh,sim,coordinates):
    if not getattr(sim,'knit_shader',None) or not hasattr(sim,'knit_base'):return
    coords=np.ascontiguousarray(coordinates,dtype='f')
    moved=(coords!=sim.knit_base).any(1)
    domain=sim.knit_domain
    if not moved.any() and domain is None:return
    if domain is None or np.any(moved & ~domain[0]):
        domain=make_domain(mesh,sim,coords,moved)
    _,ids,targets,slots,patch=domain
    patch.vertices.foreach_set('co',coords[ids].ravel());patch.update()
    sim.knit_tangents[slots]=read_tangents(patch)[targets]
    mesh.attributes[FIELD].data.foreach_set('vector',sim.knit_tangents.ravel())
