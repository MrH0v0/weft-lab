"""Bake fixed-action, two-endpoint vertex contact in deformed object space."""
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

CLEARANCE = .000075

def key_factor_driver(driver,sock):
    driver.type='SCRIPTED'
    for item in list(driver.variables):
        driver.variables.remove(item)
    key=sock.data.shape_keys.key_blocks[sock['weft_bound']]
    for name,prop in (('weight','value'),('muted','mute')):
        variable=driver.variables.new()
        variable.name=name
        variable.type='SINGLE_PROP'
        target=variable.targets[0]
        target.id_type='KEY'
        target.id=sock.data.shape_keys
        target.data_path=key.path_from_id(prop)
    driver.expression='weight * (1 - muted)'

def local_vertices(mesh):
    p=np.empty((len(mesh.vertices),3),np.float32)
    if attribute:=mesh.attributes.get('position'):
        attribute.data.foreach_get('vector',p.ravel())
    else:
        mesh.vertices.foreach_get('co',p.ravel())
    return p


def world_vertices(obj):
    p = local_vertices(obj.data)
    matrix = np.asarray(obj.matrix_world)
    if np.array_equal(matrix,np.eye(4)):
        return p.astype(float)
    return p@matrix[:3,:3].T+matrix[:3,3]

def correction(p,tree,faces):
    q,n,face,_ = tree.find_nearest(p)
    if (p-q).dot(n)>=CLEARANCE:
        return Vector((0,0,0))
    candidate=q+n*CLEARANCE
    def acceptable(point):
        x,normal,_,_=tree.find_nearest(point)
        return (point-x).dot(normal) >= CLEARANCE*.45
    if acceptable(candidate):
        return candidate-p
    # At a folded edge several adjacent triangles can be equidistant and
    # disagree on the normal. Test points inside their faces, then take the
    # nearest verified exterior candidate. No frame or vertex exceptions.
    radius=max(.001,(p-q).length+.0004)
    for _ in range(5):
        options=[]
        for hit,normal,index,_ in tree.find_nearest_range(p,radius):
            center=Vector(faces[index].mean(axis=0))
            for t in (.03,.15,.5,1):
                point=hit.lerp(center,t)+normal*CLEARANCE
                if acceptable(point):
                    options.append(point)
                    break
        if options:
            return min(options,key=lambda v:(v-p).length_squared)-p
        radius*=2
    raise RuntimeError("Unresolved posed contact at "+str(p))

def add_lookup(scene,sock,offsets):
    # Zero slot is shared by all untouched vertices.
    moving=np.any(offsets!=0,axis=(0,1,3))
    width=int(moving.sum())+1
    slots=np.zeros(len(moving),np.int32)
    slots[moving]=np.arange(1,width)
    attr=sock.data.attributes.new('weft_contact_slot','INT','POINT')
    attr.data.foreach_set('value',slots)
    blocks=np.zeros((*offsets.shape[:2],width,3),np.float32)
    blocks[:,:,1:]=offsets[:,:,moving]
    mesh=bpy.data.meshes.new('Posed contact offsets')
    mesh.vertices.add(blocks.size//3)
    mesh.vertices.foreach_set('co',blocks.ravel())
    mesh.update()
    cache=bpy.data.objects.new('Weft • fixed-action contact cache',mesh)
    scene.collection.objects.link(cache)
    cache.hide_render=True
    cache.hide_set(True)
    nt=bpy.data.node_groups.new('Weft • posed contact replay','GeometryNodeTree')
    nt.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    nt.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes=nt.nodes
    link=nt.links.new
    def math_node(op,left,right):
        a=nodes.new('ShaderNodeMath')
        a.operation=op
        for socket,v in zip(a.inputs,(left,right)):
            if isinstance(v,(float,int)):socket.default_value=v
            else:link(v,socket)
        return a.outputs[0]
    time=nodes.new('GeometryNodeInputSceneTime').outputs['Frame']
    frame=math_node('MINIMUM',math_node('MAXIMUM',math_node('SUBTRACT',time,scene.frame_start),0),len(offsets)-1)
    frame=math_node('FLOOR',frame,0)
    named=nodes.new('GeometryNodeInputNamedAttribute')
    named.data_type='INT'
    named.inputs['Name'].default_value=attr.name
    base=math_node('ADD',math_node('MULTIPLY',frame,2*width),named.outputs['Attribute'])
    info=nodes.new('GeometryNodeObjectInfo')
    info.inputs['Object'].default_value=cache
    def sample(idx):
        node=nodes.new('GeometryNodeSampleIndex')
        node.data_type='FLOAT_VECTOR'
        link(info.outputs['Geometry'],node.inputs['Geometry'])
        link(nodes.new('GeometryNodeInputPosition').outputs[0],node.inputs['Value'])
        link(idx,node.inputs['Index'])
        return node.outputs['Value']
    mix=nodes.new('ShaderNodeMix')
    mix.data_type='VECTOR'
    factor=next(i for i in mix.inputs if i.enabled and i.name=='Factor')
    curve=factor.driver_add('default_value')
    key_factor_driver(curve.driver,sock)
    for name,value in [('A',sample(base)),('B',sample(math_node('ADD',base,width)))]:
        link(value,next(i for i in mix.inputs if i.enabled and i.name==name))
    set_pos=nodes.new('GeometryNodeSetPosition')
    link(nodes.new('NodeGroupInput').outputs['Geometry'],set_pos.inputs['Geometry'])
    link(next(o for o in mix.outputs if o.enabled),set_pos.inputs['Offset'])
    link(set_pos.outputs['Geometry'],nodes.new('NodeGroupOutput').inputs['Geometry'])
    mod=sock.modifiers.new('Weft • fixed-action skin contact','NODES')
    mod.node_group=nt
    print('contact slots',width,'packed bytes',blocks.nbytes,flush=True)

def bake_contact(scene,body,sock,materials=None):
    graph=bpy.context.evaluated_depsgraph_get()
    key=sock.data.shape_keys.key_blocks[sock['weft_bound']]
    orig=key.value
    offsets=[]
    for f in range(scene.frame_start,scene.frame_end+1):
        scene.frame_set(f)
        skin=body.evaluated_get(graph)
        pos=world_vertices(skin)
        skin.data.calc_loop_triangles()
        indices=np.empty((len(skin.data.loop_triangles),3),np.int32)
        skin.data.loop_triangles.foreach_get('vertices',indices.ravel())
        if materials is not None:
            indices=indices[[t.material_index in materials for t in skin.data.loop_triangles]]
        faces=pos[indices]
        tree=BVHTree.FromPolygons(pos.tolist(),indices.tolist(),all_triangles=True)
        pair=[]
        for value in (0,1):
            key.value=value
            bpy.context.view_layer.update()
            garment=sock.evaluated_get(graph)
            local=np.linalg.inv(np.asarray(garment.matrix_world)[:3,:3])
            pair.append(np.asarray([correction(Vector(p),tree,faces) for p in world_vertices(garment)],dtype=np.float32)@local.T)
        offsets.append(pair)
    key.value=orig
    add_lookup(scene,sock,np.array(offsets,dtype=np.float32))

