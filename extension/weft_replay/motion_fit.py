"""Prepared contact offsets and provably inactive dry contact around a fitted pose."""
from base64 import b85decode, b85encode
from hashlib import sha256
from io import BytesIO
from typing import NamedTuple
from zlib import adler32, crc32

import bpy
import numpy as np
from mathutils import Vector

from .contact_audit import collision_mesh, opposing_face
from .posed_contact import CLEARANCE, correction, local_vertices, world_vertices
from .skin_surface import SkinSurface


CACHE = 'weft_motion_fit'
SCHEMA = 5
_loaded = {}
_verified = {}
_topologies = {}
preparing = set()


class Samples(NamedTuple):
    margins: np.ndarray
    planes: np.ndarray
    projected: np.ndarray


def array_key(*arrays):
    digest = sha256()
    for array in arrays:
        actual = np.ascontiguousarray(array)
        digest.update(str(actual.shape).encode('ascii'))
        digest.update(actual.view(np.uint8))
    return digest.hexdigest()


def reset(*_):
    _loaded.clear()
    _verified.clear()
    _topologies.clear()
    preparing.clear()


def invalidate(source):
    _verified.pop(source.as_pointer(),None)
    if CACHE in source:
        del source[CACHE]


def invalidate_topology(mesh):
    _topologies.pop(mesh.as_pointer(),None)


def corners_of(mesh):
    corners=np.empty(len(mesh.loops),np.int32)
    if vector:=mesh.attributes.get('.corner_vert'):
        vector.data.foreach_get('value',corners)
    else:
        mesh.loops.foreach_get('vertex_index',corners)
    return corners


def skin_topology(skin,evaluated,graph,slot=-1):
    visible='show_render' if graph.mode=='RENDER' else 'show_viewport'
    # Smooth and Armature reposition vertices without changing faces or slots.
    stable=not any(getattr(m,visible) and m.type not in {'ARMATURE','SMOOTH'} for m in skin.modifiers)
    data=skin.data if stable else evaluated.data
    key=data.as_pointer()
    cached=_topologies.get(key) if stable else None
    if cached is not None and cached[0]==slot:
        return cached[1:]
    corners=corners_of(data)
    sizes=np.empty(len(data.polygons),np.int32)
    data.polygons.foreach_get('loop_total',sizes)
    materials=np.empty(len(data.polygons) if slot>=0 else 0,np.int32)
    if slot>=0:
        data.polygons.foreach_get('material_index',materials)
    digest=array_key(corners,sizes,materials,np.asarray([slot]))
    if stable:
        _topologies[key]=(slot,corners,digest)
    return corners,digest


def fitted_key(skin,slot,positions,transform,faces,graph):
    evaluated=skin.evaluated_get(graph)
    _,geometry=skin_topology(skin,evaluated,graph,slot)
    # Both checksums cover the float bits and topology. No tolerance or mesh
    # quantization can make an edited pose reuse the result of a different one.
    return geometry+coordinate_key(positions,transform,faces,local_vertices(evaluated.data),np.asarray(evaluated.matrix_world))


def coordinate_key(*arrays):
    crc,adler=0,1
    for array in arrays:
        buf=np.ascontiguousarray(array).view(np.uint8)
        crc=crc32(buf,crc)
        adler=adler32(buf,adler)
    return f'{crc:08x}{adler:08x}'


def payload(text):
    key=text.as_pointer()
    if key not in _loaded:
        encoded=''.join(text.as_string().splitlines())
        with np.load(BytesIO(b85decode(encoded)),allow_pickle=False) as archive:
            data={name:archive[name] for name in archive.files}
        _loaded[key]=data if (int(data['schema'])==SCHEMA and str(data['blender'])==bpy.app.version_string) else None
    return _loaded[key]


def prepared_fit(source,skin,slot,positions,transform,faces,graph):
    pointer=source.as_pointer()
    _verified.pop(pointer,None)
    text=source.get(CACHE)
    scene=graph.scene_eval
    if text is None or scene.frame_subframe:
        return None
    data=payload(text)
    if data is None:
        return None
    frame=scene.frame_current
    offset=frame-int(data['start'])
    if not 0<=offset<len(data['offsets']):
        return None
    if data['keys'][offset]!=fitted_key(skin,slot,positions,transform,faces,graph):
        return None
    samples=Samples(data['margins'][offset],data['planes'][offset],data['projected'][offset])
    _verified[pointer]=(frame,str(data['triangles']),data['rest_keys'][offset],samples)
    return data['offsets'][offset]


def samples_for(source,pose_key,triangle_key,rest_key):
    entry=_verified.get(source.as_pointer())
    if entry is not None and pose_key==(entry[0],0.) and triangle_key==entry[1] and rest_key==entry[2]:
        return entry[3]
    return None


def verified_rest_key(source):
    entry = _verified.get(source.as_pointer())
    return entry[2] if entry else None


def reference_collision(source,graph):
    """Use the same tessellation throughout a prepared dry motion session."""
    scene=graph.scene_eval
    current=_verified.get(source.as_pointer())
    if current is None or scene.frame_subframe or current[0]!=scene.frame_current:
        return None
    data=payload(source[CACHE])
    if 'collision_indices' in data:
        return data['collision_indices'],data['collision_polygons']
    return None


def sample_rest(points,tree):
    """Keep unchanged-point queries and bound when a nearby constraint is inactive."""
    margins=np.zeros(len(points),np.float32)
    planes=np.zeros((len(points),4),np.float32)
    # A negative depth marks a point whose exact live query is still required.
    projected=np.zeros((len(points),4),float)
    projected[:,0]=-1.
    extra=.00024
    for i, xyz in enumerate(points):
        point=Vector(xyz)
        hit,normal,_,distance=tree.find_nearest(point)
        gap=(point-hit).dot(normal)
        if opposing_face(point,tree,hit,normal,distance) is not None:
            continue
        if distance<=.004:
            planes[i,:3]=normal
            planes[i,3]=gap
        if gap>=CLEARANCE:
            projected[i,0]=0.
        else:
            depth=CLEARANCE-gap
            candidate=xyz+depth*np.asarray(normal)
            try:
                projected[i,1:]=checked_projection(point,candidate,tree,tree.positions)
                projected[i,0]=depth
            except RuntimeError:
                pass  # The live query retains the original unresolved-contact reporting.
        if gap<=CLEARANCE+.000002:
            continue
        # Moving by r can bring a new nearest face only from within d+2r.
        # For all such planes require clearance even after motion, and normals
        # close enough to each other that an opposing-face tie is impossible.
        radius=distance+extra
        safe=extra*.5
        for other,n,_,_ in tree.find_nearest_range(point,radius):
            if normal.dot(n)<.5:
                safe=0.
                break
            safe=min(safe,(point-other).dot(n)-CLEARANCE)
            if safe<=.000002:
                break
        margins[i]=max(0.,safe-.000002)
    return Samples(margins,planes,projected)


def checked_projection(original,candidate,tree,triangles):
    proposed=Vector(candidate)
    hit,normal,_,_=tree.find_nearest(proposed)
    if (proposed-hit).dot(normal)<CLEARANCE*.45:
        return np.asarray(original+correction(original,tree,triangles))
    return candidate


def store(source,**data):
    packed=BytesIO()
    np.savez_compressed(packed,**data)
    old=source.get(CACHE)
    text=bpy.data.texts.new('Weft • verified motion fit')
    encoded=b85encode(packed.getvalue()).decode('ascii')
    text.write('\n'.join(encoded[i:i+128] for i in range(0,len(encoded),128)))
    source[CACHE]=text
    if old is not None:
        _loaded.pop(old.as_pointer(),None)
        if not old.users:
            bpy.data.texts.remove(old)


def prepare(context,source,start,end):
    """Yield at every baked pose. Install the result only after the full range."""
    from .sock_length import OFFSET, fitted_inputs, refresh_contact

    scene=context.scene
    previous=scene.frame_current,scene.frame_subframe
    skin=source['weft_contact_skin']
    slot=source['weft_skin_slot']
    keys,offsets,rest,rest_keys=[],[],[],[]
    graph=context.evaluated_depsgraph_get()
    _,reference,reference_poly=collision_mesh(skin,slot,graph)
    reference_key=array_key(reference)
    pointer=source.as_pointer()
    preparing.add(pointer)
    try:
        for frame in range(start,end+1):
            scene.frame_set(frame)
            graph=context.evaluated_depsgraph_get()
            if refresh_contact(source,graph,exact=True):
                context.view_layer.update()
            graph=context.evaluated_depsgraph_get()
            positions,transform,faces=fitted_inputs(source,graph)
            keys.append(fitted_key(skin,slot,positions,transform,faces,graph))
            local=np.empty((len(positions),3),np.float32)
            source.data.attributes[OFFSET].data.foreach_get('vector',local.ravel())
            offsets.append(local)
            points=world_vertices(source.evaluated_get(graph))
            rest_keys.append(coordinate_key(points))
            body=world_vertices(skin.evaluated_get(graph))
            rest.append(sample_rest(points,SkinSurface(body,reference,points)))
            yield frame
    finally:
        preparing.discard(pointer)
        scene.frame_set(previous[0],subframe=previous[1])
    margins,planes,projected=zip(*rest)
    store(source,schema=SCHEMA,blender=bpy.app.version_string,start=start,keys=keys,
          triangles=reference_key,offsets=offsets,margins=margins,planes=planes,
          projected=projected,rest_keys=rest_keys,collision_indices=reference,collision_polygons=reference_poly)
    scene.frame_set(previous[0],subframe=previous[1])
