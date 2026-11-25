"""Transfer cached material displacements to a garment in its own rest topology."""
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from .cache import assets,delta,frames,xyz

def coordinates(mesh):
    a=np.empty(3*len(mesh.vertices),np.float32);mesh.vertices.foreach_get("co",a);return a.reshape(-1,3)

def project_patch(offsets,scale,src_anchor,ref,ref_faces,tree):
    queries=offsets/scale+src_anchor
    ids=np.full(len(queries),-1,np.int32)
    barycentric=np.zeros((len(queries),3),np.float32)
    weights=np.zeros_like(barycentric)
    for i in np.flatnonzero(np.linalg.norm(queries-src_anchor,axis=1)<.103):
        hit,normal,face,distance=tree.find_nearest(queries[i],.032)
        if face is None:continue
        a,b,c=ref[ref_faces[face]]
        bary=np.linalg.lstsq(np.stack((b-a,c-a),axis=1),np.asarray(hit)-a,rcond=None)[0]
        falloff=np.clip((.103-np.linalg.norm(queries[i]-src_anchor))/.025,0,1)*np.clip((.032-distance)/.012,0,1)
        weights[i]=np.array([1-bary.sum(),*bary])*falloff
        barycentric[i]=[1-bary.sum(),*bary]
        ids[i]=face
    return ids,weights,barycentric

def bind(scene,target,anchor,rotation,ratio=1.0,*,knit_scale=None):
    if target.type!="MESH" or target.get("weft_role") or target.get("weft_bound"):
        raise ValueError("请选择独立的袜面网格")
    if knit_scale is not None and (not np.isfinite(knit_scale) or knit_scale<=0):
        raise ValueError("织纹尺寸倍率必须是正数")
    rest,faces=assets()["Stocking"][:2];ref=rest+delta(0)
    src_anchor=xyz(frames()[0]["anchor"]);ref_faces=faces[np.any(np.linalg.norm(ref[faces]-src_anchor,axis=2)<.122,axis=1)]
    tree=BVHTree.FromPolygons(ref.tolist(),ref_faces.tolist(),all_triangles=True)
    verts=coordinates(target.data);mat=np.asarray(target.matrix_world);rot=np.asarray(rotation)
    offsets=(verts@mat[:3,:3].T+mat[:3,3]-anchor)@rot
    ids,weights,bary=project_patch(offsets,ratio,src_anchor,ref,ref_faces,tree)
    if (ids>=0).sum()<3:raise ValueError("游标附近没有足够的袜面顶点，请对齐抓取位置、朝向与比例")
    # Non-destructive copy keeps the original modifiers, weights and shape keys.
    obj=target.copy();obj.data=target.data.copy();obj.name=target.name+" • Weft"
    obj['weft_binding_source']=target
    scene.collection.objects.link(obj)
    if obj.data.shape_keys is None:obj.shape_key_add(name="Basis")
    key=obj.shape_key_add(name="Weft displacement",from_mix=False);key.value=1
    obj["weft_bound"]=key.name
    for k in range(3):
        ac=obj.data.attributes.new("weft_node_"+str(k),"INT","POINT")
        ac.data.foreach_set("value",np.where(ids>=0,ref_faces[np.maximum(ids,0),k],-1).astype(np.int32))
    ac=obj.data.attributes.new("weft_weights","FLOAT_VECTOR","POINT");ac.data.foreach_set("vector",weights.ravel())
    obj["weft_matrix"]=(np.linalg.inv(mat[:3,:3])@rot*ratio).flatten().tolist()
    # Assign the instrumented material only to entire mapped faces inside the
    # baked chart. All other faces keep the target garment's material slots.
    source=next((o for o in scene.objects if o.get("weft_role")=="Stocking"),None)
    # The motion field uses shin length. The material chart has its own rest-space gauge.
    chart_ids=ids
    if knit_scale is not None and knit_scale!=ratio:
        chart_ids,_,bary=project_patch(offsets,knit_scale,src_anchor,ref,ref_faces,tree)
    obj["weft_knit_scale"]=ratio if knit_scale is None else knit_scale
    chart=assets()["Stocking"][2];mapped=chart_ids>=0;uv=np.zeros((len(ids),2),np.float32)
    uv[mapped]=np.einsum("ni,nij->nj",bary[mapped],chart[ref_faces[chart_ids[mapped]]])
    bounds=frames()[0]["bounds"]
    inside=mapped&(uv[:,0]>=bounds[0])&(uv[:,0]<=bounds[2])&(uv[:,1]>=bounds[1])&(uv[:,1]<=bounds[3])
    mesh=obj.data;corners=np.empty(len(mesh.loops),np.int32);mesh.loops.foreach_get("vertex_index",corners)
    starts=np.empty(len(mesh.polygons),np.int32);mesh.polygons.foreach_get("loop_start",starts)
    face_mask=np.logical_and.reduceat(inside[corners],starts)
    if face_mask.any():
        layer=mesh.uv_layers.get("Weft.Material") or mesh.uv_layers.new(name="Weft.Material")
        old_uv=np.empty((len(corners),2),np.float32);layer.uv.foreach_get("vector",old_uv.ravel())
        counts=np.diff(np.append(starts,len(corners)));use_uv=np.repeat(face_mask,counts)
        old_uv[use_uv]=uv[corners[use_uv]];layer.uv.foreach_set("vector",old_uv.ravel())
        slots=np.empty(len(starts),np.int32);mesh.polygons.foreach_get("material_index",slots)
        if not mesh.materials:mesh.materials.append(bpy.data.materials.new("Weft • unassigned surface"))
        from .scene import knit_material
        slots[face_mask]=len(mesh.materials);mesh.materials.append(source.data.materials[0] if source else knit_material())
        mesh.polygons.foreach_set("material_index",slots)
    obj["weft_knit_faces"]=int(face_mask.sum())
    return obj

def binding_state(obj):
    mesh=obj.data;n=len(mesh.vertices);ids=np.empty((3,n),np.int32);w=np.empty((n,3),np.float32)
    for k in range(3):mesh.attributes["weft_node_"+str(k)].data.foreach_get("value",ids[k])
    mesh.attributes["weft_weights"].data.foreach_get("vector",w.ravel())
    valid=ids[0]>=0
    rest=coordinates(mesh)
    nodes=ids[:,valid].T
    return valid,nodes,w[valid],rest[valid],rest.copy(),delta(0)[nodes]

def apply_bound(obj,state,current):
    valid,ids,w,rest,coords,initial=state
    d=np.einsum('ijk,ij->ik',current[ids]-initial,w)
    coords[valid]=rest+d@np.asarray(obj["weft_matrix"]).reshape(3,3).T
    key=obj.data.shape_keys.key_blocks[obj["weft_bound"]]
    key.data.foreach_set("co",coords.ravel());obj.data.update()

