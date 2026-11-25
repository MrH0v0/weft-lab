"""Make an open, single-leg stocking that follows the evaluated skin mesh."""
from types import SimpleNamespace

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .contact_audit import collider, collision_mesh
from .live_membrane import knit_chart, unit
from .posed_contact import correction, local_vertices, world_vertices
from .semi_binding import leg_chain, materials
from .skin_surface import SkinSurface
from . import motion_fit


CHART = 'weft_cuff_chart'
SKIN_A = 'weft_skin_a'
SKIN_B = 'weft_skin_b'
MIX = 'weft_skin_fraction'
FITTED = 'weft_length_fitted'
OFFSET = 'weft_length_contact'
CLEARANCE = .00010
_fitted_contact = {}
CATEGORIES = ((0., '小腿'), (1/3, '过膝'), (2/3, '大腿中段'), (1., '大腿根'))


def _leg_points(scene, rig, side):
    if scene.weft_override and getattr(scene,'weft_'+side+'shin') and getattr(scene,'weft_'+side+'foot'):
        knee,ankle=[rig.data.bones.get(getattr(scene,'weft_'+side+name)) for name in ('shin','foot')]
        if not knee or not ankle:
            raise ValueError('找不到指定的膝骨或踝骨')
    else:
        knee,ankle=leg_chain(rig,side)
    if knee.parent is None or knee not in ankle.parent_recursive:
        raise ValueError('袜长需要连续的大腿、膝和踝骨')
    return np.array([rig.matrix_world @ bone.head_local for bone in (ankle,knee,knee.parent)],dtype=float)


def _flood(seed, adjacency, heights, level):
    found=np.zeros(len(heights),bool)
    found[seed]=True
    stack=[seed]
    while stack:
        for vertex in adjacency[stack.pop()]:
            if not found[vertex] and heights[vertex]<level:
                found[vertex]=True
                stack.append(vertex)
    return found


def _closed_cuff(triangles, accepted):
    """Count connections between intersection points keyed by original skin edges."""
    crossings=triangles[(accepted[triangles].sum(1)>0)&(accepted[triangles].sum(1)<3)]
    ring={}
    for face in crossings.tolist():
        ends=[]
        for a,b in zip(face,face[1:]+face[:1]):
            if accepted[a]!=accepted[b]:
                ends.append((min(a,b),max(a,b)))
        a,b=ends
        ring.setdefault(a,set()).add(b)
        ring.setdefault(b,set()).add(a)
    if len(ring)<3 or any(len(neighbors)!=2 for neighbors in ring.values()):
        return False
    first=next(iter(ring))
    visited={first}
    previous,current=first,next(iter(ring[first]))
    while current not in visited:
        visited.add(current)
        previous,current=current,next(iter(ring[current]-{previous}))
    return len(visited)==len(ring)


def _trim(points, triangles, source_ids, level, accepted, heights):
    selected=triangles[accepted[triangles].any(1)]
    vertex_map={}
    edges=[]
    coefficients=[]
    faces=[]
    for face in selected.tolist():
        polygon=[]
        for a,b in zip(face,face[1:]+face[:1]):
            if accepted[a]:
                key=(a,a)
                if key not in vertex_map:
                    vertex_map[key]=len(edges)
                    edges.append(key)
                    coefficients.append(0.)
                polygon.append(vertex_map[key])
            if accepted[a]!=accepted[b]:
                key=(min(a,b),max(a,b))
                if key not in vertex_map:
                    vertex_map[key]=len(edges)
                    edges.append(key)
                    from_h,to_h=heights[list(key)]
                    coefficients.append(float((level-from_h)/(to_h-from_h)))
                polygon.append(vertex_map[key])
        for i in range(1,len(polygon)-1):
            faces.append((polygon[0],polygon[i],polygon[i+1]))
    edges=np.asarray(edges,dtype=np.int32)
    blend=np.asarray(coefficients)
    start,end=points[edges].transpose(1,0,2)
    return start+(end-start)*blend[:,None],np.asarray(faces,dtype=np.int32),source_ids[edges],blend


def _cut_stocking(context, template, skin, slot, height):
    scene=context.scene
    rig=skin.find_armature()
    if rig is None or template.find_armature() not in (None,rig):
        raise ValueError('请使用同一角色骨架的皮肤和袜面')
    legs={side:_leg_points(scene,rig,side) for side in 'LR'}
    previous=rig.data.pose_position
    try:
        rig.data.pose_position='REST'
        context.view_layer.update()
        graph=context.evaluated_depsgraph_get()
        points,skin_faces,_=collision_mesh(skin,slot,graph)
        original=world_vertices(template.evaluated_get(graph))
    finally:
        rig.data.pose_position=previous
        context.view_layer.update()
    side=scene.weft_side
    if side=='AUTO':
        side=template.get('weft_leg') or min(legs,key=lambda side:np.linalg.norm(original.mean(0)-legs[side][:2].mean(0)))
    ankle,knee,hip=legs[side]
    other=legs['R' if side=='L' else 'L']
    up=unit(hip-ankle)
    axis=unit(knee-other[1]-up*np.dot(knee-other[1],up))
    basis=np.column_stack((axis,np.cross(up,axis),up))
    shin,thigh=(knee-ankle)@up,(hip-knee)@up
    if min(shin,thigh)<.08 or np.linalg.norm(axis)<.9:
        raise ValueError('腿骨链长度无效。此功能使用以米为单位的角色')
    # Weld coincident seams once, retaining indices into the skin's evaluated mesh.
    unique,source_ids,inverse=np.unique(np.round(points,7),axis=0,return_index=True,return_inverse=True)
    triangles=inverse.reshape(-1)[skin_faces]
    triangles=triangles[(triangles[:,0]!=triangles[:,1])&(triangles[:,1]!=triangles[:,2])&(triangles[:,0]!=triangles[:,2])]
    unique=points[source_ids].astype(float)
    heights=(unique-knee)@up
    start_height=-.25*shin
    max_height=.97*thigh
    triangles=triangles[(heights[triangles]<max_height).any(1)]
    used=np.unique(triangles)
    adjacency=[set() for _ in unique]
    for a,b,c in triangles.tolist():
        adjacency[a].update((b,c))
        adjacency[b].update((a,c))
        adjacency[c].update((a,b))
    separation=(unique-(ankle+other[0])*.5)@axis
    lower=used[heights[used]<start_height-.015]
    own=lower[separation[lower]>0]
    if not len(own):
        raise ValueError('没找到所选侧的小腿皮肤')
    seed=int(own[np.argmin(np.linalg.norm(unique[own]-(knee-up*shin*.55),axis=1))])
    elsewhere=lower[(separation[lower]<-.025)&(heights[lower]<-.60*shin)]
    rival=int(elsewhere[np.argmin(np.linalg.norm(unique[elsewhere]-other[0],axis=1))]) if len(elsewhere) else -1
    maximum=None
    # The highest single leg loop is limited by the skin edge or where the legs join.
    for candidate in np.arange(max_height,.66*thigh,-.004):
        selected=_flood(seed,adjacency,heights,candidate)
        if (rival<0 or not selected[rival]) and _closed_cuff(triangles,selected):
            maximum=float(candidate)
            break
    if maximum is None:
        raise ValueError('所选皮肤缺少大腿根处可闭合的单腿截面')
    levels=(start_height,.12*thigh,.50*thigh,maximum)
    cutoff=float(np.interp(height,[v for v,_ in CATEGORIES],levels))
    reference=_trim(unique,triangles,source_ids,maximum,selected,heights)
    if cutoff==maximum:
        positions,faces,skin_indices,fractions=reference
    else:
        selected=_flood(seed,adjacency,heights,cutoff)
        if not _closed_cuff(triangles,selected):
            raise ValueError('该高度找不到完整的袜口。请检查皮肤接缝或略调高度')
        positions,faces,skin_indices,fractions=_trim(unique,triangles,source_ids,cutoff,selected,heights)
    chart=knit_chart((positions-knee)@basis,reference=(reference[0]-knee)@basis)
    return positions,faces,skin_indices,fractions,chart,side,cutoff,len(points)


def _skin_modifier(skin):
    graph=bpy.data.node_groups.new('Weft • follow posed skin','GeometryNodeTree')
    interface=graph.interface
    interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes=graph.nodes
    link=graph.links.new
    source=nodes.new('GeometryNodeObjectInfo')
    source.transform_space='RELATIVE'
    source.inputs['Object'].default_value=skin
    source.inputs['As Instance'].default_value=False

    def attribute(name,dtype):
        node=nodes.new('GeometryNodeInputNamedAttribute')
        node.data_type=dtype
        node.inputs['Name'].default_value=name
        return node.outputs['Attribute']

    coords=nodes.new('GeometryNodeInputPosition')
    samples=[]
    for name in (SKIN_A,SKIN_B):
        node=nodes.new('GeometryNodeSampleIndex')
        node.data_type='FLOAT_VECTOR'
        node.domain='POINT'
        link(source.outputs['Geometry'],node.inputs['Geometry'])
        link(coords.outputs['Position'],node.inputs['Value'])
        link(attribute(name,'INT'),node.inputs['Index'])
        samples.append(node.outputs['Value'])
    blend=nodes.new('ShaderNodeMix')
    blend.data_type='VECTOR'
    link(attribute(MIX,'FLOAT'),blend.inputs[0])
    link(samples[0],blend.inputs[4])
    link(samples[1],blend.inputs[5])
    place=nodes.new('GeometryNodeSetPosition')
    link(nodes.new('NodeGroupInput').outputs['Geometry'],place.inputs['Geometry'])
    link(blend.outputs[1],place.inputs['Position'])
    position=coords.outputs['Position']
    offset=nodes.new('ShaderNodeVectorMath')
    offset.operation='SCALE'
    offset.inputs['Scale'].default_value=CLEARANCE
    link(nodes.new('GeometryNodeInputNormal').outputs['Normal'],offset.inputs[0])
    lift=nodes.new('GeometryNodeSetPosition')
    link(place.outputs['Geometry'],lift.inputs['Geometry'])
    link(offset.outputs['Vector'],lift.inputs['Offset'])
    cache=nodes.new('GeometryNodeStoreNamedAttribute')
    cache.data_type='FLOAT_VECTOR'
    cache.domain='POINT'
    cache.inputs['Name'].default_value=FITTED
    link(lift.outputs['Geometry'],cache.inputs['Geometry'])
    link(position,cache.inputs['Value'])
    correct=nodes.new('GeometryNodeSetPosition')
    link(cache.outputs['Geometry'],correct.inputs['Geometry'])
    link(attribute(OFFSET,'FLOAT_VECTOR'),correct.inputs['Offset'])
    link(correct.outputs['Geometry'],nodes.new('NodeGroupOutput').inputs['Geometry'])
    return graph


def _stock_mesh(positions,faces,skin_indices,fractions,chart,style):
    from .controls import appearance
    from .live_finish import TEMPLATE,apply_template,gauge_image
    from .live_view import fabric

    mesh=bpy.data.meshes.new('Weft • adjustable cuff')
    mesh.from_pydata(positions.tolist(),[],faces.tolist())
    for label,column in ((SKIN_A,0),(SKIN_B,1)):
        mesh.attributes.new(label,'INT','POINT').data.foreach_set('value',skin_indices[:,column].astype(np.int32))
    mesh.attributes.new(MIX,'FLOAT','POINT').data.foreach_set('value',fractions.astype(np.float32))
    mesh.attributes.new(OFFSET,'FLOAT_VECTOR','POINT')
    coord=np.column_stack((chart,np.zeros(len(chart)))).astype(np.float32)
    mesh.attributes.new(CHART,'FLOAT_VECTOR','POINT').data.foreach_set('vector',coord.ravel())
    face_uv=chart[faces].copy()
    face_uv[:,:,0]-=np.round(face_uv[:,:,0]-face_uv[:,:1,0])
    face_uv[:,:,0]*=144
    mesh.uv_layers.new(name='Knit').uv.foreach_set('vector',face_uv.astype(np.float32).ravel())
    mesh.polygons.foreach_set('use_smooth',np.ones(len(mesh.polygons),bool))
    palette,thickness=style.get('weft_dry_palette',0),style.get('weft_dry_thickness',.64)
    state=SimpleNamespace(uv=chart)
    if template:=style.get(TEMPLATE):
        state.first_row,state.loop_pixels,state.loop_image=gauge_image(chart)
        material=apply_template(template,state.loop_image,state.first_row,palette,thickness)
    else:
        material=fabric(state,palette,thickness)
    state.loop_pixels[:]=(1,1,0,.5)
    state.loop_image.pixels.foreach_set(state.loop_pixels.ravel())
    state.loop_image.update()
    state.loop_image.pack()
    appearance(material,style)
    mesh.materials.append(material)
    return mesh


def apply_length(context,template,skin,slot,height):
    scene=context.scene
    old=scene.weft_length_object
    if old is not None:
        template=old['weft_length_source']
        skin=old['weft_contact_skin']
        slot=old['weft_skin_slot']
    data=_cut_stocking(context,template,skin,slot,height)
    mesh=_stock_mesh(*data[:5],old or template)
    if old is None:
        stocking=bpy.data.objects.new(template.name+' • 长筒袜',mesh)
        group=_skin_modifier(skin)
        modifier=stocking.modifiers.new('Weft • follow posed skin','NODES')
        modifier.node_group=group
        scene.collection.objects.link(stocking)
        stocking['weft_length_source']=template
        stocking['weft_length_original_hidden']=template.hide_get()
        stocking['weft_length_original_render']=template.hide_render
        for key in ('weft_live_finish','weft_dry_palette','weft_dry_thickness','weft_visual_knit_scale',
                    'weft_recovery','weft_break_extension','weft_run','weft_wrinkles','weft_pretension',
                    'weft_fit','weft_idle','weft_tear_volume'):
            if key in template:
                stocking[key]=template[key]
        template.hide_set(True)
        template.hide_render=True
    else:
        stocking=old
        motion_fit.invalidate(stocking)
        previous=stocking.data
        stocking.data=mesh
        material_slot=stocking.material_slots[0]
        if material_slot.link=='OBJECT':
            # A palette/scale edit can leave an object-level material bound to
            # the old mesh's shorter gauge image.
            material_slot.material=None
            material_slot.link='DATA'
        if previous.users==0:
            bpy.data.meshes.remove(previous)
    stocking['weft_contact_skin']=skin
    stocking['weft_skin_slot']=slot
    stocking['weft_leg']=data[5]
    stocking['weft_cuff_above_knee_m']=data[6]
    stocking['weft_length_skin_vertices']=data[7]
    stocking['weft_cuff_height']=height
    scene.weft_length_object=stocking
    scene.weft_cuff_height=height
    context.view_layer.objects.active=stocking
    stocking.select_set(True)
    context.view_layer.update()
    if refresh_contact(stocking,context.evaluated_depsgraph_get()):
        context.view_layer.update()
    return stocking


def _keep_vertices_outside(points,indices,skin,faces):
    for index in indices:
        point=Vector(points[index])
        hit,normal,_,_=skin.find_nearest(point)
        if (point-hit).dot(normal)<0.:
            points[index]+=np.asarray(correction(point,skin,faces))


def _keep_faces_outside(points,triangles,skin,faces, *, sample_gap=-.00007):
    # The skin's evaluated quads can flip their triangulation during a bend.
    # Check the interiors of overlapping faces as well as the fitted vertices.
    samples=np.array([[1/3,1/3,1/3],[.5,.5,0.],[0.,.5,.5],[.5,0.,.5]])
    tri_list=triangles.tolist()
    for _ in range(6):
        clothing=BVHTree.FromPolygons(points.tolist(),tri_list,all_triangles=True)
        intersections=(skin.overlap_garment(clothing,points) if isinstance(skin,SkinSurface)
                       else clothing.overlap(skin))
        if not intersections:
            return
        touched=triangles[np.unique([cloth_face for cloth_face,_ in intersections])]
        probes=(samples@points[touched]).reshape(-1,3)
        shifts=np.zeros_like(points)
        counts=np.zeros(len(points),np.int32)
        for index,probe in enumerate(probes):
            point=Vector(probe)
            hit,normal,_,_=skin.find_nearest(point)
            if (point-hit).dot(normal)>=sample_gap:
                continue
            row,sample=divmod(index,len(samples))
            vertices=touched[row,samples[sample]>0.]
            shifts[vertices]+=np.asarray(correction(point,skin,faces))*1.15
            counts[vertices]+=1
        updated=np.flatnonzero(counts)
        if not len(updated):
            return
        points[updated]+=shifts[updated]/counts[updated,None]
        _keep_vertices_outside(points,updated,skin,faces)


@persistent
def clear_contact_cache(*_):
    _fitted_contact.clear()
    motion_fit.reset()


def fitted_collision(stocking,vertices,indices):
    fitted=_fitted_contact.get(stocking.as_pointer())
    if fitted is not None:
        _,tree,_,_,_=fitted
        if np.array_equal(tree.vertices,vertices) and np.array_equal(tree.indices,indices):
            return tree
    return None


def fitted_inputs(stocking,graph):
    evaluated=stocking.evaluated_get(graph)
    raw=evaluated.data.attributes.get(FITTED)
    if raw is None:
        return None
    positions=np.empty((len(raw.data),3),np.float32)
    raw.data.foreach_get('vector',positions.ravel())
    transform=np.asarray(evaluated.matrix_world)
    triangles=motion_fit.corners_of(stocking.data).reshape(-1,3)
    world=positions@transform[:3,:3].T+transform[:3,3]
    return world,transform,triangles


def prepared_rest(stocking, graph):
    """Read a verified corrected pose without retagging the hidden source mesh."""
    evaluated = stocking.evaluated_get(graph)
    fitted = evaluated.data.attributes.get(FITTED)
    current = evaluated.data.attributes.get(OFFSET)
    skin = stocking.get('weft_contact_skin')
    if fitted is None or current is None or skin is None:
        return None
    local = np.empty((len(fitted.data), 3), np.float32)
    old_offset = np.empty_like(local)
    fitted.data.foreach_get('vector', local.ravel())
    if len(current.data) != len(local):
        return None
    current.data.foreach_get('vector', old_offset.ravel())
    if not np.array_equal(local + old_offset, local_vertices(evaluated.data)):
        return None
    transform = np.asarray(evaluated.matrix_world)
    linear, translation = transform[:3, :3].T, transform[:3, 3]
    triangles = motion_fit.corners_of(stocking.data).reshape(-1, 3)
    offset = motion_fit.prepared_fit(stocking, skin, stocking['weft_skin_slot'],
                                     local @ linear + translation, transform, triangles, graph)
    if offset is None:
        return None
    corrected = local + offset
    posed = corrected.astype(float) if np.array_equal(transform, np.eye(4)) else corrected @ linear + translation
    info = motion_fit.verified_rest_key(stocking)
    return posed if motion_fit.coordinate_key(posed) == info else None


def refresh_contact(stocking,graph, *, exact=False):
    skin=stocking.get('weft_contact_skin')
    if skin is None or len(skin.evaluated_get(graph).data.vertices)!=stocking.get('weft_length_skin_vertices'):
        return False
    inputs=fitted_inputs(stocking,graph)
    attr=stocking.data.attributes.get(OFFSET)
    if inputs is None or attr is None or len(inputs[0])!=len(attr.data):
        return False
    previous=np.empty((len(attr.data),3),np.float32)
    attr.data.foreach_get('vector',previous.ravel())
    uncorrected,transform,triangles=inputs
    local=None if exact else motion_fit.prepared_fit(stocking,skin,stocking['weft_skin_slot'],*inputs,graph)
    if local is None:
        local=fit_contact(stocking,skin,graph,uncorrected,transform,triangles,exact=exact)
    if np.array_equal(previous,local):
        return False
    attr.data.foreach_set('vector',local.ravel())
    stocking.data.update()
    return True


def fit_contact(stocking,skin,graph,uncorrected,transform,triangles, *, exact=False):
    vertices,skin_indices,_=collision_mesh(skin,stocking['weft_skin_slot'],graph)
    pointer=stocking.as_pointer()
    fitted=_fitted_contact.get(pointer)
    if (not exact and fitted is not None and np.array_equal(fitted[0],uncorrected)
            and np.array_equal(fitted[1].vertices,vertices) and np.array_equal(fitted[1].indices,skin_indices)
            and np.array_equal(fitted[2],triangles) and np.array_equal(fitted[3],transform)):
        local=fitted[4]
    else:
        points=uncorrected.copy()
        tree=SkinSurface(vertices,skin_indices,points,exact=exact)
        _keep_vertices_outside(points,range(len(points)),tree,tree.positions)
        _keep_faces_outside(points,triangles,tree,tree.positions)
        displacement=points-uncorrected
        local=(displacement@np.linalg.inv(transform[:3,:3]).T).astype(np.float32)
        _fitted_contact[pointer]=(uncorrected,tree,triangles,transform,local)
    return local


@persistent
def update_contact(scene,graph):
    stocking=scene.weft_length_object
    if not stocking:
        return
    skin=stocking.get('weft_contact_skin')
    if skin is None:
        return
    rig=skin.find_armature()
    if any(item.id.original==skin.data for item in graph.updates):
        motion_fit.invalidate_topology(skin.data)
    sources=(skin,skin.data,rig,rig.data if rig else None)
    if (stocking.as_pointer() not in motion_fit.preparing
            and any(item.id.original in sources or (item.id.original==stocking and item.is_updated_transform)
                    for item in graph.updates)):
        refresh_contact(stocking,graph)


@persistent
def update_frame_contact(scene,graph):
    stocking=scene.weft_length_object
    if stocking and stocking.as_pointer() not in motion_fit.preparing:
        refresh_contact(stocking,graph)


def restore_length(scene):
    stocking=scene.weft_length_object
    _fitted_contact.pop(stocking.as_pointer(),None)
    template=stocking.get('weft_length_source')
    if template is not None:
        template.hide_set(stocking['weft_length_original_hidden'])
        template.hide_render=stocking['weft_length_original_render']
    stocking.hide_render=True
    stocking.hide_set(True)
    scene.weft_length_object=None
    return template


class WEFT_OT_apply_length(bpy.types.Operator):
    bl_idname='weft.apply_length'
    bl_label='应用袜口高度'
    bl_options={'REGISTER','UNDO'}
    height: bpy.props.FloatProperty(default=-1.,min=-1.,max=1.,options={'SKIP_SAVE'})

    @classmethod
    def poll(cls,context):
        from .controls import live_garment
        return live_garment(context.scene) is None and not context.scene.get('weft_check_running')

    def execute(self,context):
        from .controls import source_for
        scene=context.scene
        source=source_for(scene,context.active_object)
        if source is None or source.type!='MESH':
            self.report({'ERROR'},'请先选袜面')
            return {'CANCELLED'}
        skin=scene.weft_skin or source.get('weft_contact_skin')
        if skin is None or skin==source:
            self.report({'ERROR'},'请在绑定面板选独立的角色皮肤')
            return {'CANCELLED'}
        try:
            selection=materials(scene,skin) if scene.weft_skin else None
            slot=next(iter(selection)) if selection else source.get('weft_skin_slot',-1)
            apply_length(context,source,skin,slot,self.height if self.height>=0 else scene.weft_cuff_height)
        except ValueError as error:
            self.report({'ERROR'},str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


class WEFT_OT_restore_length(bpy.types.Operator):
    bl_idname='weft.restore_length'
    bl_label='切回原长度'
    bl_options={'REGISTER','UNDO'}

    @classmethod
    def poll(cls,context):
        from .controls import live_garment
        return (context.scene.weft_length_object is not None and live_garment(context.scene) is None
                and not context.scene.get('weft_check_running'))

    def execute(self,context):
        template=restore_length(context.scene)
        if template is not None:
            context.view_layer.objects.active=template
            template.select_set(True)
        return {'FINISHED'}
