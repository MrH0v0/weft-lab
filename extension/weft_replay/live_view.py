"""Blender viewport tool for repeated picks in a fixed or moving pose."""
import time
from pathlib import Path

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils as view
from .binding import coordinates
from .contact_audit import collider
from .dry_fabric import TOE_FINISH_ATTRIBUTE, controls as dry_appearance, patch_optical_density, shade_dry_fabric
from .fitted_chart import fitted_chart
from .fracture import FabricResponse
from .live_contact import LiveContact
from .live_finish import GAUGE, TEMPLATE, apply_template, gauge_image
from .sheer_optics import THREAD, apply_sheer_finish
from .live_rims import publish_warning_yarn, publish_yarn_ends, remove_yarn_ends
from .live_skin import publish_skin, remove_skin_display, restore_skin
from .live_membrane import Membrane, faces_meeting_box, knit_chart, refine_surface
from .semi_binding import viewport_ray
from .vroid_foot import close_toe_boundaries, foot_chart, is_vroid_foot, round_foot_surface, sew_foot_boundary
from .wet_controls import parameters
from .wet_display import WetDisplay, remove_wet_objects
from .wet_film import DT
from .wet_scene import WetSurface, tree_for
from . import continue_view

ACTIVE = {}


def refined_hit(sim, coarse_hit, origin, ray):
    # Splitting edges preserves the surface. Intersect just the triangles
    # near the original hit, before publishing the new mesh to the viewport.
    tolerance = 2e-6 * max(1.0, max(abs(value) for value in coarse_hit))
    faces = np.flatnonzero(faces_meeting_box(sim.pos, sim.faces, coarse_hit, tolerance))
    if not len(faces):
        return None
    vertices, indices = np.unique(sim.faces[faces], return_inverse=True)
    tree = BVHTree.FromPolygons(sim.pos[vertices].tolist(),
                                indices.reshape(-1, 3).tolist(), all_triangles=True)
    point, _, child, _ = tree.ray_cast(origin, ray)
    if child is None or (point - coarse_hit).length > tolerance:
        return None
    return point, int(faces[child])


def material_hit(sim, mesh, face, displayed_hit):
    if not sim.cracks:
        return displayed_hit
    vertices=mesh.polygons[face].vertices
    shown=np.array([mesh.vertices[i].co for i in vertices])
    ab=np.linalg.lstsq((shown[1:]-shown[0]).T,np.asarray(displayed_hit)-shown[0],rcond=None)[0]
    return Vector(np.r_[1-ab.sum(),ab]@sim.pos[sim.faces[face]])


def grip_marker(tool):
    if bpy.context.area!=tool.area:return
    tool.motion_draw_pending = False
    session=getattr(tool,'dry_input',None)
    if session is not None:session.waiting_for_draw=False
    if tool.await_motion_draw:
        tool.started=tool.last=time.perf_counter()
        tool.await_motion_draw=False
    sim=tool.sim
    tremor=sim.warning_offset() if sim.warning is not None else 0.
    if sim.pose is not None and sim.pose.marker is not None:
        ids,weights,face,skin_weights=sim.pose.marker
        dots=((weights@sim.pos[ids]+tremor,(.15,1.,.68,1),7),
              (skin_weights@sim.pose.triangles[face],(1.,.62,.13,1),4))
    elif sim.grip is not None:
        dots=((sim.bary@sim.pos[sim.grip]+tremor,(.15,1.,.68,1),7),)
    else:
        return
    theta=np.linspace(0,2*np.pi,33)
    circle=np.column_stack((np.cos(theta),np.sin(theta)))
    shader=gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()
    for pos,color,radius in dots:
        center=view.location_3d_to_region_2d(tool.region,tool.rv,Vector(pos))
        if center is not None:
            shader.uniform_float('color',color)
            batch_for_shader(shader,'LINE_STRIP',{'pos':np.array(center)+radius*circle}).draw(shader)


def fabric(sim, palette=0, thickness=.64, *, sheer=True):
    name = "Weft • dry stocking"
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.surface_render_method = 'BLENDED'
    mat.use_transparency_overlap = False
    p = mat.node_tree.nodes.get("Principled BSDF")
    uv = mat.node_tree.nodes.new("ShaderNodeTexCoord")
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.label = THREAD
    tex.image = bpy.data.images.load(str(Path(__file__).with_name('data')/'live-loop-profile.png'), check_existing=True)
    tex.image.colorspace_settings.name='Non-Color'
    tex.image.pack()
    # Open the loop head and narrow its neck according to local cloth strain.
    # The periodic offset vanishes at stitch boundaries, preserving interlock.
    nodes=mat.node_tree.nodes
    link=mat.node_tree.links.new
    split=nodes.new('ShaderNodeSeparateXYZ')
    link(uv.outputs['UV'],split.inputs[0])
    def math_op(op, a, b=None):
        node=nodes.new('ShaderNodeMath')
        node.operation=op
        for socket,value in zip(node.inputs,(a,b)):
            if isinstance(value,(float,int)):socket.default_value=value
            elif value is not None:link(value,socket)
        return node.outputs[0]
    x,y=split.outputs['X'],split.outputs['Y']
    sim.first_row,sim.loop_pixels,sim.loop_image=gauge_image(sim.uv)
    height=sim.loop_pixels.shape[0]
    lookup=nodes.new('ShaderNodeTexImage')
    lookup.label="Per-loop head, neck and shear"
    lookup.image=sim.loop_image
    lookup.interpolation='Linear'
    address=nodes.new('ShaderNodeCombineXYZ')
    link(math_op('DIVIDE',x,144),address.inputs[0])
    link(math_op('DIVIDE',math_op('SUBTRACT',y,sim.first_row),height),address.inputs[1])
    link(address.outputs[0],lookup.inputs[0])
    widths=nodes.new('ShaderNodeSeparateColor')
    link(lookup.outputs['Color'],widths.inputs[0])
    fx,fy=math_op('FRACT',x),math_op('FRACT',y)
    center=math_op('SUBTRACT',math_op('MULTIPLY',fx,2),1)
    wave=math_op('SINE',math_op('MULTIPLY',fy,np.pi))
    blend=math_op('MULTIPLY',wave,wave)
    head,neck,lean=widths.outputs[:3]
    width=math_op('ADD',neck,math_op('MULTIPLY',blend,math_op('SUBTRACT',head,neck)))
    # Taper to identity at shared row crossings. A monotone lateral map
    # preserves the order of the two yarn banks even at maximum neck cinch.
    edge=math_op('MINIMUM',1,math_op('MULTIPLY',18,math_op('MINIMUM',fy,math_op('SUBTRACT',1,fy))))
    width=math_op('ADD',1,math_op('MULTIPLY',edge,math_op('SUBTRACT',width,1)))
    denom=math_op('ADD',width,math_op('MULTIPLY',math_op('SUBTRACT',1,width),math_op('ABSOLUTE',center)))
    mapped=math_op('DIVIDE',center,denom)
    shift=math_op('MULTIPLY',math_op('MULTIPLY',lean,blend),math_op('SUBTRACT',1,math_op('MULTIPLY',mapped,mapped)))
    warped=nodes.new('ShaderNodeCombineXYZ')
    inside=math_op('ADD',.5,math_op('MULTIPLY',.5,math_op('SUBTRACT',mapped,shift)))
    link(math_op('ADD',math_op('FLOOR',x),inside),warped.inputs[0])
    link(y,warped.inputs[1])
    link(warped.outputs[0],tex.inputs['Vector'])
    if sheer:
        dry_appearance(mat,palette,thickness)
        apply_sheer_finish(mat)
    else:
        channels=nodes.new('ShaderNodeSeparateColor')
        link(tex.outputs['Color'],channels.inputs['Color'])
        shade_dry_fabric(mat,p,channels,lookup,head,neck,y,math_op,palette,thickness)
    return mat


def edge_keys(vertices):
    a,b=vertices.T
    return (np.minimum(a,b).astype(np.int64)<<32)|np.maximum(a,b)


def seed_edge_cache(sim,arrays):
    if arrays is None:return
    sim.bank_edge_cache=make_edge_cache(arrays)


def make_edge_cache(arrays):
    corners,edges=arrays
    keys=edge_keys(edges);order=np.argsort(keys)
    counts=np.bincount(corners.ravel(),minlength=len(edges))
    return corners,keys[order],order,{},counts


def update_bank_edges(mesh,sim,tris):
    """Only the loop incidences on a newly paid-out stitch have changed."""
    previous=getattr(sim,'published_triangles',None)
    if previous is None or sim.published_faces is not sim.faces or previous.shape!=tris.shape:
        return False
    if getattr(sim,'cracks',None):
        from .live_sections import bank_face_rows
        covered=getattr(sim,'changing_bank_rows',None)
        if covered is None or covered[0] is not sim.faces:
            covered=sim.changing_bank_rows=(sim.faces,bank_face_rows(sim))
        candidates=covered[1]
    else:candidates=getattr(sim,'bank_rows',None)
    if candidates is None:
        dirty=np.flatnonzero((tris!=previous).any(axis=1))
    else:
        dirty=candidates[(tris[candidates]!=previous[candidates]).any(axis=1)]
    if not len(dirty):return True
    updated=tris[dirty]
    if not np.array_equal(sim.uv[previous[dirty]],sim.uv[updated]):return False
    cached=getattr(sim,'bank_edge_cache',None)
    count=len(mesh.edges)
    if cached is None:
        corners=np.empty(tris.shape,dtype='i')
        mesh.attributes['.corner_edge'].data.foreach_get('value',corners.ravel())
        edges=np.empty((count,2),'i')
        mesh.edges.foreach_get('vertices',edges.ravel())
        keys=edge_keys(edges);order=np.argsort(keys)
        counts=np.bincount(corners.ravel(),minlength=count)
        cached=(corners,keys[order],order,{},counts)
    corners,old_keys,order,changed_keys,counts=cached
    a=updated.ravel();b=np.roll(updated,-1,axis=1).ravel()
    desired,inverse=np.unique(edge_keys(np.column_stack((a,b))),return_inverse=True)
    where=np.searchsorted(old_keys,desired).clip(0,len(old_keys)-1)
    slots=np.where(old_keys[where]==desired,order[where],-1)
    for j,key in enumerate(desired):slots[j]=changed_keys.get(int(key),slots[j])
    old,removed=np.unique(corners[dirty],return_counts=True)
    freed=np.setdiff1d(old[counts[old]==removed],slots[slots>=0],assume_unique=True)
    missing=np.flatnonzero(slots<0)
    if len(freed)>len(missing):return False
    extra=len(missing)-len(freed)
    destinations=np.r_[freed,np.arange(count,count+extra,dtype='i')]
    slots[missing]=destinations
    if extra:
        mesh.edges.add(extra)
        counts=np.pad(counts,(0,extra))
    edge_data=mesh.attributes['.edge_verts'].data
    for key,slot in zip(desired[missing],destinations):
        slot=int(slot)
        if slot<count:
            prior=edge_data[slot].value
            a,b=sorted(prior);changed_keys[(int(a)<<32)|int(b)]=-1
        key=int(key);edge_data[slot].value=(key>>32,key&0xffffffff)
        changed_keys[key]=slot
    previous_slots=corners[dirty].copy()
    next_slots=slots[inverse.reshape(-1,3)]
    np.add.at(counts,previous_slots.ravel(),-1)
    np.add.at(counts,next_slots.ravel(),1)
    cv,ce=mesh.attributes['.corner_vert'].data,mesh.attributes['.corner_edge'].data
    for row,col in np.argwhere(previous[dirty]!=updated):
        cv[3*int(dirty[row])+int(col)].value=int(updated[row,col])
    for row,col in np.argwhere(previous_slots!=next_slots):
        ce[3*int(dirty[row])+int(col)].value=int(next_slots[row,col])
    corners[dirty]=next_slots
    sim.bank_edge_cache=(corners,old_keys,order,changed_keys,counts)
    return True


def write_topology(mesh, sim):
    if getattr(sim,'bank_weave',None) is not None:
        from .yarn_chart import check_domain
        check_domain(sim,mesh)
    # A growing bank changes the smooth fans themselves. Blender evaluates
    # these tangents together with the new draw batch after the first cut.
    if sim.cracks and getattr(sim,'knit_shader',None):
        from .knit_tangent import finish
        finish(sim,mesh)
    tris=sim.display_faces()
    parts=getattr(sim,'sections',None);packet=getattr(sim,'draw_packet',None)
    if parts is not None and packet is not None and packet[0] is sim.faces and np.array_equal(tris,packet[2]):
        spare=parts.take(packet,-1)
        if spare is not None:
            previous=parts.owner.data;parts.owner.data=spare
            corners,keys,order,_,counts=packet[5]
            sim.bank_edge_cache=(corners.copy(),keys,order,{},counts.copy())
            write_positions(spare,sim,topology=True)
            update_loop_image(sim)
            sim.published_faces=sim.faces;sim.published_triangles=tris;sim.changed=False
            if getattr(sim,'bank_weave',None) is not None:
                from .yarn_chart import write
                write(spare,sim.bank_weave)
            if not previous.users:parts.retired.append(previous)
            parts.drop_spares()
            return
    if update_bank_edges(mesh,sim,tris):
        write_positions(mesh,sim,topology=True)
        mesh.update()
        if getattr(sim,'knit_shader',None):
            from .knit_tangent import initialize
            initialize(mesh,sim)
        update_loop_image(sim)
        sim.published_triangles=tris
        sim.changed=False
        return
    sim.bank_edge_cache=None
    mesh.clear_geometry()
    mesh.vertices.add(len(sim.pos))
    write_positions(mesh,sim,topology=True)
    mesh.loops.add(tris.size)
    corner_vertex=mesh.attributes.get('.corner_vert')
    if corner_vertex is None:
        mesh.loops.foreach_set('vertex_index',tris.ravel())
    else:
        corner_vertex.data.foreach_set('value',tris.ravel())
    mesh.polygons.add(len(tris))
    mesh.polygons.foreach_set('loop_start',np.arange(0,tris.size,3,dtype='i'))
    # In Blender's mesh representation the absence of sharp_face means smooth.
    sharp=mesh.attributes.get('sharp_face')
    if sharp is not None:sharp.data.foreach_set('value',np.zeros(len(tris),bool))
    from .contact_index import mesh_edges,assign_edges
    packet=getattr(sim,'draw_packet',None)
    if packet is not None and packet[0] is sim.faces and np.array_equal(tris,packet[2]) and packet[4] is not None:
        assign_edges(mesh,packet[4])
        corners,keys,order,_,counts=packet[5]
        sim.bank_edge_cache=(corners.copy(),keys,order,{},counts.copy())
        uv=packet[3]
    else:
        seed_edge_cache(sim,mesh_edges(mesh,tris,len(sim.pos)))
        uv=sim.uv[tris].copy()
        uv[:,:,0]-=np.round(uv[:,:,0]-uv[:,:1,0])
        uv[:,:,0]*=144.
    layer=mesh.uv_layers.get('Knit') or mesh.uv_layers.new(name='Knit')
    layer.uv.foreach_set('vector',uv.astype('f',copy=False).ravel())
    if hasattr(sim, "toe_finish"):
        direction, start, end = sim.toe_finish[:3], sim.toe_finish[3], sim.toe_finish[4]
        ramp = np.clip((sim.rest @ direction-start)/(end-start), 0, 1)
        finish = mesh.attributes.get(TOE_FINISH_ATTRIBUTE)
        if finish is None:
            finish = mesh.attributes.new(TOE_FINISH_ATTRIBUTE, "FLOAT", "POINT")
        finish.data.foreach_set("value", (ramp**2*(3-2*ramp)).astype("f"))
    if getattr(sim,'bank_weave',None) is not None:
        from .yarn_chart import write
        write(mesh,sim.bank_weave)
    mesh.update()
    from .knit_tangent import initialize
    initialize(mesh,sim)
    update_loop_image(sim)
    sim.published_faces=sim.faces
    sim.published_triangles=tris
    sim.changed=False


def write_positions(mesh, sim, *, freeze_rim=False,topology=False,dirty=None):
    positions = sim.display_positions()
    if freeze_rim:
        sim.pose.constrain_frozen_rim(sim.pos, positions, sim.display_faces())
    coords=positions.astype('f',copy=False)
    mesh.attributes['position'].data.foreach_set('vector',coords.ravel())
    if getattr(sim,'sections',None) is not None:
        sim.sections.update(sim,coords,dirty)
    if not topology and getattr(sim,'knit_shader',None):
        from .knit_tangent import update
        update(mesh,sim,coords)


def update_loop_image(sim):
    if not hasattr(sim,'loop_image'):return
    sim.loop_pixels[:]=(1,1,0,.5)
    if sim.patch is not None:
        loops=sim.patch.loops
        x,y=loops.cells.T
        density = patch_optical_density(sim)
        sim.loop_pixels[y-sim.first_row,x]=np.column_stack((loops.head,loops.neck,loops.lean,.5*density))
    if sim.cracks:
        # Retracted loops gather yarn around the open banks. Write in material
        # coordinates so an old rim retains its coverage on the next grab.
        failed=sim.failed_cells
        rim=np.roll(failed,1,axis=1)|np.roll(failed,-1,axis=1)
        rim[1:]|=failed[:-1]
        rim[:-1]|=failed[1:]
        shoulder=np.roll(rim,1,axis=1)|np.roll(rim,-1,axis=1)
        shoulder[1:]|=rim[:-1]
        shoulder[:-1]|=rim[1:]
        alpha=sim.loop_pixels[:,:,3]
        alpha[shoulder]=np.maximum(alpha[shoulder],.64)
        alpha[rim]=np.maximum(alpha[rim],.81)
        # A failed cell has loose shoulders along the banks, but no longer
        # retains the coverage of a complete interlocked loop.
        directional=hasattr(sim,'run_image')
        # In the directional finish the yarn-free centre is open geometry.
        # Material covering the failed chart cell survives only on its rolled lips.
        alpha[failed]=.67 if directional else .20
        if sim.loose_yarn is not None and not directional:
            loosened = (sim.loose_yarn > 0.) & ~shoulder & ~failed
            alpha[loosened] = np.minimum(alpha[loosened],.50-sim.loose_yarn[loosened])
    sim.preview_count=0
    if sim.dry and sim.warning is not None:
        _, cells, exposed = sim.loaded_stitches()
        index=(cells[:,1]-sim.first_row,cells[:,0])
        stock=sim.loop_pixels[:,:,3]
        stock[index]=np.minimum(stock[index],.50-1.10*exposed)
        sim.preview_count=int(np.count_nonzero(exposed>.12))

    if sim.dry and getattr(sim,'pose',None) is None:
        # foreach_set marks the native float image dirty. Keep the live atlas
        # current without asking Blender to rebuild its display buffer twice.
        last=getattr(sim,'published_loop_pixels',None)
        if last is None or not np.array_equal(last,sim.loop_pixels):
            from .atlas_transfer import publish
            publish(sim,'loop',sim.loop_pixels)
            if last is None:sim.published_loop_pixels=sim.loop_pixels.copy()
            else:np.copyto(last,sim.loop_pixels)
    else:
        sim.loop_image.pixels.foreach_set(sim.loop_pixels.ravel())
        sim.loop_image.update()
    if sim.dry:
        from .run_finish import update_run
        update_run(sim)


def fabric_response(source):
    defaults=FabricResponse()
    return FabricResponse(
        recovery=float(np.clip(source.get('weft_recovery',defaults.recovery),.3,2.)),
        break_extension=float(np.clip(source.get('weft_break_extension',defaults.break_extension),.25,2.5)),
        run=float(np.clip(source.get('weft_run',defaults.run),0.,1.)),
        pretension=float(np.clip(source.get('weft_pretension',defaults.pretension),0.,.24)),
        fit=float(np.clip(source.get('weft_fit',defaults.fit),0.,1.)),
        wrinkles=bool(source.get('weft_wrinkles',defaults.wrinkles)))


def bind_skin_rays(sim,skin_rays):
    """Seat the opened cloth against the current curved skin collider."""
    def clearance(positions,normals):
        distances=skin_rays(positions+.014*normals,-normals,.025)
        return np.where(np.isfinite(distances),distances-.014,.025)
    sim.clearance=clearance
    def project_skin(points, normals, clearance=.00065):
        # Follow the actual curved contact surface after a tangential drag.
        # Approach from the garment side so coincident skin faces with
        # conflicting winding cannot flip the contact direction.
        reach=.012
        limit=reach+clearance
        distances=skin_rays(points+reach*normals,-normals,limit)
        # A Blender ray miss supplies None, represented as NaN in the array.
        hit=np.isfinite(distances)
        # Match the scalar path's rounding before multiplying by the normal.
        offsets=(limit-distances[hit]).astype(normals.dtype,copy=False)
        points[hit]+=offsets[:,None]*normals[hit]
        return points

    sim.project_skin=project_skin
    def seat_opening(neutral,points,normals):
        reach=.019
        def distances(positions):
            return skin_rays(positions+reach*normals,-normals,.043)
        original=distances(neutral)
        current=distances(points)
        hit=np.isfinite(original)&np.isfinite(current)
        # Follow the curvature without removing the original drape.
        # Keep a small stand-off for the rolled yarn along a cut edge.
        correction=(np.maximum(original[hit],reach+.00048)-current[hit]).clip(-.008,.008)
        points[hit]+=correction[:,None]*normals[hit]
        return points
    sim.seat_opening=seat_opening


def prepare(ctx, source, motion=False, *, prepared_motion=True, dry=True):
    dg=ctx.evaluated_depsgraph_get()
    from .sock_length import CHART,refresh_contact
    if 'weft_length_skin_vertices' in source and refresh_contact(source,dg):
        ctx.view_layer.update()
    evaluated=source.evaluated_get(dg)
    mesh=evaluated.data
    if not len(mesh.polygons):raise ValueError('袜面需要至少一个有效面')
    pose=coordinates(mesh)
    mat=np.array(evaluated.matrix_world)
    pose=pose@mat[:3,:3].T+mat[:3,3]
    mesh.calc_loop_triangles()
    tri=np.empty((len(mesh.loop_triangles),3),np.int32)
    mesh.loop_triangles.foreach_get('vertices',tri.ravel())
    # UV seams on the avatar often contain several copies of a vertex.
    # Weld exact posed duplicates, so a cut can traverse those seams.
    _,idx,inv=np.unique(np.round(pose,7),axis=0,return_index=True,return_inverse=True)
    tri=inv.reshape(-1)[tri]
    tri=tri[(tri[:,0]!=tri[:,1])&(tri[:,0]!=tri[:,2])&(tri[:,1]!=tri[:,2])]
    if not len(tri):raise ValueError('袜面没有可抓取的三角面')
    chart=coordinates(source.data) if len(source.data.vertices)==len(pose) else pose
    rest=chart[idx]
    points=pose[idx]
    fit=len(source.data.vertices)==len(pose) and is_vroid_foot(source,rest)
    if fit:
        points,tri,rest,idx=sew_foot_boundary(points,tri,rest,idx)
        tri=close_toe_boundaries(tri,rest)
    explicit=source.data.attributes.get(CHART)
    if explicit is not None:
        values=np.empty((len(source.data.vertices),3),np.float32)
        explicit.data.foreach_get('vector',values.ravel())
        uv=values[idx,:2].astype(float)
    else:
        uv=fitted_chart(source,rest) if source.get('weft_candidate_ankle') is not None else knit_chart(rest)
    skin=source.get('weft_contact_skin')
    if skin is not None:
        if explicit is not None and len(skin.evaluated_get(dg).data.vertices)!=source.get('weft_length_skin_vertices'):
            raise ValueError('皮肤拓扑已变更，请重新应用袜口高度')
        contact=collider(skin,source.get('weft_skin_slot',-1),dg)
        tree=contact[0]
    if fit:
        uv=foot_chart(rest,uv)
        if skin is not None:
            points=round_foot_surface(points,tri,rest,tree)
    # Both operators load the same knitted fabric. Only skin contact and the water clock differ.
    sim=Membrane(points,tri.astype(np.int32),uv,fabric_response(source),dry=True)
    sim.pose=(LiveContact(source,skin,source.get('weft_skin_slot',-1),dg,idx,points,tri,
                          prepared_motion=prepared_motion, wet_contact=not dry) if motion else None)
    if source.get("weft_toe_finish") is not None:
        sim.toe_finish = np.array(source["weft_toe_finish"])
    if sim.pose is not None:
        sim.skin_planes=sim.pose.skin_planes
        s=ctx.scene
        sim.pose.update_pose(sim,dg,(s.frame_current,s.frame_subframe))
    elif skin is not None:
        sim.monitor_contact=contact
        from .contact_index import ContactIndex
        skin_points=contact[1].reshape(-1,3)
        index=ContactIndex.create(skin_points,np.arange(len(skin_points),dtype='i').reshape(-1,3))
        def skin_rays(origins,directions,limit):
            if index is not None:
                return index.first_hits(origins,directions,limit)
            return np.array([tree.ray_cast(o,d,limit)[3] for o,d in zip(origins.tolist(),directions.tolist())],dtype=float)
        if index is not None:
            sim.held_face_lifts=index.wet_face_lifts
            sim.async_neutral=dry
        if index is not None and dry:sim.static_skin=index
        def skin_planes(rest,normal):
            if index is not None and dry:
                origin,normals,unknown=index.patch_planes(rest,normal)
            else:
                origin=rest.copy();normals=np.zeros_like(rest);unknown=range(len(rest))
            for i in unknown:
                p=rest[i]
                q,n,_,d=tree.find_nearest(p)
                if d>.004 or np.dot(n,normal[i])<.65:continue
                if any(n.dot(m)<-.8 for _,m,_,_ in tree.find_nearest_range(p,d+.000002)):continue
                normals[i]=n
                gap=max(0,np.dot(p-np.asarray(q),n)-.000075)
                origin[i]=p-gap*normals[i]
            return origin,normals
        sim.skin_planes=skin_planes

        bind_skin_rays(sim,skin_rays)
    palette, thickness = source.get('weft_dry_palette',0), source.get('weft_dry_thickness',.64)
    template = source.get(TEMPLATE)
    if template is None:
        material = fabric(sim, palette, thickness)
    else:
        sim.first_row,sim.loop_pixels,sim.loop_image=gauge_image(sim.uv)
        material = apply_template(template, sim.loop_image, sim.first_row, palette, thickness)
    from .run_finish import bind_run
    bind_run(material,sim)
    from .controls import appearance
    appearance(material,source)
    data=bpy.data.meshes.new("Weft • live garment")
    data.materials.append(material)
    write_topology(data,sim)
    obj=bpy.data.objects.new(source.name+" • Live",data)
    ctx.scene.collection.objects.link(obj)
    obj["weft_live_source"]=source
    obj["weft_live_frame"]=ctx.scene.frame_current
    if motion:
        obj["weft_live_motion"]=True
    obj["weft_source_render_hidden"]=source.hide_render
    source.hide_set(True);source.hide_render=True
    if dry and sim.pose is None:
        from .live_sections import Sections
        sim.sections=Sections(obj,sim)
        from .live_rims import prepare_strands
        prepare_strands(obj,sim)
    return obj,sim


class WEFT_OT_live(bpy.types.Operator):
    bl_idname="weft.live_grab"
    bl_label="开始自由抓取与撕裂"
    bl_options={'BLOCKING'}
    motion: bpy.props.BoolProperty(default=False, options={'SKIP_SAVE'})
    wet_mode: bpy.props.BoolProperty(default=False, options={'SKIP_SAVE'})
    resume: bpy.props.BoolProperty(default=False, options={'SKIP_SAVE'})

    def invoke(self,ctx,event):
        if ctx.area.type!='VIEW_3D' or ctx.screen.is_animation_playing:return {'CANCELLED'}
        scene=ctx.scene
        if scene.get('weft_check_running'):
            self.report({'WARNING'},'请先完成正在运行的袜面准备或检查');return {'CANCELLED'}
        if scene.as_pointer() in ACTIVE:return {'CANCELLED'}
        frozen=next((o for o in scene.objects if o.get('weft_live_source')),None)
        if frozen and (self.motion or self.wet_mode):
            self.report({'WARNING'},'先恢复原袜，再开启新一轮操作');return {'CANCELLED'}
        from .controls import prepare_audio, source_for
        source=source_for(scene,ctx.active_object)
        if not source or source.type!='MESH' or (not frozen and not source.visible_get()):
            self.report({'WARNING'},'请选择可见的袜面网格');return {'CANCELLED'}
        self.area=ctx.area
        self.window=ctx.window
        self.scene=scene
        self.region=next(r for r in self.area.regions if r.type=='WINDOW')
        self.rv=self.area.spaces.active.region_3d
        speed=scene.weft_motion_speed if self.motion or (self.wet_mode and scene.weft_wet.moving) else 0.
        self.continued=bool(frozen)
        try:
            if frozen:
                saved=continue_view.snapshot_data(frozen)
                self.sim=continue_view.read_sheet(frozen,saved['fracture'],fabric_response(source))
                self.obj=frozen
                scene.frame_set(saved['pose_frame'],subframe=saved.get('pose_subframe',0.))
                skin=source.get('weft_contact_skin')
                if skin is not None:
                    body=collider(skin,source.get('weft_skin_slot',-1),ctx.evaluated_depsgraph_get())
                    continue_view.contact(self.sim,body)
                from .seam_gap import restore_saved
                repaired=restore_saved(self.sim)
                if repaired:
                    frozen.data.vertices.foreach_set('co',self.sim.pos.astype('f').ravel())
                    frozen.data.update()
            else:
                self.obj,self.sim=prepare(ctx,source,motion=speed>0 or self.wet_mode,
                                          prepared_motion=not self.wet_mode,dry=not self.wet_mode)
        except ValueError as err:
            self.report({'WARNING'},str(err));return {'CANCELLED'}
        if frozen:
            from . import bank_finish
            bank_finish.schedule(self,ctx.evaluated_depsgraph_get(),immediate=True)
        self.sim.warn_before_break=True
        self.sim.fixed_wet_pose=self.wet_mode and speed==0
        self.sim.wet_fabric_contact=bool(self.wet_mode)
        if self.wet_mode:
            bind_skin_rays(self.sim,self.sim.pose.wet_rays)
        self.last=time.perf_counter()
        self.started=self.last
        self.start_pose=(scene.frame_current, scene.frame_subframe)
        self.start_frame=scene.frame_current+scene.frame_subframe
        self.speed=speed
        self.await_motion_draw=bool(speed and not self.wet_mode)
        self.motion_draw_pending = False
        self.pending_drag=[]
        self.last_drag_event=0.
        self.dense_drag_until=0.
        self.obj['weft_live_motion']=bool(speed)
        self.drag_xy=None
        self.last_input=None
        self.refined_unpublished=False
        self.advancing=False
        self.wet=None
        self.wet_step=0
        self.wet_pointer_changed=False
        self.monitor=None
        self.monitor_pause=None
        self.monitor_released=False
        self.monitor_skip_release=False
        self.audio_cursor=len(self.sim.fracture_events)
        prepare_audio(self.obj)
        self.timer=ctx.window_manager.event_timer_add(DT if self.wet_mode else 1/(45 if speed else 90),window=ctx.window)
        self.marker=bpy.types.SpaceView3D.draw_handler_add(grip_marker,(self,),'WINDOW','POST_PIXEL')
        ACTIVE[scene.as_pointer()]=self
        self.area.header_text_set("湿态    W 加水 · Shift + W 固定落点 · 左键抓取 · Enter 结束" if self.wet_mode else
                                  "自由抓取    左键按住并拖动 · 松开后换抓点    中键旋转    Enter / Esc 留下当前袜面")
        ctx.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def committed_pose(self):
        return self.sim.pose.pose_key if self.sim.pose is not None else self.start_pose

    def pin_committed_pose(self):
        frame, subframe = self.committed_pose()
        scene = self.scene
        if (scene.frame_current, scene.frame_subframe) != (frame, subframe):
            scene.frame_set(frame, subframe=subframe)

    def motion_pose_due(self, scene, now):
        elapsed=self.wet_step*DT if getattr(self,'wet_mode',False) else now-self.started
        if self.await_motion_draw:
            elapsed=0.
        advance=int(elapsed*self.speed*scene.render.fps/scene.render.fps_base)
        frame=min(max(self.start_frame,scene.frame_end),self.start_frame+advance)
        whole=int(frame)
        return whole, frame-whole

    def sync_pose(self, ctx, now):
        s=ctx.scene
        # Update the pose once per timeline frame. The target and collision use this sample.
        whole,subframe=self.motion_pose_due(s,now)
        pose = self.sim.pose
        source = pose.source
        if (self.speed and not self.wet_mode and pose.prepared_motion and not subframe
                and 'weft_motion_fit' in source and (whole, subframe) != (s.frame_current,s.frame_subframe)):
            from . import motion_fit, sock_length
            pointer = source.as_pointer()
            motion_fit.preparing.add(pointer)
            try:
                s.frame_set(whole,subframe=subframe)
                graph = ctx.evaluated_depsgraph_get()
                ready = sock_length.prepared_rest(source, graph)
                if ready is None and sock_length.refresh_contact(source, graph):
                    ctx.view_layer.update()
                    graph = ctx.evaluated_depsgraph_get()
                elif ready is not None and not self.obj.get('weft_source_fit_deferred'):
                    self.obj['weft_source_fit_deferred'] = True
                return pose.update_pose(self.sim, graph, (whole, subframe), ready)
            finally:
                motion_fit.preparing.discard(pointer)
        if (whole,subframe)!=(s.frame_current,s.frame_subframe):
            s.frame_set(whole,subframe=subframe)
        if (whole,subframe)!=self.sim.pose.pose_key:
            return self.sim.pose.update_pose(self.sim,ctx.evaluated_depsgraph_get(),(whole,subframe))
        return False

    def target_at(self, xy):
        sim=self.sim
        if self.speed and not self.wet_mode and sim.grip is not None:
            # Follow the current material depth without removing lateral pull.
            # Intersect the camera plane directly for wide orthographic clips.
            reference=Vector(self.posed_grip_center())+self.grab_view_offset
            origin,ray=self.pick_ray(xy)
            depth_axis=self.rv.view_matrix.to_3x3().row[2]
            plane=origin+ray*((reference-origin).dot(depth_axis)/ray.dot(depth_axis))
        else:
            reference=self.start
            pv=self.rv.perspective_matrix
            clip=pv@reference.to_4d()
            ndc=Vector((2*xy[0]/self.region.width-1,2*xy[1]/self.region.height-1,clip.z/clip.w,1))
            world=pv.inverted()@ndc
            plane=world.xyz/world.w
            ray=view.region_2d_to_vector_3d(self.region,self.rv,xy)
        stroke=plane-reference
        offset=getattr(self,'grab_offset',0.)
        if getattr(self,'wet',None) is not None:
            # A wet grip takes up slack against the attached film before lifting.
            take_up=min(1.,stroke.length/.014)
            lift=min(.024,.46*stroke.length)*(.55+.45*take_up*take_up*(3.-2.*take_up))
            return np.asarray(plane-ray*lift)+offset
        # Pulling toward the outside of a side-facing surface should retain
        # that lift in its silhouette, even when camera depth alone cannot.
        length=stroke.length
        # A new finger grip first takes up the slack; the fully tensioned
        # yarn keeps its lateral travel. Existing banks retain their grip depth.
        if sim.cracks or (getattr(sim,'pose',None) is not None and not getattr(sim,'fixed_wet_pose',False)):
            take_up=1.
        else:
            taut=np.clip((length-.001)/.022,0.,1.)
            take_up=.38+.62*taut*taut*(3.-2.*taut)
        normal_lift=take_up*min(.022,.75*max(0.,np.asarray(stroke)@sim.normal))
        return np.asarray(plane-ray*(take_up*min(.032,.71*length)))+normal_lift*sim.normal+offset

    def posed_grip_center(self):
        return self.sim.grip_base()

    def held_cursor(self):
        if self.sim.grip is None:
            return None
        if self.speed and not self.wet_mode and self.drag_xy is None:
            return self.last_pick[0]
        return self.drag_xy

    def pick_ray(self, xy):
        return viewport_ray(self.region,self.rv,xy)

    def pick_visible(self, ctx, origin, ray):
        graph=ctx.evaluated_depsgraph_get()
        hit,point,_,face,obj,_=ctx.scene.ray_cast(graph,origin,ray)
        sections=getattr(self.sim,'sections',None)
        if hit and sections is not None:
            section=sections.sections.get(obj.get('weft_courses'))
            if section is not None and obj.original==section.obj:
                return True,point,int(section.rows[face]),self.obj
        if hit and (obj.get('weft_wet_owner')==self.obj or
                    (obj.parent==self.obj and obj.get('weft_yarn_ends'))):
            hit,point,_,face=self.obj.ray_cast(origin,ray,depsgraph=graph)
            obj=self.obj if hit else None
        return hit,point,face,obj

    def pointer_in_view(self, xy):
        x, y = xy
        if not (0 <= x < self.region.width and 0 <= y < self.region.height):
            return False
        x, y = x+self.region.x, y+self.region.y
        # Blender overlays sidebars and headers on the WINDOW region.
        return not any(r.type != 'WINDOW' and r.width > 1 and r.height > 1
                       and r.x <= x < r.x+r.width and r.y <= y < r.y+r.height
                       for r in self.area.regions)

    def tick(self, target=None, now=None, *, pointer_changed=False, pose_changed=False, advance_film=True):
        if getattr(self,'continued',False):
            return continue_view.tick(self,target,now)
        now=time.perf_counter() if now is None else now
        if (self.sim.grip is not None and self.sim.pose is None and self.sim.dry and not self.monitor
                and getattr(self.sim,'static_skin',None) is not None
                and (target is not None or getattr(self,'dry_input',None))):
            from .dry_input import DryInput
            if getattr(self,'dry_input',None) is None:self.dry_input=DryInput(self,now)
            self.dry_input.tick(target,now);self.last=now
            return
        if getattr(self.sim,'neutral_job',None) is not None:
            from .neutral_worker import collect
            collect(self.sim)
        if target is not None and self.sim.grip is not None:
            target=self.sim.limit_target(target)
        if self.sim.pose is not None:
            wet_mode=getattr(self,'wet_mode',False)
            wet=self.wet if wet_mode else None
            if wet is not None:wet.prepare_contact()
            dt=min(max(now-self.last,.0001),.1)
            if wet_mode:
                dt=DT if advance_film else min(dt,DT)
            elapsed=self.wet_step*DT if wet_mode else now-self.started
            changed=self.sim.pose.step(self.sim,dt,target,elapsed,driven_input=pointer_changed,wet=wet,
                                       fabric_dt=max(0.,now-self.last) if wet_mode else dt)
            if pose_changed or changed is not False or self.sim.changed or self.refined_unpublished:
                if wet_mode:
                    self.sim.pose.seat_wet_grip(self.sim)
                self.publish_positions()
                if self.speed and not wet_mode:
                    self.motion_draw_pending = True
            if wet is not None:
                if advance_film:
                    wet.advance()
                else:
                    # Rebind the film to the final cloth shape on the same liquid frame.
                    wet.rebind()
                    wet.remove_failed_sites()
                    wet.prepare_probe_depth()
                self.wet_display.publish()
            self.last=now
            self.capture_monitor()
            return
        if target is None and self.sim.grip is not None:
            # Under a still hand the yarn can slide or fail under sustained load.
            self.sim.hold(now-self.last)
            if self.sim.changed:
                self.publish_positions()
            else:
                if self.sim.warning_dirty:
                    write_positions(self.obj.data,self.sim)
                    update_loop_image(self.sim)
                    publish_warning_yarn(self.obj,self.sim)
                    self.sim.warning_dirty=False
                    self.region.tag_redraw()
                elif self.sim.held_optics_changed:
                    update_loop_image(self.sim)
                    self.region.tag_redraw()
            self.last=now
            self.capture_monitor()
            return
        if target is None:
            from .controls import idle
            if idle(self,now):
                self.last=now
                return
        if self.sim.grip is not None or self.sim.moving:
            self.sim.step(now-self.last,target)
            self.publish_positions()
            self.capture_monitor()
        self.last=now

    def capture_monitor(self):
        monitor=getattr(self,'monitor',None)
        if monitor is not None:
            monitor.capture(getattr(self,'wet',None))
            if self.wet_mode:
                from .monitor_ui import refresh_sock_layer
                refresh_sock_layer(self, bpy.context.scene)
            self.area.tag_redraw()

    def interpolate_pick(self, a, b, fraction):
        if self.sim.pose is not None:
            self.sim.pose.interpolate(a,b,fraction)
        if getattr(self,'monitor',None) is not None:
            self.monitor.surface.interpolate(a,b,fraction)

    def advance_motion(self, ctx, xy=None, *, pointer_changed=False, now=None, use_visible_pose=False):
        now=time.perf_counter() if now is None else now
        self.advancing=True
        try:
            target=None
            if self.wet_mode:
                self.wet_step+=1
                if xy is not None:
                    target=self.target_at(xy)
            # The warning has already presented the loaded pose. Commit the
            # cut on that pose before advancing the wall-clock animation again.
            opening_now=self.speed and not self.wet_mode and self.sim.warning_ready(now-self.last)
            pose_changed=False if use_visible_pose or opening_now else self.sync_pose(ctx,now)
            if xy is not None and not self.wet_mode:
                target=self.target_at(xy)
            self.tick(target,now,pointer_changed=pointer_changed,pose_changed=pose_changed)
        except ValueError as error:
            self.finish(ctx)
            self.report({'WARNING'},str(error))
            return False
        finally:
            self.advancing=False
            if self.wet_mode:
                self.wet_pointer_changed=False
        return True

    def publish_positions(self):
        from .controls import fracture_audio
        fracture_audio(self)
        if self.sim.pose is None and not getattr(self,'worker_seated',False):
            self.sim.seat_held_faces()
        if self.sim.changed:
            write_topology(self.obj.data,self.sim)
            self.refined_unpublished=False
            count = len(self.sim.cracks)
            # Press and release set the grip message themselves. Growing an existing
            # hole updates the viewport without invalidating its unchanged header.
            if self.obj.get('weft_live_tears')!=count:
                self.obj['weft_live_tears']=count
                if self.sim.grip is None:
                    self.area.header_text_set(f"已松手    破口 {count}    可继续选取其他位置    Enter 结束")
                else:
                    self.area.header_text_set(f"抓住袜面    破口 {count}    松开左键释放")
        elif self.refined_unpublished:
            self.publish_refinement()
        else:
            write_positions(self.obj.data,self.sim)
            if self.sim.pose is not None:
                self.obj.data.update()
            update_loop_image(self.sim)
        publish_skin(bpy.context,self.sim)
        publish_warning_yarn(self.obj,self.sim)
        publish_yarn_ends(self.obj,self.sim)
        self.sim.warning_dirty=False
        self.region.tag_redraw()

    def publish_refinement(self):
        if self.refined_unpublished:
            write_topology(self.obj.data,self.sim)
            self.refined_unpublished=False

    def refresh_monitor_snapshot(self):
        if getattr(self,'dry_input',None) is not None:
            self.dry_input.finish();self.dry_input=None
        self.advancing=True
        try:
            self.publish_refinement()
            if self.wet is not None and self.wet.faces is not self.sim.faces:
                self.wet.rebind()
                self.wet.prepare_probe_depth()
                self.wet_display.publish()
            self.capture_monitor()
        finally:
            self.advancing=False

    def publish_pending_surface(self):
        if self.refined_unpublished or (self.wet is not None and self.wet.faces is not self.sim.faces):
            self.refresh_monitor_snapshot()

    def enqueue_drag(self,xy):
        last=self.pending_drag[-1] if self.pending_drag else self.drag_xy or self.last_pick[0]
        if last==xy:return
        now=time.perf_counter()
        # Measure idle time after the previous handler returned. A costly
        # solve does not make two already-queued mouse events independent.
        if now-self.last_drag_event<.004:
            self.dense_drag_until=now+.14
        if self.pending_drag:
            origin=self.pending_drag[-2] if len(self.pending_drag)>1 else self.drag_xy or self.last_pick[0]
            ax,ay=last[0]-origin[0],last[1]-origin[1]
            bx,by=xy[0]-last[0],xy[1]-last[1]
            # Collinear samples between two draws share one endpoint. Keep
            # true reversals and turns that deviate by more than a pixel.
            if ax*bx+ay*by>=0 and (ax*by-ay*bx)**2<=1.5**2*((ax+bx)**2+(ay+by)**2):
                self.pending_drag[-1]=xy
            else:self.pending_drag.append(xy)
        else:self.pending_drag.append(xy)
        if now>=self.dense_drag_until:
            self.flush_drag()
        self.last_drag_event=time.perf_counter()

    def flush_drag(self,*,finish=False):
        while self.pending_drag and self.sim.grip is not None:
            self.drag_xy=self.pending_drag.pop(0)
            self.tick(self.target_at(self.drag_xy))
            self.last_drag_event=time.perf_counter()
            if not finish:break

    def flush_wet_pointer(self):
        if (not self.wet_pointer_changed or self.sim.grip is None or
                self.monitor_pause is not None or self.advancing):
            return
        self.wet_pointer_changed=False
        self.advancing=True
        try:
            self.tick(self.target_at(self.drag_xy),pointer_changed=True,advance_film=False)
        finally:
            self.advancing=False

    def release_grip(self, *, commit=True):
        if self.wet_pointer_changed:
            if commit:self.flush_wet_pointer()
            self.wet_pointer_changed=False
        if commit and self.pending_drag:self.flush_drag(finish=True)
        self.pending_drag.clear()
        if getattr(self,'dry_input',None) is not None:
            self.dry_input.finish();self.dry_input=None
        self.dense_drag_until=self.last_drag_event=0.
        parts=getattr(self.sim,'sections',None)
        if parts is not None:parts.free_retired()
        if getattr(self,'continued',False) and self.sim.grip is not None and self.sim.plan:
            continue_view.tick(self,settle=True)
        if getattr(self,'continued',False) and self.sim.plan and self.sim.plan.reached:
            cut=self.sim.plan
            if getattr(cut,'checked_rest',-1)!=cut.reached:
                from .seam_overlap import correct
                correct(self.sim,released=True)
        cut=getattr(self.sim,'plan',None)
        protected=(np.r_[cut.path[:cut.reached+1],list(cut.copies.values())].astype(int)
                   if cut is not None and cut.reached else ())
        self.sim.release()
        if commit and not self.wet_mode and self.sim.cracks:
            from . import bank_finish
            bank_finish.schedule(self,bpy.context.evaluated_depsgraph_get(),protected)
        if parts is not None:parts.drop_spares()
        from .atlas_transfer import flush
        flush(self.sim)
        if getattr(self,'continued',False):
            self.drag_xy=None
            self.last=time.perf_counter()
            self.area.header_text_set(f'已松手　当前 {len(self.sim.cracks)} 处缺口　可再抓布桥或孔尖')
            self.region.tag_redraw()
            return
        if self.sim.warning_dirty:
            self.publish_positions()
        if commit:
            self.refresh_monitor_snapshot()
        self.area.header_text_set(f"已松手    破口 {len(self.sim.cracks)}    可继续选取其他位置    Enter 结束")

    def pour_at(self, ctx, xy, fixed):
        sim=self.sim
        p=ctx.scene.weft_wet
        key='weft_wet_anchor' if p.side=='SHEET' else 'weft_slide_anchor'
        anchor=ctx.scene.get(key) if fixed else None
        if anchor is not None:
            eye=Vector(ctx.scene['weft_wet_pick_eye'])
            point,_,face,_=tree_for(sim.pos,sim.faces).ray_cast(Vector(anchor)+eye*.12,-eye,.24)
            if point is None:return
        else:
            origin,ray=self.pick_ray(xy)
            hit,point,face,obj=self.pick_visible(ctx,origin,ray)
            if not hit or obj!=self.obj:return
            point=material_hit(sim,self.obj.data,face,point)
        point=np.asarray(point)
        landing=None
        if self.wet is None:
            self.wet=WetSurface(sim,face,point,parameters(ctx.scene))
            self.wet.film.step=self.wet_step
            self.wet_display=WetDisplay(ctx.scene,self.obj,self.wet)
        else:
            _,weights=self.wet.landing_stencil(face,point)
            if not weights.any():
                raise ValueError('落点超出本轮湿态取样范围，重置可重新选点')
            landing=point
        if p.side=='SHEET':self.wet.pour(1 if p.direction=='A' else -1,landing,face=face)
        else:self.wet.dose(p.side.lower(),landing)
        self.wet.prepare_probe_depth()
        self.wet_display.publish()
        self.capture_monitor()

    def modal(self,ctx,event):
        if self.timer is None:return {'FINISHED'}
        sections=getattr(self.sim,'sections',None)
        if sections is not None:sections.shading(self.area.spaces.active.shading.type)
        if ctx.scene!=getattr(self,'scene',ctx.scene):
            self.finish(ctx);return {'FINISHED'}
        moving=self.sim.pose is not None
        wet_mode=getattr(self,'wet_mode',False)
        if not self.speed or self.monitor_pause is not None:
            self.pin_committed_pose()
        if event.type in {'ESC','RET','NUMPAD_ENTER'}:
            self.finish(ctx);return {'FINISHED'}
        xy=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
        from . import monitor_ui
        result=monitor_ui.event(self,ctx,event,xy)
        if result is not None:
            return result
        if event.type=='TIMER' and getattr(self,'bank_job',None) and self.sim.grip is None:
            from .bank_finish import collect
            collect(self)
        if event.type=='TIMER' and self.pending_drag and self.sim.grip is not None:
            self.flush_drag()
            return {'RUNNING_MODAL'}
        if wet_mode and event.type=='W' and event.value=='PRESS':
            if not self.pointer_in_view(xy):return {'PASS_THROUGH'}
            self.publish_pending_surface()
            self.advancing=True
            try:
                self.pour_at(ctx,xy,event.shift)
            except ValueError as error:
                self.report({'WARNING'},str(error))
            finally:
                self.advancing=False
            return {'RUNNING_MODAL'}
        if getattr(self,'continued',False):
            return continue_view.modal(self,ctx,event,xy)
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            if not self.pointer_in_view(xy):return {'PASS_THROUGH'}
            self.publish_pending_surface()
            # Pick the geometry committed for display. The timer advances the
            # newly selected material point through subsequent poses.
            origin,ray=self.pick_ray(xy)
            hit,loc,face,obj=self.pick_visible(ctx,origin,ray)
            self.last_pick=(xy,obj.name if hit else None,face)
            if hit and obj==self.obj:
                if continue_view.adopt(self,np.asarray(loc)):
                    continue_view.press(self,face,loc,xy)
                    return {'RUNNING_MODAL'}
                sim=self.sim
                visible=loc.copy()
                loc=material_hit(sim,self.obj.data,face,visible)
                offset=loc-visible
                before=len(sim.pos)
                history=getattr(sim,'neutral_history',None)
                neutral=getattr(sim,'open_rest',None)
                carry=(sim.dry and sim.fixed_material() and sim.cracks and history is not None and
                       neutral is not None and history[0] is sim.faces and neutral[0] is sim.faces)
                fields=(sim.opening,sim.seating,*history[1:]) if carry else (sim.opening,sim.seating)
                if getattr(sim,'bank_weave',None) is not None:
                    from .yarn_chart import check_domain
                    check_domain(sim,self.obj.data)
                weave=getattr(sim,'bank_weave',None)
                if weave is not None:fields=(*fields,weave)
                offsets=np.column_stack(fields)
                protected = (np.unique(np.concatenate([c['rim_ids'].ravel() for c in sim.cracks]))
                             if sim.dry and sim.cracks else None)
                sim.rest,sim.faces,sim.uv,sim.pos,offsets=refine_surface(
                    sim.rest,sim.faces,sim.uv,sim.pos,offsets,sim.rest[sim.faces[face]].mean(0),
                    interpolate=self.interpolate_pick if moving or getattr(self,'monitor',None) is not None else None,
                    protected=protected)
                sim.opening,sim.seating=offsets[:,:3].copy(),offsets[:,3:6].copy()
                if weave is not None:sim.bank_rest_owner=sim.rest
                if len(sim.pos)!=before:
                    if weave is not None:sim.bank_weave=offsets[:,-weave.shape[1]:].astype('f')
                    if sim.dry and sim.fixed_material():
                        sim.refresh_dry_support()
                        if carry:
                            # Edge midpoints lie on the already accepted released sheet.
                            # Retain both paid-out banks across a change of material indexing.
                            old_target=offsets[:,6:9].copy();old_neutral=offsets[:,9:12].copy()
                            sim.neutral_history=(sim.faces,old_target,old_neutral)
                            folded=None
                            if neutral[2] is not None:
                                sim.current_tear_faces()
                                folded=sim.tear_contact_band()
                            sim.open_rest=(sim.faces,old_neutral,folded)
                    self.refined_unpublished=True
                    picked=refined_hit(sim,loc,origin+offset,ray)
                    if picked is None:
                        self.publish_pending_surface()
                        ctx.view_layer.update()
                        hit,loc,_,face=self.obj.ray_cast(origin,ray,depsgraph=ctx.evaluated_depsgraph_get())
                        if not hit:
                            self.release_grip()
                            return {'RUNNING_MODAL'}
                        visible=loc.copy()
                        loc=material_hit(sim,self.obj.data,face,visible)
                    else:
                        loc,face=picked
                sim.pick(face,np.array(loc))
                self.grab_offset=np.asarray(loc-visible)
                self.grab_view_offset=visible-Vector(self.posed_grip_center())
                sim.grab_surface_offset=-self.grab_offset
                if moving:
                    sim.pose.pick(sim,face)
                self.drag_xy=None
                self.start=visible;self.last=time.perf_counter()
                if self.speed and not wet_mode:
                    # The press already requests a draw of this pose. A timer
                    # before that draw would submit the refinement twice.
                    self.motion_draw_pending=True
                if wet_mode:
                    self.refresh_monitor_snapshot()
                elif not moving:
                    self.publish_refinement()
                self.region.tag_redraw()
                self.area.header_text_set(f"抓住袜面    破口 {len(self.sim.cracks)}    松开左键释放")
            return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='RELEASE':
            in_view=self.pointer_in_view(xy)
            if not in_view and self.sim.grip is None:return {'PASS_THROUGH'}
            if wet_mode and self.sim.grip is not None and in_view and xy!=(self.drag_xy or self.last_pick[0]):
                self.drag_xy=xy
                self.wet_pointer_changed=True
            if moving and not wet_mode and self.sim.warning is None:
                self.publish_pending_surface()
                if not self.advance_motion(ctx,self.held_cursor()):
                    return {'FINISHED'}
            if not self.wet_pointer_changed:self.last=time.perf_counter()
            self.release_grip()
            return {'RUNNING_MODAL'} if in_view else {'PASS_THROUGH'}
        if event.type=='MOUSEMOVE' and self.sim.grip is not None:
            if not self.pointer_in_view(xy):
                return {'RUNNING_MODAL'}
            self.last_input=time.perf_counter()
            if not moving and not wet_mode:
                self.enqueue_drag(xy)
                return {'RUNNING_MODAL'}
            changed=xy!=(self.drag_xy or self.last_pick[0])
            if moving and self.speed and not wet_mode and self.drag_xy is None and not changed:
                return {'RUNNING_MODAL'}
            first_drag=self.drag_xy is None and changed
            self.drag_xy=xy
            if wet_mode:
                self.wet_pointer_changed |= changed
                return {'RUNNING_MODAL'}
            if moving:
                # Skin projection can shift the resolved target as the pose
                # moves. Only fresh pointer input drives the existing tear rule.
                # The first displacement refers to the pose the user grabbed.
                # Present that response before advancing the motion clock.
                if not self.advance_motion(ctx,xy,pointer_changed=changed,
                                           use_visible_pose=bool(self.speed and first_drag)):
                    return {'FINISHED'}
            else:
                self.tick(self.target_at(xy))
            return {'RUNNING_MODAL'}
        if event.type=='TIMER':
            if moving:
                # Present a committed grip before taking another timer sample.
                # Mouse and release events still solve immediately.
                if self.speed and not wet_mode and self.sim.grip is not None and self.motion_draw_pending:
                    return {'RUNNING_MODAL'}
                if (self.speed and not wet_mode and self.sim.grip is not None
                        and self.refined_unpublished and self.drag_xy is None
                        and self.motion_pose_due(ctx.scene, time.perf_counter()) == self.committed_pose()):
                    return {'RUNNING_MODAL'}
                if not self.advance_motion(ctx,self.held_cursor(),pointer_changed=wet_mode and self.wet_pointer_changed):return {'FINISHED'}
            else:
                self.tick()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

    def finish(self,ctx, *, finalize_snapshot=True):
        if self.timer is None:return
        if finalize_snapshot and self.wet_pointer_changed:self.flush_wet_pointer()
        if finalize_snapshot and self.pending_drag:self.flush_drag(finish=True)
        self.pending_drag.clear()
        if getattr(self,'dry_input',None) is not None:
            self.dry_input.finish();self.dry_input=None
        from .lifecycle import snapshot
        self.pin_committed_pose()
        if getattr(self.sim,'neutral_job',None) is not None:
            from .neutral_worker import collect
            collect(self.sim,settle=True)
        if self.sim.warning is not None:
            self.sim.cancel_warning()
            if finalize_snapshot:self.publish_positions()
        elif finalize_snapshot:
            self.publish_pending_surface()
        if finalize_snapshot and self.speed and not self.wet_mode and self.sim.grip is None and self.sim.cracks:
            write_positions(self.obj.data, self.sim, freeze_rim=True)
            self.obj.data.update()
        if finalize_snapshot and self.continued:
            if self.sim.grip is not None and self.sim.plan:continue_view.tick(self,settle=True)
            cut=self.sim.plan
            if cut and cut.reached and getattr(cut,'checked_rest',-1)!=cut.reached:
                from .seam_overlap import correct
                correct(self.sim,released=True,move_held=False)
            continue_view.store_neutral(self.obj.data,self.sim)
        if getattr(self,'bank_job',None):
            from .bank_finish import collect
            if finalize_snapshot:collect(self,wait=True)
            else:self.bank_job[-1].cancel();self.bank_job=None
        if finalize_snapshot and getattr(self.sim,'bank_weave',None) is not None:
            if self.continued:continue_view.store_neutral(self.obj.data,self.sim)
            else:
                neutral=self.sim.relaxed_tears()
                mesh=self.obj.data
                attr=mesh.attributes.get(continue_view.NEUTRAL) or mesh.attributes.new(continue_view.NEUTRAL,'FLOAT_VECTOR','POINT')
                attr.data.foreach_set('vector',neutral.astype('f').ravel())
        if getattr(self.sim,'sections',None) is not None:
            self.sim.sections.close();self.sim.sections=None
        from .knit_tangent import finish as finish_tangents
        finish_tangents(self.sim,self.obj.data)
        from .atlas_transfer import flush
        flush(self.sim,close=True)
        from .live_rims import clear_empty_strands
        clear_empty_strands(self.obj)
        if finalize_snapshot:snapshot(self)
        if getattr(self,'monitor',None) is not None:
            from . import monitor_ui
            monitor_ui.detach(self)
        self.sim.release()
        restore_skin(self.sim)
        if finalize_snapshot and hasattr(self.sim,'loop_image'):self.sim.loop_image.pack()
        if finalize_snapshot and getattr(self.sim,"run_image",None):
            self.sim.run_image.pack()
        ctx.window_manager.event_timer_remove(self.timer)
        self.timer=None
        bpy.types.SpaceView3D.draw_handler_remove(self.marker,'WINDOW')
        self.area.tag_redraw()
        self.area.header_text_set(None)
        ACTIVE.pop(self.scene.as_pointer(),None)

    def cancel(self,ctx):
        self.finish(ctx)


def motion_readings(scene, full=False):
    """Read a committed step, or None while a pose is being evaluated."""
    tool=ACTIVE.get(scene.as_pointer())
    if tool is None or tool.sim.pose is None or tool.advancing:
        return None
    return tool.sim.pose.readings(tool.sim,full)


def fabric_readings(scene, uv=None):
    """Read local stretch, aperture area and material failures at the last live step."""
    tool=ACTIVE.get(scene.as_pointer())
    if tool is None or tool.advancing:
        return None
    return tool.sim.fracture_readings(uv)


def motion_contact_at(scene, world_xyz):
    """Query a world-space point against the skin at the last committed step."""
    tool=ACTIVE.get(scene.as_pointer())
    if tool is None or tool.sim.pose is None or tool.advancing:
        return None
    return tool.sim.pose.contact_at(world_xyz)


def wet_readings(scene):
    """Return the committed wet probe and inventory on the 02 step clock."""
    tool=ACTIVE.get(scene.as_pointer())
    if tool is None or getattr(tool,'wet',None) is None or tool.advancing:
        return None
    return tool.wet.reading()


def wet_surface_fields(scene):
    """Return both material lattices from the committed step, or None."""
    tool=ACTIVE.get(scene.as_pointer())
    if tool is None or getattr(tool,'wet',None) is None or tool.advancing:
        return None
    return tool.wet.fields()


class WEFT_OT_live_reset(bpy.types.Operator):
    bl_idname="weft.live_reset"
    bl_label="结束实验并恢复原袜"
    bl_options={'REGISTER','UNDO'}

    def execute(self,ctx):
        tool=ACTIVE.get(ctx.scene.as_pointer())
        if tool is not None:tool.finish(ctx,finalize_snapshot=False)
        obj=next((o for o in ctx.scene.objects if o.get('weft_live_source')),None)
        if not obj:return {'CANCELLED'}
        rests=[o for o in ctx.scene.objects if o.get("weft_skin_response_source")]
        for skin_display in rests:
            remove_skin_display(skin_display)
        src=obj['weft_live_source']
        src.hide_set(False);src.hide_render=obj['weft_source_render_hidden']
        if obj.get('weft_source_fit_deferred'):
            from .sock_length import refresh_contact
            refresh_contact(src,ctx.evaluated_depsgraph_get())
        data=obj.data
        material=data.materials[0] if data.materials else None
        gauges = []
        if material and material.use_nodes:
            gauges = [node.image for node in material.node_tree.nodes if node.type=='TEX_IMAGE'
                      and node.image and node.label in (GAUGE,"Surviving wale and course yarn")]
        remove_yarn_ends(obj)
        remove_wet_objects(ctx.scene,obj)
        snapshot=obj.get('weft_snapshot')
        bpy.data.objects.remove(obj,do_unlink=True)
        if snapshot is not None and snapshot.users <= int(snapshot.use_fake_user):
            bpy.data.texts.remove(snapshot)
        if not data.users:bpy.data.meshes.remove(data)
        if gauges and not material.users:bpy.data.materials.remove(material)
        for image in gauges:
            if not image.users:bpy.data.images.remove(image)
        return {'FINISHED'}
