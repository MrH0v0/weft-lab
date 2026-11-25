"""Draw the gathered yarn along the rim and the attached free ends."""
import bpy
import numpy as np

from .dry_fabric import PALETTES


def publish_yarn_ends(obj, sim):
    if not sim.cracks:
        return
    paths = [strand for _,rim,ends in sim.damage_yarns() for strand in (*rim,*ends)]
    publish_paths(obj, sim, paths, 'yarn_ends', 'loose yarn', radius=.00023 if sim.dry else .00012)


def publish_warning_yarn(obj, sim):
    """Show the yarn still bridging interlocks as the loaded row opens."""
    if not sim.dry:
        return
    warning = sim.warning
    stitches = None
    if warning is not None:
        loops = sim.patch.loops
        stitches, _, reveal = sim.loaded_stitches()
        visible = reveal > .04
        stitches, reveal = stitches[visible], reveal[visible]
        x, y = loops.cells[stitches].T
        exposed = sim.loop_pixels[y-sim.first_row, x, 3] < .10
        stitches, reveal = stitches[exposed], reveal[exposed]
    if stitches is None or not len(stitches):
        child = getattr(sim, 'warning_yarn_obj', None)
        if child is not None and len(child.data.splines):
            child.data.splines.clear()
            sim.warning_yarn_points=None
        return
    corners = sim.patch.ids[loops.corners[stitches]]
    sample = np.einsum('nkq,nkqd->nkd', loops.weights[stitches], sim.pos[corners])
    normal = np.cross(sample[:,1]-sample[:,0], sample[:,3]-sample[:,2])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    normal *= np.where(normal@sim.normal < 0., -1., 1.)[:,None]
    across = 2*warning['axis']
    bridges = np.stack((sample[:,across], sample.mean(1)+.00023*normal,
                        sample[:,across+1]), axis=1)
    bridges += .00025*normal[:,None]
    radii = reveal[:,None]*np.array([.52, 1., .52])
    publish_paths(obj, sim, list(zip(bridges,radii)),
                  'warning_yarn', 'loaded knit', radius=.000135)


def publish_traced_yarn(obj, sim, paths):
    segments = [(path,np.ones(len(path))) for path in paths]
    publish_paths(obj, sim, segments, 'traced_yarn', 'traced stitch', (.30,.095,.022,1))


def strand_object(obj,sim,key,name,color,radius):
    child=getattr(sim,key+'_obj',None)
    if child is not None:return child
    curve = bpy.data.curves.new(f'Weft • {name}', 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = radius
    curve.bevel_resolution = 1
    palette = int(np.clip(obj['weft_live_source'].get('weft_dry_palette', 0),0,len(PALETTES)-1))
    color = PALETTES[palette] if color is None else color
    material = bpy.data.materials.new(curve.name)
    material.use_nodes = True
    tint_yarn(material, color)
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Roughness'].default_value = .62
    shader.inputs['Specular IOR Level'].default_value = .20
    shader.inputs['Sheen Weight'].default_value = .12
    if key=='yarn_ends' and sim.dry:
        # Individual untwisted filaments transmit skin light. An opaque
        # bundle on a sheer stocking reads as a black plug at the small eye.
        set_strand_opacity(material,obj['weft_live_source'].get('weft_dry_thickness',.64))
    curve.materials.append(material)
    child = bpy.data.objects.new(curve.name, curve)
    child['weft_yarn_ends'] = True
    child.parent = obj
    child.hide_select = True
    obj.users_collection[0].objects.link(child)
    setattr(sim,key+'_obj',child)
    return child


def publish_paths(obj, sim, paths, key, name, color=None, *, radius=.00012):
    if not paths:
        return
    points = np.vstack([path for path,_ in paths])
    child=strand_object(obj,sim,key,name,color,radius)
    curve = child.data
    previous = getattr(sim,key+'_points',None)
    sizes=tuple(len(path) for path,_ in paths)
    old_sizes=tuple(len(spline.points) for spline in curve.splines)
    same_layout=old_sizes==sizes
    radii=np.concatenate([radius for _,radius in paths])
    if (same_layout and previous is not None and np.max(abs(previous-points)) < .00004
            and np.max(abs(getattr(sim,key+'_radii')-radii)) < .03):
        return
    if not same_layout:
        prefix=next((i for i,(a,b) in enumerate(zip(old_sizes,sizes)) if a!=b),min(len(old_sizes),len(sizes)))
        for spline in list(curve.splines)[prefix:]:curve.splines.remove(spline)
        for size in sizes[prefix:]:curve.splines.new('POLY').points.add(size-1)
    coords=np.ones((len(points),4),'f');coords[:,:3]=points
    packed_radii=radii.astype('f',copy=False)
    starts=np.r_[0,np.cumsum(sizes,dtype=int)[:-1]]
    if same_layout and previous is not None:
        # RNA writes tag the curve for rebuilding. Preserve every stored float
        # while leaving stationary strands and their original taper alone.
        different=(coords[:,:3]!=previous.astype('f',copy=False)).any(1)
        moved=np.logical_or.reduceat(different,starts)
        last_radii=getattr(sim,key+'_radii')
        tapered=np.logical_or.reduceat(packed_radii!=last_radii.astype('f',copy=False),starts)
    else:
        moved=tapered=np.ones(len(sizes),bool)
    for i,(spline,size) in enumerate(zip(curve.splines,sizes)):
        start=starts[i];end=start+size
        if moved[i]:spline.points.foreach_set('co',coords[start:end].ravel())
        if tapered[i]:spline.points.foreach_set('radius',packed_radii[start:end])
    setattr(sim,key+'_points',points)
    setattr(sim,key+'_radii',radii)


def prepare_strands(obj,sim):
    # Link the empty curve datablocks on tool activation. The first genuine
    # rupture only supplies the positions and taper of the surviving yarn.
    strand_object(obj,sim,'yarn_ends','loose yarn',None,.00023)
    strand_object(obj,sim,'warning_yarn','loaded knit',None,.000135)


def clear_empty_strands(obj):
    for child in list(obj.children):
        if child.type=='CURVE' and child.get('weft_yarn_ends') and not len(child.data.splines):
            remove_path(child)


def tint_yarn(material, color):
    material.diffuse_color = color
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = color


def set_strand_opacity(material, thickness):
    # The reference loose end transmits 36 percent at the default gauge.
    # Scale its optical depth with the garment, including its thin residual yarn.
    density=.050 + (float(thickness)/.64)*(-np.log(.36)-.050)
    material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value=-np.expm1(-density)
    if material.surface_render_method!='BLENDED':
        material.surface_render_method='BLENDED'
    material.use_transparency_overlap=False


def update_yarn_appearance(sim, source):
    palette=source.get('weft_dry_palette',0)
    for name in ('yarn_ends_obj', 'warning_yarn_obj'):
        obj = getattr(sim,name,None)
        if obj is not None:
            material=obj.data.materials[0]
            tint_yarn(material,PALETTES[palette])
            if name=='yarn_ends_obj' and sim.dry:
                set_strand_opacity(material,source.get('weft_dry_thickness',.64))


def remove_path(child):
    curve = child.data
    material = curve.materials[0]
    bpy.data.objects.remove(child, do_unlink=True)
    if not curve.users:
        bpy.data.curves.remove(curve)
    if not material.users:
        bpy.data.materials.remove(material)


def remove_yarn_ends(obj):
    for child in list(obj.children):
        if child.get('weft_yarn_ends'):
            remove_path(child)
