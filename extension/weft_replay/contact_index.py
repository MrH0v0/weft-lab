"""Optional native index for reliable nearest-face queries in dry live contact."""
import ctypes
from pathlib import Path
import platform
import sys
from weakref import finalize

import numpy as np


def load_index():
    if sys.platform != 'win32' or platform.machine().lower() not in {'amd64','x86_64'}:
        return None
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name('contact_index_win64.dll')))
    except OSError:
        return None
    address, count = ctypes.c_void_p, ctypes.c_int
    for name,args in (
        ('weft_patch_planes',[address,address,address,count,address,address,address]),
        ('weft_project_grab',[address,address,address,address,address,address,count,address,address]),
        ('weft_material_normals',[address,address,count,count,address]),('weft_face_probe_groups',[address,address,count,address])):
        method=getattr(library,name);method.argtypes=args;method.restype=None
    library.weft_buckle_patch.argtypes=[address,address,count,address,count]+[address]*14+[ctypes.c_double]*2
    library.weft_buckle_patch.restype=None
    library.weft_mesh_edges.argtypes=[address,count,count,address,address]
    library.weft_mesh_edges.restype=count
    library.weft_index_new.argtypes = [address,count,address,count]
    library.weft_index_new.restype = address
    library.weft_index_delete.argtypes = [address]
    library.weft_index_delete.restype = None
    library.weft_index_refit.argtypes = [address,address]
    library.weft_index_refit.restype = None
    library.weft_index_nearest.argtypes = [address,address,count,count,address,address,address]
    library.weft_index_nearest.restype = None
    library.weft_first_hits.argtypes=[address,address,address,count,ctypes.c_double,address]
    library.weft_overlap_candidates.argtypes=[address,address,count,address]
    library.weft_overlap_changed_candidates.argtypes=[address,address,count,address,address]
    library.weft_overlap_changed_candidates.restype=None
    library.weft_project_sheet.argtypes=[address,count,count,address,count,address,address,address,address,address,address,address]
    for name in ('weft_first_hits','weft_project_sheet','weft_overlap_candidates'):
        getattr(library,name).restype=None
    library.weft_project_dry_hold.argtypes=library.weft_project_sheet.argtypes
    library.weft_project_dry_hold.restype=None
    library.weft_close_courses.argtypes=[address,address,count,ctypes.c_float,address]
    library.weft_close_courses.restype=None
    library.weft_wet_face_lifts.argtypes=[address,address,address,address,count,address,count,address]
    library.weft_wet_face_lifts.restype=None
    library.weft_transport_film.argtypes=[count,count]+[ctypes.c_double]*7+[address]*11
    library.weft_transport_film.restype=None
    library.weft_advance_grip.argtypes=[address,address,count,address,address,address,address,ctypes.c_double,address,address]
    library.weft_advance_grip.restype=None
    library.weft_pinched_profile.argtypes=[address]*5+[count]+[address]*4+[ctypes.c_double]*4+[address]*3
    library.weft_pinched_profile.restype=None
    library.weft_gauges.argtypes=[address,count,address,address,count]+[address]*4
    library.weft_gauges.restype=None
    library.weft_repair_fans.argtypes=[address,address,count,address,count,count,address]
    library.weft_repair_fans.restype=None
    library.weft_tangent_fans.argtypes=[address,address,address,count,address,count]
    library.weft_tangent_fans.restype=count
    return library


_index = load_index()


def advance_grip(sim,goal,factor,shift):
    if _index is None or sim.pos.dtype!=np.float64 or not sim.pos.flags.c_contiguous:return False
    # A grip picked on Blender's float32 mesh survives a cut, which promotes
    # the subdivided positions to float64. The native Vec ABI always reads doubles.
    goal,weights,bary,target,normal=(np.ascontiguousarray(value,dtype=np.float64)
                                      for value in (goal,sim.weight,sim.bary,sim.target,sim.normal))
    if shift is not None:shift=np.ascontiguousarray(shift,dtype=np.float64)
    grip=np.ascontiguousarray(sim.grip,dtype=np.int32)
    ptr=lambda a:a.ctypes.data
    _index.weft_advance_grip(ptr(sim.pos),ptr(goal),len(goal),ptr(weights),ptr(grip),ptr(bary),
                             ptr(target),factor,None if shift is None else ptr(shift),ptr(normal))
    return True


def pinched_profile(sim,goal,travel,direction,spread,length,lift,tension,flank):
    ids=np.ascontiguousarray(sim.drag_ids,dtype=np.int32)
    offset,distance,taper,travel,direction,normal=(np.ascontiguousarray(a,dtype=np.float64)
        for a in (sim.offset,sim.distance_sq,sim.drag_taper,travel,direction,sim.normal))
    sim.weight=np.ascontiguousarray(sim.weight,dtype=np.float64)
    if flank is not None:flank=np.ascontiguousarray(flank,dtype=np.float64)
    shift=None if flank is None else np.zeros(len(goal))
    ptr=lambda a:a.ctypes.data
    grip=np.ascontiguousarray(sim.grip,dtype=np.int32)
    _index.weft_pinched_profile(ptr(goal),ptr(offset),ptr(distance),ptr(taper),
                                ptr(ids),len(ids),ptr(grip),ptr(travel),ptr(direction),ptr(normal),
                                spread,length,lift,tension,None if flank is None else ptr(flank),ptr(sim.weight),
                                None if shift is None else ptr(shift))
    return shift


def measure_gauges(gauges,points):
    if _index is None or points.dtype not in (np.float32,np.float64) or not points.flags.c_contiguous:
        return None
    count=len(gauges.corners)
    centers=np.empty((count,3));vectors=np.empty((count,2,3))
    lengths=np.empty((count,2));angles=np.empty(count)
    ptr=lambda a:a.ctypes.data
    _index.weft_gauges(ptr(points),points.itemsize,ptr(gauges.corners),ptr(gauges.weights),count,
                       ptr(centers),ptr(vectors),ptr(lengths),ptr(angles))
    gauges.centers=centers;gauges.vectors=vectors
    return lengths,angles


def repair_fans(points,reference,seed,attempt,active):
    if _index is None or points.dtype!=np.float64 or not points.flags.c_contiguous:return False
    ref=np.ascontiguousarray(reference,dtype=np.float64)
    ids=np.ascontiguousarray(seed,dtype=np.int32)
    _index.weft_repair_fans(points.ctypes.data,ref.ctypes.data,len(points),ids.ctypes.data,
                            len(ids),attempt,active.ctypes.data)
    return True


def tangent_fans(points,reference,tri,normals):
    if _index is None or points.dtype!=np.float64 or not points.flags.c_contiguous:return None
    ref,axes=(np.ascontiguousarray(a,dtype=np.float64) for a in (reference,normals))
    faces=np.ascontiguousarray(tri,dtype=np.int32)
    return bool(_index.weft_tangent_fans(points.ctypes.data,ref.ctypes.data,axes.ctypes.data,
                                         len(points),faces.ctypes.data,len(faces)))


def transport_film(film,geometry,steps,dt,uptake_rate):
    """Update the same field buffers as Film.transport and retain its NumPy fallback."""
    if _index is None:
        return False
    connected,edge,length,width=geometry
    result=np.zeros(7)
    liquid=film.liquid
    read=[film.transport_neighbours,connected,edge,length,width,film.area,film.capacity]
    buffers=[np.ascontiguousarray(a) for a in read]+[film.outer,film.knit,film.momentum,result]
    _index.weft_transport_film(len(film.outer),steps,dt,film.pitch,liquid.viscosity,liquid.tension,
                               liquid.uptake,liquid.drying,uptake_rate,*(a.ctypes.data for a in buffers))
    film.flow+=result[:3]
    film.outer_runoff+=result[3];film.out+=result[3]
    film.downslope_runoff+=result[4];film.outer_to_knit+=result[5];film.evap+=result[6]
    return True


class ContactIndex:
    def __init__(self, points, triangles):
        self.geometry = points
        self.vertices = np.ascontiguousarray(points, dtype=np.float32)
        faces = np.ascontiguousarray(triangles, dtype=np.int32)
        self.handle = _index.weft_index_new(self.vertices.ctypes.data, len(self.vertices),
                                            faces.ctypes.data, len(faces))
        if self.handle:
            finalize(self, _index.weft_index_delete, self.handle)

    @classmethod
    def create(cls, points, triangles):
        if _index is None or not np.isfinite(points).all():
            return None
        result = cls(points, triangles)
        return result if result.handle else None

    def update(self, points):
        if points is not self.geometry:
            vertices = np.ascontiguousarray(points, dtype=np.float32)
            _index.weft_index_refit(self.handle, vertices.ctypes.data)
            self.vertices = vertices
            self.geometry = points

    def first_hits(self, origins, directions, limit):
        origins=np.ascontiguousarray(origins,dtype=np.float32)
        directions=np.ascontiguousarray(directions,dtype=np.float32)
        result=np.empty(len(origins),dtype=np.float32)
        _index.weft_first_hits(self.handle,origins.ctypes.data,directions.ctypes.data,len(origins),limit,result.ctypes.data)
        return result

    def wet_face_lifts(self,positions,normals,faces,bary):
        xyz,axes,weights=(np.ascontiguousarray(a,dtype=np.float64) for a in (positions,normals,bary))
        tiles=np.ascontiguousarray(faces,dtype=np.int32)
        lifts=np.empty(len(tiles),dtype=np.float64)
        _index.weft_wet_face_lifts(self.handle,xyz.ctypes.data,axes.ctypes.data,tiles.ctypes.data,len(tiles),
                                   weights.ctypes.data,len(weights),lifts.ctypes.data)
        return lifts

    def patch_planes(self,points,axes):
        points=np.ascontiguousarray(points,dtype=np.float64);axes=np.ascontiguousarray(axes,dtype=np.float64)
        anchors=points.copy();normals=np.zeros_like(points);status=np.zeros(len(points),dtype='b')
        _index.weft_patch_planes(self.handle,points.ctypes.data,axes.ctypes.data,len(points),anchors.ctypes.data,
                                 normals.ctypes.data,status.ctypes.data)
        return anchors,normals,np.flatnonzero(status<0)

    def nearest(self, points, *, check_faces=False, clear_at=None):
        """Nonnegative ids are unique hits, -2 certifies clearance, -1 uses Blender."""
        xyz = np.ascontiguousarray(points, dtype=np.float32).reshape(-1,3)
        data = np.zeros((len(xyz),8), np.float32)
        faces = np.empty(len(xyz), np.int32)
        limits = None if clear_at is None else np.ascontiguousarray(
            np.broadcast_to(clear_at, (len(xyz),)), dtype=np.float32)
        _index.weft_index_nearest(self.handle, xyz.ctypes.data, len(xyz), int(check_faces),
                                  limits.ctypes.data if limits is not None else None,
                                  data.ctypes.data, faces.ctypes.data)
        return faces, data

def constraint_projection(points, frame, neutral=None,*,dry_hold=False):
    if _index is None or points.dtype not in (np.float32,np.float64) or not points.flags.c_contiguous:
        return False
    ids,l,t,h,n,incidence=frame
    slots,neighbours=(np.ascontiguousarray(v,dtype=np.int64) for v in incidence)
    ids=np.ascontiguousarray(ids,dtype=np.int32)
    values=[np.ascontiguousarray(v,dtype=np.float64) for v in (l,t,h,n)]
    ref=None if neutral is None else np.ascontiguousarray(neutral,dtype=points.dtype)
    project=_index.weft_project_dry_hold if dry_hold else _index.weft_project_sheet
    project(points.ctypes.data,points.itemsize,len(points),ids.ctypes.data,len(ids),
                              *(v.ctypes.data for v in values),None if ref is None else ref.ctypes.data,
                              slots.ctypes.data,neighbours.ctypes.data)
    return True


def close_courses(points,faces,margin):
    if _index is None:return None
    xyz=np.ascontiguousarray(points,dtype=np.float32)
    tiles=np.ascontiguousarray(faces,dtype=np.int32)
    flags=np.zeros(len(tiles),dtype=np.uint8)
    _index.weft_close_courses(xyz.ctypes.data,tiles.ctypes.data,len(tiles),margin,flags.ctypes.data)
    return np.flatnonzero(flags)


def overlap_candidates(points,faces,active=None):
    if _index is None:
        return None
    xyz=np.ascontiguousarray(points,dtype=np.float32)
    tiles=np.ascontiguousarray(faces,dtype=np.int32)
    candidates=np.zeros(len(tiles),dtype=np.uint8)
    if active is None:
        _index.weft_overlap_candidates(xyz.ctypes.data,tiles.ctypes.data,len(tiles),candidates.ctypes.data)
    else:
        changed=np.ascontiguousarray(active,dtype=np.uint8)
        _index.weft_overlap_changed_candidates(xyz.ctypes.data,tiles.ctypes.data,len(tiles),changed.ctypes.data,candidates.ctypes.data)
    return np.flatnonzero(candidates)

def mesh_edges(mesh,tri,nv):
    if _index is None:
        mesh.update(calc_edges=True)
        return
    return assign_edges(mesh,edge_arrays(tri,nv))


def edge_arrays(tri,nv):
    if _index is None:return None
    tri=np.ascontiguousarray(tri,dtype='i')
    edges=np.empty((tri.size,2),dtype='i');corners=np.empty_like(tri)
    count=_index.weft_mesh_edges(tri.ctypes.data,nv,len(tri),edges.ctypes.data,corners.ctypes.data)
    return corners,edges[:count]


def assign_edges(mesh,arrays):
    corners,edges=arrays
    mesh.edges.add(len(edges))
    mesh.attributes['.edge_verts'].data.foreach_set('value',edges.ravel())
    mesh.attributes['.corner_edge'].data.foreach_set('value',corners.ravel())
    return arrays


def material_normals(points,faces):
    if _index is None or points.dtype!=np.float64:return None
    xyz=np.ascontiguousarray(points)
    tri=np.ascontiguousarray(faces,dtype='i')
    out=np.zeros((len(xyz),3))
    _index.weft_material_normals(xyz.ctypes.data,tri.ctypes.data,len(xyz),len(tri),out.ctypes.data)
    return out


def face_probe_groups(points,faces):
    if _index is None or points.dtype!=np.float64:return None
    tri=np.ascontiguousarray(faces,dtype='i');xyz=np.ascontiguousarray(points)
    out=np.empty(len(tri),dtype='i')
    _index.weft_face_probe_groups(xyz.ctypes.data,tri.ctypes.data,len(tri),out.ctypes.data)
    return out


def buckle_patch(patch,goal,direction,grip_normal,weight,center,travel):
    if _index is None or goal.dtype!=np.float64:return False
    if not hasattr(patch,'native_folds'):
        patch.native_folds=tuple(np.ascontiguousarray(a,dtype='d') for a in
            (*patch.fold_basis,*patch.fold_edges,patch.fold_area,patch.corner_area,patch.offset,patch.normal,patch.distance))
        patch.fold_ids=np.ascontiguousarray(patch.ids,dtype='i')
        patch.fold_tri=np.ascontiguousarray(patch.tri,dtype='i')
    across=np.cross(grip_normal,direction);skew=.18+.23*float(across@patch.wale)
    vectors=tuple(np.ascontiguousarray(a,dtype=np.float64) for a in (weight,direction,across,grip_normal))
    if center is not None:center=np.ascontiguousarray(center,dtype=np.float64)
    args=(*patch.native_folds,*vectors)
    _index.weft_buckle_patch(goal.ctypes.data,patch.fold_ids.ctypes.data,len(patch.ids),patch.fold_tri.ctypes.data,len(patch.tri),
        *(v.ctypes.data for v in args),None if center is None else center.ctypes.data,travel,skew)
    return True


def project_grab(sim,skin):
    pos=sim.pos
    if _index is None or pos.dtype!=np.float64 or not pos.flags.c_contiguous:return False
    cache=getattr(sim,'grab_contact_cache',None)
    if cache is None or cache[0] is not sim.patch or cache[1] is not sim.contact_normals:
        axes=sim.contact_normals.copy();axes[sim.patch.ids]=sim.patch.normal
        cache=sim.grab_contact_cache=(sim.patch,sim.contact_normals,axes,np.empty_like(pos),np.zeros(len(pos),'B'))
    grip=None if sim.grip is None else np.ascontiguousarray(sim.grip,dtype='i')
    _index.weft_project_grab(skin.handle,pos.ctypes.data,sim.rest.ctypes.data,sim.seating.ctypes.data,cache[2].ctypes.data,
                             None if grip is None else grip.ctypes.data,len(pos),cache[3].ctypes.data,cache[4].ctypes.data)
    return True


