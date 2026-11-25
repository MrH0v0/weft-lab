"""Conforming dry tear banks in the garment's unwrapped stitch chart."""
import numpy as np
from mathutils.geometry import delaunay_2d_cdt
from mathutils.bvhtree import BVHTree
from .fracture import chart_delta, seam_bend
from .knit_patch import surface_normals


def bridge_samples(sim, active, other, hand_side):
    """Locate intact material on the loaded shoulder between the two facing tips."""
    load=active['load_axis']
    run=1-load
    separation=chart_delta(other['uv'],active['uv'])
    forward=separation[run]>0
    a=np.zeros(2)
    a[run]=active['fronts'][int(forward)]
    b=separation.copy()
    b[run]+=other['fronts'][int(not forward)]
    points=a+(b-a)*np.array([[.20],[.80]])
    points[:,load]+=hand_side
    center=.5*(active['center']+other['center'])
    faces=sim.faces[np.linalg.norm(sim.rest[sim.faces].mean(1)-center,axis=1)<.065]
    chart=chart_delta(sim.uv[faces],active['uv'])
    matrix=(chart[:,1:]-chart[:,:1]).transpose(0,2,1)
    good=(np.ptp(chart[:,:,0],axis=1)<72)&(abs(np.linalg.det(matrix))>1e-8)
    faces,chart=faces[good],chart[good]
    if not len(faces):
        return None
    ab=np.einsum('fij,pfj->pfi',np.linalg.inv(matrix[good]),points[:,None]-chart[None,:,0])
    weights=np.concatenate((1.-ab.sum(2,keepdims=True),ab),axis=2)
    hit=weights.min(2).argmax(1)
    weights=weights[np.arange(2),hit]
    if weights.min() < -1e-5:
        return None
    return faces[hit],weights


def strip_seating(sim, triangles):
    ids,local=np.unique(triangles,return_inverse=True)
    local=local.reshape(-1,3)
    normals=surface_normals(sim.rest,sim.faces)[ids]
    points=sim.rest[ids]+sim.seating[ids]
    # The original vertices can clear curved skin while a new diagonal crosses it.
    # Preserve the material reference. Add the required clearance to the neutral pose.
    moved=sim.project_skin(points.copy(),normals,.00022)
    for corners in (local, np.unique(np.sort(local[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)):
        sample=moved[corners].mean(1)
        n=normals[corners].mean(1)
        n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
        projected=sim.project_skin(sample.copy(),n,.00022)
        depth=np.einsum('ij,ij->i',projected-sample,n)
        lift=np.zeros(len(ids))
        np.maximum.at(lift,corners,np.maximum(0.,depth[:,None]))
        moved+=lift[:,None]*normals
    return ids,moved-points


def seat_strip(sim,triangles,prepared=None):
    if not hasattr(sim,'project_skin'):return
    ids,shift=prepared if prepared is not None else strip_seating(sim,triangles)
    sim.seating[ids]+=shift
    sim.pos[ids]+=shift


def inside_boundary(triangles,boundary):
    # Classify the constrained faces by crossing actual cavity edges. A
    # nearest point at a centroid can land on the wrong side of a narrow bay.
    from collections import defaultdict,deque
    edges=defaultdict(list)
    for i,(a,b,c) in enumerate(triangles.tolist()):
        for v,w in ((a,b),(b,c),(c,a)):edges[min(v,w),max(v,w)].append(i)
    wall={tuple(sorted(e)) for e in boundary.tolist()}
    if not wall.issubset(edges):return None
    states=np.full(len(triangles),-1,dtype='b');neighbors=[[] for _ in states];queue=deque()
    for key,fs in edges.items():
        crosses=int(key in wall)
        if len(fs)==1:
            f=fs[0]
            if states[f]>=0 and states[f]!=crosses:return None
            if states[f]<0:states[f]=crosses;queue.append(f)
        elif len(fs)==2:
            a,b=fs;neighbors[a].append((b,crosses));neighbors[b].append((a,crosses))
        else:return None
    while queue:
        a=queue.popleft()
        for b,flip in neighbors[a]:
            expected=states[a]^flip
            if states[b]<0:states[b]=expected;queue.append(b)
            elif states[b]!=expected:return None
    return states==1 if (states>=0).all() else None


def plan_strip(sim, center, uv, load, phase):
    run = 1-load
    chart = chart_delta(sim.uv,uv)[:,[load,run]]
    faces = sim.faces
    middle = chart[faces].mean(1)
    selected = (abs(middle[:,0])<5.) & (abs(middle[:,1])<19.)
    rows=np.flatnonzero(selected)
    selected[rows]=np.linalg.norm(sim.rest[faces[rows]].mean(1)-center,axis=1)<.071
    if sim.cracks:
        other = np.concatenate([c['rim_ids'].ravel() for c in sim.cracks])
        rows=np.flatnonzero(selected)
        selected[rows]=~np.isin(faces[rows],other).any(1)
    cut = faces[selected]
    if len(cut)<3:
        return None
    # The cavity boundary is shared exactly with the retained garment.
    edges, count = np.unique(np.sort(cut[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),
                             axis=0,return_counts=True)
    boundary = edges[count==1]
    vertices = np.unique(cut)
    triangles = chart[cut]
    matrix = (triangles[:,1:]-triangles[:,:1]).transpose(0,2,1)
    valid = abs(np.linalg.det(matrix))>1e-8
    cut,triangles = cut[valid],triangles[valid]
    inv = np.linalg.inv(matrix[valid])
    if not len(cut):
        return None
    flat=triangles.reshape(-1,2)
    tree=BVHTree.FromPolygons(np.column_stack((flat,np.zeros(len(flat)))).tolist(),
                              np.arange(len(flat)).reshape(-1,3).tolist(),all_triangles=True)

    def locate(points):
        hits=[tree.find_nearest((x,y,0.)) for x,y in points]
        face=np.fromiter((h[2] for h in hits),dtype=int,count=len(hits))
        ab=np.einsum('nij,nj->ni',inv[face],points-triangles[face,0])
        weights=np.column_stack((1.-ab.sum(1),ab))
        return weights.min(1)>-1e-5,cut[face],weights

    # Half-pitch endpoints are the boundaries of individual failed loops.
    # The same stations on either shoulder prevent long fans along short rim edges.
    station = np.arange(-16.5,17.,.5)
    line = np.column_stack((seam_bend(station,phase),station))
    good = np.logical_and.reduce([locate(line+[offset,0])[0] for offset in (-.85,0.,.85)])
    middle = len(station)//2
    if not good[middle-1:middle+2].all():
        return None
    lo = next((i+1 for i in range(middle-1,-1,-1) if not good[i]),0)
    hi = next((i for i in range(middle+1,len(good)) if not good[i]),len(good))
    station,line = station[lo:hi],line[lo:hi]
    if station[0]>-1. or station[-1]<1.:
        return None
    kept = chart[vertices]
    core = ((abs(kept[:,0]-seam_bend(kept[:,1],phase))<1.45) &
            (kept[:,1]>=station[0]-.35)&(kept[:,1]<=station[-1]+.35))
    fixed = np.unique(np.r_[boundary.ravel(),sim.grip if sim.grip is not None else []]).astype(int)
    kept = vertices[~core|np.isin(vertices,fixed)]
    # Keep a constrained slit and a single supporting row on either side.
    # All new rest and live fields use the same material barycentrics.
    points = np.vstack((line,line+[-.85,0.],line+[.85,0.]))
    offset = len(kept)
    chart_points = np.vstack((chart[kept],points))
    spine = offset + np.arange(len(line))
    limits = np.searchsorted(kept,boundary)
    # Preserve the character's existing triangulation away from the slit.
    # Flipping even a valid chart diagonal on a curved thigh can cut into skin.
    old_edges=edges[np.isin(edges,kept).all(1)]
    coords=chart[old_edges]
    cross=coords[:,:,0]-seam_bend(coords[:,:,1],phase)
    outside=(cross.min(1)>1.45)|(cross.max(1)<-1.45)
    outside|=(coords[:,:,1].max(1)<station[0]-.35)|(coords[:,:,1].min(1)>station[-1]+.35)
    constraints=np.unique(np.vstack((limits,np.searchsorted(kept,old_edges[outside]),
                          np.column_stack((spine[:-1],spine[1:])))),axis=0)
    _,_,result,mapping,_,_ = delaunay_2d_cdt(chart_points.tolist(),constraints.tolist(),[],0,1e-7)
    # A chart overlap or an intersecting cavity boundary cannot represent a single sheet.
    if any(len(m)!=1 for m in mapping) or len(mapping)!=len(chart_points):
        return None
    indices = np.array([m[0] for m in mapping])
    tris = indices[np.asarray(result,dtype=int)]
    interior = locate(chart_points[tris].mean(1))[0]
    all_tri=tris
    tris=all_tri[interior]
    common=np.sort(tris[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
    boundary_ids,uses=np.unique(common,axis=0,return_counts=True)
    if set(map(tuple,boundary_ids[uses==1])) != set(map(tuple,np.sort(limits,axis=1))):
        interior=inside_boundary(all_tri,limits)
        if interior is None:return None
        tris=all_tri[interior]
    # CDT is counterclockwise in (load, run). The source can use either winding.
    if np.median(np.linalg.det(matrix[valid]))<0:
        tris=tris[:,[0,2,1]]
    inside,parents,weights = locate(points)
    if not inside.all():
        return None
    count = len(line)
    start = len(sim.rest)
    source = np.r_[kept,np.arange(start,start+len(points))]
    # Each interior station has a vertex for each bank, both with identical UVs.
    first = source[spine]
    duplicate = start+len(points)+np.arange(count-2)
    rails = np.column_stack((first,first))
    rails[1:-1,1]=duplicate
    coord = chart_points[tris].mean(1)
    right = coord[:,0]>np.interp(coord[:,1],station,line[:,0])
    result=source[tris]
    lut=np.arange(start+len(points)+count-2)
    lut[first[1:-1]]=duplicate
    result[right]=lut[result[right]]
    merged=np.vstack((faces[~selected],result)).astype(np.int32)
    return parents,weights,first,merged,rails,station


def cut_strip(sim,center,uv,load,phase):
    from . import warned_lane
    ready=warned_lane.get(sim,uv,load)
    plan,patch,packet=ready if ready is not None else (plan_strip(sim,center,uv,load,phase),None,None)
    if plan is None:return None
    if patch is not None and getattr(sim,'predicting_birth',False):
        from copy import copy
        patch=copy(patch);patch.loops=copy(patch.loops)
    parents,weights,first,faces,rails,station=plan
    start=len(sim.rest)
    if packet is not None and sim.pose is None and getattr(sim,'monitor_interpolate_faces',None) is None and getattr(sim,'monitor_interpolate',None) is None:
        # The prepared material vertices and chart depend only on this exact
        # strip. Interpolate the three live fields at their current displacement.
        for name in ('pos','opening','seating'):
            values=getattr(sim,name)
            added=np.einsum('ni,nij->nj',weights,values[parents])
            setattr(sim,name,np.concatenate((values,added,added[first[1:-1]-start])))
        ref=getattr(sim,'cut_reference',None)
        if ref is not None:
            added=np.einsum('nc,ncij->nij',weights,ref[parents])
            sim.cut_reference=np.concatenate((ref,added,added[first[1:-1]-start]))
        sim.rest,sim.uv=packet[6],packet[11]
        weave=getattr(sim,'bank_weave',None)
        if weave is not None:
            added=np.einsum('ni,nij->nj',weights,weave[parents]).astype('f')
            sim.bank_weave=np.concatenate((weave,added,added[first[1:-1]-start]))
            sim.bank_rest_owner=sim.rest
    else:
        sim.append_material_samples(parents,weights)
        sim.interpolate_vertices(first[1:-1],first[1:-1],np.zeros(len(first)-2))
    sim.faces=faces
    seating=None if packet is None else packet[9]
    if seating is None:seat_strip(sim,faces[(faces>=start).any(1)])
    else:seat_strip(sim,None,seating)
    sim.prepared_patch=patch
    sim.draw_packet=packet
    warned_lane.cancel(sim)
    return rails,station
