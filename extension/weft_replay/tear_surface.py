"""Fair the fabric supporting each open rim while keeping the cut boundary fixed."""
import numpy as np

from .fracture import chart_delta, rim_slice, seam_bend
from .contact_index import constraint_projection


def smoothstep(x):
    x = np.clip(x,0.,1.)
    return x*x*(3.-2.*x)


def strain_frame(rest, faces):
    triangle=rest[faces]
    lengths=np.linalg.norm(np.roll(triangle,-1,axis=1)-triangle,axis=2)
    longest=lengths.argmax(1)
    ids=np.take_along_axis(faces,(longest[:,None]+np.arange(3))%3,axis=1)
    a,b,c=rest[ids].transpose(1,0,2)
    edge=b-a
    length=np.linalg.norm(edge,axis=1)
    direction=edge/np.maximum(length[:,None],1e-12)
    t=np.einsum('ij,ij->i',c-a,direction)/np.maximum(length,1e-12)
    normal=np.cross(direction,c-a)
    height=np.linalg.norm(normal,axis=1)
    good=height>1e-8
    faces=ids[good]
    corners=faces.ravel()
    order=np.argsort(corners,kind='stable')
    starts=np.r_[0,np.cumsum(np.bincount(corners,minlength=len(rest)))]
    return faces,length[good],t[good],height[good],normal[good]/height[good,None],(starts,order//3)


def incident_faces(vertices,incidence):
    starts,faces=incidence
    begin=starts[vertices]
    count=starts[vertices+1]-begin
    ends=np.cumsum(count)
    # Gather the ragged list for each moved corner, in the same face order
    # used by the initial full projection.
    slots=np.arange(ends[-1])+np.repeat(begin+count-ends,count)
    return np.unique(faces[slots])


def limit_strain(points, frame):
    """Bound stretch of both the longest edge and the material altitude."""
    if constraint_projection(points,frame):return
    faces,length,t,height,normal,incidence=frame
    scope=slice(None)
    for _ in range(10):
        local=faces[scope]
        a,b,c=points[local].transpose(1,0,2)
        edge=b-a
        size=np.linalg.norm(edge,axis=1)
        lateral=c-a-t[scope,None]*edge
        altitude=np.linalg.norm(lateral,axis=1)
        across=np.cross(normal[scope],edge)
        across/=np.maximum(np.linalg.norm(across,axis=1)[:,None],1e-12)
        signed=(lateral*across).sum(1)
        strained=(altitude>2.65*height[scope])|(signed<.13*height[scope])|(size>2.65*length[scope])
        if not strained.any():
            break
        ids=local[strained]
        v=lateral[strained]
        h=height[scope][strained]
        n=across[strained]
        goal=v*np.minimum(1.,2.60*h/np.maximum(altitude[strained],1e-12))[:,None]
        goal+=np.maximum(0.,.16*h-(goal*n).sum(1))[:,None]*n
        f=t[scope][strained,None]
        corr=(goal-v)/(1.+(1.-f)**2+f*f)
        stretch=.5*np.maximum(0.,1.-2.60*length[scope][strained]/np.maximum(size[strained],1e-12))[:,None]*edge[strained]
        vertices,inverse=np.unique(ids,return_inverse=True)
        inverse=inverse.reshape(-1,3)
        adjust=np.zeros((len(vertices),3),dtype=points.dtype)
        np.add.at(adjust,inverse[:,0],stretch-(1.-f)*corr)
        np.add.at(adjust,inverse[:,1],-stretch-f*corr)
        np.add.at(adjust,inverse[:,2],corr)
        points[vertices]+=adjust/np.bincount(inverse.ravel())[:,None]
        scope=incident_faces(vertices,incidence)

    # A wide, unloaded double-lobed opening can leave small inverted fans
    # after the stretch projection. Recover their signed area, including
    # neighbouring faces that share a corrected corner.
    area=length*height
    scope=slice(None)
    for _ in range(48):
        local=faces[scope]
        tri=points[local]
        signed=np.einsum('ij,ij->i',np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),normal[scope])
        bad=signed<.065*area[scope]
        if not bad.any():
            break
        ids=local[bad]
        p=tri[bad]
        gradients=np.cross(np.roll(p,-1,axis=1)-np.roll(p,-2,axis=1),normal[scope][bad,None])
        scale=(.16*area[scope][bad]-signed[bad])/np.maximum(np.einsum('nij,nij->n',gradients,gradients),1e-15)
        vertices,inverse=np.unique(ids,return_inverse=True)
        delta=np.zeros((len(vertices),3))
        np.add.at(delta,inverse.ravel(),(gradients*scale[:,None,None]).reshape(-1,3))
        points[vertices]+=delta/np.bincount(inverse.ravel())[:,None]
        scope=incident_faces(vertices,incidence)


def limit_held_travel(points,released,frame):
    """Keep the local translation of a bank while easing its folded triangle fans."""
    if constraint_projection(points,frame,released):return
    faces,length,_,height,normal,incidence=frame
    minimum=.018*length*height
    scope=slice(None)
    for _ in range(32):
        local=faces[scope]
        tri=points[local]
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        bad=np.einsum('ij,ij->i',cross,normal[scope])<minimum[scope]
        if not bad.any():
            break
        # Translate each folded triangle with its own centroid. Sharing the
        # correction only through welded vertices keeps the other lip free.
        ids=local[bad]
        displacement=tri[bad]-released[ids]
        correction=displacement.mean(axis=1,keepdims=True)-displacement
        vertices,inverse=np.unique(ids,return_inverse=True)
        delta=np.zeros((len(vertices),3))
        np.add.at(delta,inverse.ravel(),correction.reshape(-1,3))
        points[vertices]+=.85*delta/np.bincount(inverse.ravel())[:,None]
        scope=incident_faces(vertices,incidence)


def limit_dry_held_travel(points,released,frame):
    """Keep a stretched course on its own side without moving both ends of a long edge."""
    if constraint_projection(points,frame,released,dry_hold=True):return
    faces,length,foot,height,normal,incidence=frame
    a,b,c=released[faces].transpose(1,0,2)
    neutral=np.einsum('ij,ij->i',np.cross(b-a,c-a),normal)
    # A newly opened lip may already have rotated in its unloaded state.
    # Constrain only the added hand displacement against each saved frame.
    minimum=np.minimum(.035*length*height,.35*neutral)
    usable=neutral>1.e-14
    scope=np.arange(len(faces))
    for _ in range(24):
        ids=faces[scope]
        a,b,c=points[ids].transpose(1,0,2)
        signed=np.einsum('ij,ij->i',np.cross(b-a,c-a),normal[scope])
        bad=usable[scope]&(signed<minimum[scope])
        if not bad.any():return
        selected=scope[bad];bank=ids[bad]
        start,end,tip=bank.T
        # The third corner of a sliver lies close to the longest edge.
        # Interpolate the hand displacement at its material-space foot.
        # Averaging all three corners moves even the two sound neighbours
        # and propagates a fold along otherwise intact courses.
        f=foot[selected,None]
        change=(1.-f)*(points[start]-released[start])+f*(points[end]-released[end])-(points[tip]-released[tip])
        gradient=np.cross(normal[selected],points[end]-points[start])
        residual=2.*minimum[selected]-signed[bad]-np.einsum('ij,ij->i',change,gradient)
        change+=gradient*(np.maximum(0.,residual)/np.maximum(1.e-20,np.einsum('ij,ij->i',gradient,gradient)))[:,None]
        vertices,inverse=np.unique(tip,return_inverse=True)
        correction=np.zeros((len(vertices),3))
        np.add.at(correction,inverse,change)
        points[vertices]+=correction/np.bincount(inverse)[:,None]
        scope=incident_faces(vertices,incidence)


def support_strips(rest, uv, faces, cracks):
    protected = np.unique(np.concatenate([c['rim_ids'].ravel() for c in cracks]))
    edges = faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    opposite = faces[:,[2,0,1]].ravel()
    strips = []
    for crack in cracks:
        load = crack['load_axis']
        chart = chart_delta(uv,crack['uv'])[:,[load,1-load]]
        phase = crack.get('curve_phase')
        if phase is not None:
            chart[:,0] -= seam_bend(chart[:,1],phase)
        suitable = ((abs(chart[:,0])>1e-7)&(abs(chart[:,0])<4.)&
                    (np.linalg.norm(rest-crack['center'],axis=1)<.033))
        suitable[protected] = False
        ids = np.flatnonzero(suitable)
        eligible = suitable[opposite]
        local_edges, local_vertices = edges[eligible], opposite[eligible]
        wedges = []
        for bank in (0,1):
            index = np.full(len(rest),-1,int)
            index[crack['rim_ids'][:,bank]] = np.arange(len(crack['rim_axis']))
            edge = np.sort(index[local_edges],axis=1)
            valid = (edge[:,0]>=0)&(edge[:,1]-edge[:,0]==1)
            wedges.append((local_vertices[valid],edge[valid]))
        strips.append((ids,chart[ids],wedges))
    return strips


def fair_rim(positions, rest, crack, strip, grip):
    """Move only the welded seam and its attached triangles outside the opening."""
    region = rim_slice(crack)
    rails = crack['rim_ids']
    axis = crack['rim_axis']
    if region.stop-region.start<3:
        return np.empty(0,int),np.empty((0,3))
    lo,hi = crack['fronts']
    opened = positions[rails[region]]
    start,end = opened[[0,-1]].mean(1)
    origin = .5*(start+end)
    normal = crack['normal']
    along = end-start
    along -= (along@normal)*normal
    length = np.linalg.norm(along)
    if length<1e-8 or hi<=lo:
        return np.empty(0,int),np.empty((0,3))
    along /= length
    across = np.cross(along,normal)
    if across@(opened[:,1]-opened[:,0]).mean(0)<0:
        across = -across
    frame = np.stack((across,along,normal))
    rail = (positions[rails]-origin)@frame.T

    # The intact seam just beyond a stretched tip must continue away from it.
    outside = np.maximum(lo-axis,axis-hi)
    welded = (outside>1e-7)&(outside<8.)
    welded[region] = False
    if grip is not None:
        welded &= ~np.isin(rails,grip).any(1)
    pitch = np.linalg.norm(rest[rails[region.start,0]]-rest[rails[region.stop-1,0]])/(hi-lo)
    y = rail[:,:,1].mean(1)
    limit = np.where(axis<lo,rail[region.start,:,1].mean()-.73*pitch*(lo-axis),
                     rail[region.stop-1,:,1].mean()+.73*pitch*(axis-hi))
    shift = np.where(axis<lo,np.minimum(limit-y,0.),np.maximum(limit-y,0.))
    rail[:,:,1] += (shift*smoothstep((8.-outside)/3.)*welded)[:,None]

    ids,chart,wedges = strip
    cross,span = chart.T
    active = (span>lo-7.)&(span<hi+7.)
    if grip is not None:
        active &= ~np.isin(ids,grip)
    ids,cross,span = ids[active],cross[active],span[active]
    side = np.sign(cross)
    point = (positions[ids]-origin)@frame.T
    banks = np.stack([np.column_stack([np.interp(span,axis,c) for c in rail[:,n].T])
                      for n in (0,1)])
    anchor = banks[(side>0).astype(int),np.arange(len(ids))]
    weight = smoothstep((4.-abs(cross))/1.5)*smoothstep((7.-np.maximum(lo-span,span-hi))/3.)
    point[:,0] += weight*side*np.maximum(abs(cross)*.00055-side*(point[:,0]-anchor[:,0]),0.)
    point[:,1:] += weight[:,None]*(anchor[:,1:]-point[:,1:])

    # Each fan apex must stay on the outside of its released rim edge. Slide
    # perpendicular to the edge, including for very short, steep segments.
    mapping = np.full(len(positions),-1,int)
    mapping[ids] = np.arange(len(ids))
    guards = []
    for bank,(vertices,edges) in enumerate(wedges):
        v = mapping[vertices]
        active_edges = (v>=0)&(edges[:,0]>=region.start)&(edges[:,1]<region.stop)
        first,last = rail[edges[active_edges],bank,:2].transpose(1,0,2)
        flow = last-first
        flow /= np.maximum(np.linalg.norm(flow,axis=1,keepdims=True),1e-8)
        outward = (2*bank-1)*np.column_stack((flow[:,1],-flow[:,0]))
        valid = flow[:,1]>.08
        guards.extend(zip(v[active_edges][valid],first[valid],outward[valid]))
    for _ in range(4):
        for vertex,edge,outward in guards:
            penetration = (edge-point[vertex,:2])@outward
            if penetration>-.000035:
                point[vertex,:2] += (penetration+.000035)*outward

    moved = np.r_[ids,rails[welded].ravel()]
    target = np.vstack((point,rail[welded].reshape(-1,3)))@frame+origin
    delta = target-positions[moved]
    limit = max(.005,.45*np.linalg.norm(opened[:,1]-opened[:,0],axis=1).max())
    delta *= np.minimum(1.,limit/np.maximum(np.linalg.norm(delta,axis=1),1e-9))[:,None]
    return moved,delta

def keep_area(reference, positions, faces, held=None, rim=None, *, crowded=False):
    """Project folded triangles back above a fraction of their resting area."""
    start = reference[faces]
    normal = np.cross(start[:, 1]-start[:, 0], start[:, 2]-start[:, 0])
    area = np.linalg.norm(normal, axis=1)
    valid = area > 1e-10
    faces, area, normal = faces[valid], area[valid], normal[valid]
    normal /= area[:, None]
    mobility = np.ones(len(positions))
    if rim is not None:
        mobility[rim] = .15
    if held is not None:
        mobility[held] = 0.
    for _ in range(20):
        triangle = positions[faces]
        cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
        shortfall = .12*area - np.einsum('ij,ij->i', cross, normal)
        bad = shortfall > 1e-10
        if not bad.any():
            break
        ids = faces[bad]
        p = triangle[bad]
        gradient = np.cross(np.roll(p, -1, axis=1)-np.roll(p, -2, axis=1), normal[bad, None])
        weights = mobility[ids]
        denom = np.einsum('nij,nij,ni->n', gradient, gradient, weights)
        step = shortfall[bad] / np.maximum(denom, 1e-15)
        delta = np.zeros_like(positions)
        np.add.at(delta, ids, gradient * (step[:, None] * weights)[..., None])
        count = np.bincount(ids.ravel(), minlength=len(positions))
        moved = count > 0
        positions[moved] += delta[moved] / count[moved, None]

    if crowded:
        # Tiny seam subdivisions between several holes can mutually block
        # the area projections. Reduce the local travel of those supports.
        for _ in range(12):
            triangle = positions[faces]
            cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
            bad = np.einsum('ij,ij->i', cross, normal) < .05*area
            ids = np.unique(faces[bad])
            ids = ids[mobility[ids] > 0.]
            if not len(ids):
                break
            scale = np.where(mobility[ids] < 1., .74, .60)[:, None]
            positions[ids] = reference[ids] + scale*(positions[ids]-reference[ids])


def limit_fairing(solved, positions, faces):
    """Keep the crease normals from reversing when fairing the cut support."""
    affected = np.any(positions != solved, axis=1)
    faces = faces[affected[faces].any(axis=1)]
    start = solved[faces]
    normal = np.cross(start[:, 1]-start[:, 0], start[:, 2]-start[:, 0])
    area = np.einsum('ij,ij->i', normal, normal)
    keep = area > 1e-20
    faces, normal, area = faces[keep], normal[keep], area[keep]
    while len(faces):
        triangle = positions[faces]
        cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
        bad = np.einsum('ij,ij->i', cross, normal) < .18*area
        if not bad.any():
            break
        # Restore this fan and retest its neighbors. No vertices outside the
        # original fairing support can be moved by this correction.
        positions[faces[bad]] = solved[faces[bad]]

def limit_intersections(settled, positions, faces, grip):
    """Soften only colliding cloth around a held bridge between tears."""
    from mathutils.bvhtree import BVHTree

    if not len(faces):
        return np.empty(0,dtype=int)
    ids, local = np.unique(faces,return_inverse=True)
    local = local.reshape(-1,3)
    free = ~np.isin(ids,grip)
    adjusted = np.zeros(len(ids),bool)
    reference = settled[ids]
    xyz = positions[ids]
    for _ in range(3):
        tree = BVHTree.FromPolygons(xyz.tolist(),local.tolist(),all_triangles=True)
        pairs = np.asarray(tree.overlap(tree),dtype=np.int32).reshape(-1,2)
        pairs = pairs[pairs[:,0]<pairs[:,1]]
        if not len(pairs):
            break
        corners = np.unique(np.round(xyz,6),axis=0,return_inverse=True)[1][local]
        a, b = corners[pairs].transpose(1,0,2)
        crossing = ~(a[:,:,None]==b[:,None,:]).any(axis=(1,2))
        touched = np.unique(local[pairs[crossing]])
        touched = touched[free[touched]]
        if not len(touched):
            break
        xyz[touched] = reference[touched]+.43*(xyz[touched]-reference[touched])
        adjusted[touched] = True
    positions[ids[adjusted]] = xyz[adjusted]
    return ids[adjusted]
