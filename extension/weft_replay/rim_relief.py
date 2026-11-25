"""Redistribute compressed courses on the fabric side of an unloaded dry rim."""
import numpy as np
from mathutils.kdtree import KDTree
from .knit_patch import surface_normals
from .seam_overlap import intersections


def admit(p,trial,tri,src,dst,normals):
    a,b,c=p[tri].transpose(1,0,2)
    skin=normals[tri].mean(1);normal=np.cross(b-a,c-a)
    signed=(normal*skin).sum(1);area=np.linalg.norm(normal,axis=1)
    limit=np.minimum(.065*area,signed-.004*area)
    existing=intersections(p,tri)
    known=set(map(tuple,existing.tolist()))
    delta=trial-p;factor=np.ones(len(p))
    for attempt in range(48):
        a,b,c=trial[tri].transpose(1,0,2)
        bad=np.flatnonzero((np.cross(b-a,c-a)*skin).sum(1)<limit-1.e-16)
        if not len(bad) or attempt in (5,11,18) or attempt>=24:
            extra=[ij for ij in intersections(trial,tri).tolist() if tuple(ij) not in known]
            if extra:bad=np.union1d(bad,np.ravel(extra))
        if not len(bad):return True
        loss=np.zeros(len(p));loss[tri[bad]]=1.
        for _ in range(3):
            spread=np.zeros_like(loss);np.maximum.at(spread,dst,loss[src])
            loss=np.maximum(loss,.66*spread)
        factor*=1.-(.44 if attempt<15 else 1.)*loss
        trial[:]=p+factor[:,None]*delta
    return False


def fair_boundaries(points,knit,rims,pitch,influence,ids,open_ends):
    from .open_seam import closest_tips
    pinned=[]
    for i,one in enumerate(rims):
        for two in rims[i+1:]:
            a,b=closest_tips(points,knit,one,two)
            if np.linalg.norm(points[a]-points[b])<.065:pinned.extend((a,b))
    result=np.zeros((len(ids),3))
    for loop in rims:
        if int(loop[0]) in open_ends:continue
        local=np.searchsorted(ids,loop);curr=points[loop]
        du=np.diff(knit[np.r_[loop,loop[0]]],axis=0);du[:,0]-=144.*np.round(du[:,0]/144.)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(du*pitch,axis=1))]
        if arc[-1]<.020:continue
        pos=np.vstack((curr,curr[:1]))
        # Sample at equal intervals in the original knit, not at equal vertex
        # counts. A densely subdivided turn must not attract the entire rim.
        span=.011;offset=np.linspace(-span,span,15)
        weights=np.exp(-.5*(offset/(.48*span))**2);weights/=weights.sum()
        at=(arc[:-1,None]+offset)%arc[-1]
        target=np.column_stack([np.interp(at,arc,pos[:,j])@weights for j in range(3)])
        change=target-curr
        fade=influence[local].copy()
        if pinned:fade*=np.clip((distances(curr,points[pinned])-.005)/.004,0.,1.)
        scale=np.minimum(.9,.0034/np.maximum(np.linalg.norm(change,axis=1),1.e-12))
        result[local]=change*(fade*scale)[:,None]
    return result


def distances(points,seeds):
    tree=KDTree(len(seeds))
    for i,p in enumerate(seeds):tree.insert(p,i)
    tree.balance()
    return np.fromiter((tree.find(p)[2] for p in points),float,len(points))


def smooth_joined_banks(points,faces,rings,skin,protected,corridors):
    """Ease short sawtooth turns on a newly joined free edge, once per join."""
    if not rings:return False
    boundary=np.unique(np.concatenate([ring for ring,_ in rings]))
    active=np.zeros(len(points),bool);active[boundary]=True
    for _ in range(4):
        active[faces[active[faces].any(1)].ravel()]=True
    ids,tri=np.unique(faces[active[faces].any(1)],return_inverse=True)
    tri=tri.reshape(-1,3);p=points[ids]
    edges=np.unique(np.sort(tri[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    a,b=edges.T;src,dst=np.r_[a,b],np.r_[b,a]
    norm=surface_normals(p,tri)
    if skin is not None and skin.index is not None:
        hit,near=skin.index.nearest(p)
        valid=(hit>=0)&(near[:,6]<.016);norm[valid]=near[valid,3:6]
    length=np.linalg.norm(p[dst]-p[src],axis=1)
    w=1./np.maximum(length,.0003)
    total=np.maximum(np.bincount(dst,w,minlength=len(p)),1.)
    def mean(field):
        return np.column_stack([np.bincount(dst,w*field[src,j],minlength=len(p)) for j in range(3)])/total[:,None]
    toward_cloth=mean(p)-p
    disp=np.zeros_like(p);pinned=np.zeros(len(p),bool)
    for ring,open_arc in rings:
        ix=np.searchsorted(ids,ring);line=p[ix];cycle=line if open_arc else np.vstack((line,line[0]))
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(cycle,axis=0),axis=1))]
        if arc[-1]<.025:continue
        def filter_at(width):
            offset=np.linspace(-2.*width,2.*width,13)
            weight=np.exp(-.5*(offset/width)**2);weight/=weight.sum()
            stations=arc if open_arc else arc[:-1]
            at=stations[:,None]+offset
            at=np.clip(at,0.,arc[-1]) if open_arc else at%arc[-1]
            return np.column_stack([np.interp(at,arc,cycle[:,j])@weight for j in range(3)])
        narrow=filter_at(.0023)-line;broad=filter_at(.0058)-line
        # Remove the broad turn of a rounded lobe from the correction.
        shift=narrow-(.0023/.0058)**2*broad
        shift-=norm[ix]*(shift*norm[ix]).sum(1)[:,None]
        tangent=np.roll(line,-1,axis=0)-np.roll(line,1,axis=0)
        lateral=np.cross(tangent,norm[ix])
        lateral/=np.maximum(np.linalg.norm(lateral,axis=1,keepdims=True),1.e-12)
        lateral*=np.sign((lateral*toward_cloth[ix]).sum(1))[:,None]
        # A sharp protrusion may retract into the stocking. Preserve the
        # existing clear opening when a trough would move the opposite way.
        shift-=lateral*np.minimum(0.,(shift*lateral).sum(1)+.00009)[:,None]
        gain=np.minimum(.94,.00125/np.maximum(np.linalg.norm(shift,axis=1),1.e-12))
        if open_arc:gain*=np.clip((np.minimum(arc,arc[-1]-arc)-.003)/.004,0.,1.)
        disp[ix]=gain[:,None]*shift;pinned[ix]=True
    if not pinned.any():return False
    for _ in range(7):
        step=mean(disp);disp[~pinned]=.80*step[~pinned]
    disp[~active[ids]]=0.
    if len(protected):disp*=np.clip((distances(p,points[protected])-.003)/.005,0.,1.)[:,None]
    # The remaining bridge and its root define the next grip and its last
    # paid station. Aesthetic relaxation must not advance that stopping point.
    for a,b in corridors:
        edge=points[b]-points[a]
        t=np.clip(((p-points[a])@edge)/max(edge@edge,1.e-14),0.,1.)
        distance=np.linalg.norm(p-(points[a]+t[:,None]*edge),axis=1)
        disp*=np.clip((distance-.012)/.007,0.,1.)[:,None]
    trial=p+disp
    if skin is not None:
        moved=np.linalg.norm(disp,axis=1)>.000012
        if moved.any():trial[moved]=skin.seat(p[moved],trial[moved],norm[moved])
    if not admit(p,trial,tri,src,dst,norm):return False
    points[ids]=trial
    return True


def relax(points,faces,knit,rims,skin,open_ends=(),protected=(),previous=None):
    if not len(rims):return False
    boundary=np.unique(np.concatenate(rims))
    lo=points[boundary].min(0)-.036;hi=points[boundary].max(0)+.036
    box=((points>=lo)&(points<=hi)).all(1)
    rim_dist=np.full(len(points),np.inf)
    rim_dist[box]=distances(points[box],points[boundary])
    face_ids=np.flatnonzero((rim_dist[faces]<.029).any(1))
    if not len(face_ids):return False
    ids,tri=np.unique(faces[face_ids],return_inverse=True);tri=tri.reshape(-1,3)
    p=points[ids];chart=knit[ids][tri]
    uv=chart[:,1:]-chart[:,:1];uv[:,:,0]-=144.*np.round(uv[:,:,0]/144.)
    det=uv[:,0,0]*uv[:,1,1]-uv[:,0,1]*uv[:,1,0]
    valid=np.abs(det)>1.e-7
    if not valid.any():return False
    derivative=np.zeros((len(tri),2,3))
    derivative[valid]=np.linalg.solve(uv[valid],p[tri[valid,1:]]-p[tri[valid,:1]])
    lengths=np.linalg.norm(derivative,axis=2)
    far=valid&(rim_dist[ids][tri].mean(1)>.014)
    pitch=np.median(lengths[far if far.any() else valid],axis=0).clip(.0003,.012)
    axes=derivative/pitch[None,:,None]
    eigen=np.linalg.eigvalsh(axes@axes.transpose(0,2,1)).clip(0.)
    damaged=valid&(rim_dist[ids][tri].min(1)<.012)&(eigen[:,0]<.43**2)&(eigen[:,0]<.14*eigen[:,1])
    if not damaged.any():return False
    near=distances(p,p[np.unique(tri[damaged])])
    influence=np.clip((.018-near)/.010,0.,1.)
    influence=influence**2*(3.-2.*influence)
    edges=np.unique(np.sort(tri[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    a,b=edges.T;src,dst=np.r_[a,b],np.r_[b,a]
    axis=surface_normals(p,tri)
    if skin is not None and skin.index is not None:
        hit,info=skin.index.nearest(p)
        ok=(hit>=0)&(info[:,6]<.020);axis[ok]=info[ok,3:6]
    stock_normals=axis.copy()
    degree=np.maximum(np.bincount(dst,minlength=len(p)),1)
    for _ in range(3):
        mean=np.column_stack([np.bincount(dst,axis[src,j],minlength=len(p)) for j in range(3)])/degree[:,None]
        axis=.5*(axis+mean);axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1.e-12)
    geom=influence.copy()
    if len(protected):
        geom*=np.clip((distances(p,points[protected])-.004)/.005,0.,1.)
    accepted=p.copy()
    if skin is not None:
        height=skin.clearance(p,axis)
        band=(rim_dist[ids]>.010)&(height>.00005)&(height<.003)
        if band.any():
            stock=np.median(height[band])
            relief=stock+.00016*np.exp(-.5*(rim_dist[ids]/.0012)**2)
            change=np.where(abs(height)<.015,np.clip(relief-height,-.0035,.0025),0.)
            trial=p+(geom*change)[:,None]*axis
            if admit(p,trial,tri,src,dst,axis):accepted=trial
    # Move the surrounding fabric with a rounded turn. Nearby samples must
    # share the same displacement even when their tessellation density differs.
    # Each previously rounded neighbourhood has already shed its excess turn.
    fair=geom if previous is None else geom*(1.-np.clip(previous[ids],0.,1.))
    boundary_goal=fair_boundaries(points,knit,rims,pitch,fair,ids,open_ends)
    acc=np.zeros_like(p);mass=np.zeros(len(p))
    for bank in rims:
        ix=np.searchsorted(ids,bank);rim=p[ix]
        length=.5*(np.linalg.norm(rim-np.roll(rim,1,axis=0),axis=1)+np.linalg.norm(rim-np.roll(rim,-1,axis=0),axis=1))
        for k in range(0,len(ix),32):
            sub=ix[k:k+32]
            d=p[:,None]-p[sub];dist2=np.einsum('ijk,ijk->ij',d,d)
            weights=np.exp(-dist2/(2.*.0035**2))*length[None,k:k+32]
            acc+=weights@boundary_goal[sub];mass+=weights.sum(1)
    shift=acc/(.001+mass[:,None])
    trial=accepted+shift*geom[:,None]
    if skin is not None:
        moved=np.linalg.norm(trial-accepted,axis=1)>.000015
        if moved.any():trial[moved]=skin.seat(accepted[moved],trial[moved],axis[moved])
    if not admit(accepted,trial,tri,src,dst,axis):trial=accepted
    points[ids]=trial
    slip=np.clip((.029-near)/.020,0.,1.)
    feather=np.clip((rim_dist[ids]-.00045)/.0025,0.,1.)
    shoulder=np.exp(-.5*(rim_dist[ids]/.009)**2)*feather**2*(3.-2.*feather)
    return ids, slip**2*(3.-2.*slip), .90*shoulder*influence,stock_normals*(.98*influence[:,None])
