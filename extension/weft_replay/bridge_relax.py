"""Unloaded tension on the two free sides of an existing fabric bridge."""
import numpy as np
from .seam_overlap import intersections


def preserve_open(previous,points,left,right):
    # A paid bank can recoil, but one movement of the front must not erase the
    # already resolved clearance of a stitch. New pairs have zero prior width.
    before=previous[right]-previous[left]
    length=np.linalg.norm(before,axis=1)
    along=np.einsum('ij,ij->i',points[right]-points[left],before)/np.maximum(length,1.e-12)
    keep=np.minimum(.70*length,.0023)
    closing=(length>.0002)&(along<keep)
    if not closing.any():return False
    step=np.min((length[closing]-keep[closing])/(length[closing]-along[closing]))
    points[:]=previous+step*(points-previous)
    return True


def unfold(points,tri,source,dest,sheet,normals):
    """Slide a crossed bank into its own connected triangles, including legacy folded fans."""
    from .seam_contact import clear_skin
    pairs=intersections(points,tri)
    if not len(pairs):return True
    count=np.maximum(1,np.bincount(dest,minlength=len(points)))[:,None]
    for _ in range(16):
        affected=np.zeros(len(points),bool);affected[tri[pairs].ravel()]=True
        halo=affected.copy();halo[dest[affected[source]]]=True
        origin=points.copy()
        for _ in range(3):
            mean=np.column_stack([np.bincount(dest,points[source,j],minlength=len(points)) for j in range(3)])/count
            delta=mean[halo]-points[halo]
            weight=np.where(affected[halo],.50,.20)
            weight=np.minimum(weight,.00045/np.maximum(np.linalg.norm(delta,axis=1),1.e-12))
            points[halo]+=weight[:,None]*delta
        clear_skin(sheet,origin,points,tri,normals)
        pairs=intersections(points,tri)
        if not len(pairs):return True
    return False


def retract(cut, ids, points, triangles, fixed, source, dest, weights, diag):
    # Each oriented broken edge releases the in-plane line tension into its
    # own connected fan. This remains transverse where the knit path turns.
    steps=np.arange(cut.reached+1)
    left=np.searchsorted(ids,cut.path[steps])
    right=np.searchsorted(ids,[cut.copies.get(int(k),cut.path[k]) for k in steps])
    edge=np.diff(points[left],axis=0)
    size=np.maximum(np.linalg.norm(edge,axis=1,keepdims=True),1.e-12)
    tangent=edge/size
    edges=triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    third=triangles[:,[2,0,1]].ravel()
    codes=edges.min(1).astype(np.int64)*len(ids)+edges.max(1)
    order=np.argsort(codes);codes=codes[order]
    tension=.5*np.clip(sum(cut.widths)/cut.length,.3,1.4)
    load=np.zeros_like(points)
    for side in (left,right):
        a,b=side[:-1],side[1:]
        keys=np.minimum(a,b).astype(np.int64)*len(ids)+np.maximum(a,b)
        rim=points[third[order[np.searchsorted(codes,keys)]]]-points[a]
        inward=rim-np.sum(rim*tangent,axis=1,keepdims=True)*tangent
        force=(tension*size/np.maximum(np.linalg.norm(inward,axis=1,keepdims=True),1.e-12))*inward
        np.add.at(load,side[:-1],force);np.add.at(load,side[1:],force)
    terminal=cut.done and cut.length<.009
    if terminal or cut.rolled_start:
        # A short ligament still carries the tension of its two old free rims.
        _,first,uses=np.unique(codes,return_index=True,return_counts=True)
        border=edges[order[first[uses==1]]]
        pull=.25*min(cut.widths)*(1.-cut.length/.009)
        if not terminal:
            # The first broken link frees two pre-tensioned shoulders of the
            # old rolled rim before the cut travels out into the ligament.
            front=cut.stations[cut.reached]
            remaining=1.-np.clip((front-.009)/.018,0.,1.)
            pull=.8*min(cut.widths)*min(1.,front/.004)*remaining
        for side in (left,right):
            ends=[(side[0],side[1])]
            if terminal:ends.append((side[-1],side[-2]))
            for v,inside in ends:
                last=int(inside);at=int(v);arc=0.
                for _ in range(20):
                    hits=border[(border[:,0]==at)|(border[:,1]==at)].ravel()
                    follow=hits[(hits!=at)&(hits!=last)]
                    if len(follow)!=1:break
                    last,at=at,int(follow[0])
                    arc+=np.linalg.norm(points[at]-points[last])
                    if arc>=(.004 if terminal else .009):break
                direction=points[at]-points[v]
                load[v]+=pull*direction/max(np.linalg.norm(direction),1.e-12)
    load[fixed]=0.
    delta=np.zeros_like(points);residual=load.copy()
    mass=np.maximum(diag,1.e-12)[:,None]
    z=residual/mass;direction=z.copy();energy=(z*residual).sum()
    for _ in range(130):
        if energy<1.e-15:break
        neighbour=np.column_stack([np.bincount(dest,weights*direction[source,j],minlength=len(ids)) for j in range(3)])
        grad=mass*direction-neighbour;grad[fixed]=0.
        step=energy/max((direction*grad).sum(),1.e-25)
        delta+=step*direction;residual-=step*grad;z=residual/mass
        next_energy=(residual*z).sum()
        direction=z+(next_energy/energy)*direction;energy=next_energy
    return points+delta
