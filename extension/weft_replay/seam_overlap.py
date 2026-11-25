"""Restore a colliding stitch fan's local shape while retaining its travel."""
import numpy as np
from mathutils.bvhtree import BVHTree
from .open_seam import smooth
from .knit_patch import surface_normals
from .seam_contact import clear_skin
from .contact_index import overlap_candidates,repair_fans


def intersections(points,faces,active=None):
    candidates=overlap_candidates(points,faces,active)
    if candidates is not None and len(candidates)<2:return np.empty((0,2),dtype='i')
    tiles=faces if candidates is None else faces[candidates]
    vertices,compact=np.unique(tiles,return_inverse=True)
    tree=BVHTree.FromPolygons(points[vertices].tolist(),compact.reshape(-1,3).tolist(),all_triangles=True)
    pairs=np.asarray(tree.overlap(tree),dtype='i').reshape(-1,2)
    pairs=pairs[pairs[:,0]<pairs[:,1]]
    if not len(pairs):return pairs
    if candidates is not None:pairs=candidates[pairs]
    if active is not None:
        pairs=pairs[active[faces[pairs]].any((1,2))]
        if not len(pairs):return pairs
    used,inverse=np.unique(faces[pairs],return_inverse=True)
    keys=np.unique(np.round(points[used],6),axis=0,return_inverse=True)[1]
    a,b=keys[inverse.reshape(-1,2,3)].transpose(1,0,2)
    return pairs[~(a[:,:,None]==b[:,None,:]).any((1,2))]


class LoadedContact:
    """Transport almost touching, unwelded dry courses under one local strain."""
    def __init__(self,neutral,faces):
        from .contact_index import close_courses
        nearby=close_courses(neutral,faces,.00010)
        self.faces=faces if nearby is None else faces[nearby]
        self.known=set(map(tuple,intersections(neutral,self.faces))) if len(self.faces) else set()
        self.patches=[]

    def patch(self,reference,vertices,slope=1.):
        # One local affine motion preserves separate layers through a fold.
        # Feather it outside the measured contact so the fabric still curves
        # around the hand. Fit on all three axes with a regularised thin axis.
        center=reference[vertices].mean(0)
        dist=np.linalg.norm(reference-center,axis=1)
        core=dist[vertices].max()+.005
        region=np.flatnonzero(dist<core+.008)
        offset=reference[region]-center
        weight=smooth((core+.008-dist[region])/.008)[:,None]
        basis=np.column_stack((np.ones(len(region)),offset/.002))
        gram=basis.T@basis+np.diag([0.,.015,.015,.015])
        fit=np.linalg.solve(gram,basis.T)
        return vertices,center,core,slope,region,weight,basis,fit

    def transport(self,points,neutral,requested):
        points[:]=requested
        for _,_,_,slope,region,weight,basis,fit in self.patches:
            delta=requested[region]-neutral[region]
            coefficients=fit@delta
            coefficients[1:]*=slope
            dest=neutral[region]+basis@coefficients
            points[region]+=weight*(dest-points[region])

    def correct(self,points,neutral):
        if not len(self.faces):return
        requested=points.copy()
        if self.patches:self.transport(points,neutral,requested)
        for _ in range(6):
            pairs=[pair for pair in intersections(points,self.faces) if tuple(pair) not in self.known]
            if not pairs:return
            cells=self.faces[np.asarray(pairs)].reshape(-1,6)
            centers=neutral[cells].mean(1)
            changed=False
            while len(cells):
                group=np.linalg.norm(centers-centers[0],axis=1)<.010
                vertices=np.unique(cells[group])
                center=neutral[vertices].mean(0)
                old=[patch for patch in self.patches if np.linalg.norm(patch[1]-center)<patch[2]+.006]
                cells,centers=cells[~group],centers[~group]
                slope=1.
                if old:
                    vertices=np.unique(np.concatenate([vertices]+[patch[0] for patch in old]))
                    if len(old)==1 and np.array_equal(vertices,old[0][0]):
                        if old[0][3]==0.:continue
                        slope=max(0.,old[0][3]-.5)
                    superseded={id(patch) for patch in old}
                    self.patches=[patch for patch in self.patches if id(patch) not in superseded]
                self.patches.append(self.patch(neutral,vertices,slope));changed=True
            if not changed:return
            self.transport(points,neutral,requested)


def paid_out_reference(material,requested,previous,tri):
    # Line search on the newly paid-out displacement. Both banks may have
    # occupied the same position in the factory mesh. That coincident rest
    # shape cannot restore the shape of a stitch which is already open.
    if previous is not None and not len(intersections(previous,tri)):
        return previous,True
    start=material if previous is None else previous
    travel=requested-start
    for fraction in (.5,.25,.125,.0625):
        candidate=start+fraction*travel
        if not len(intersections(candidate,tri)):return candidate,True
    return material,False


def settle_paid_out(base,target,tri,previous):
    """Advance the paid-out rest state up to contact with the neighbouring sheet."""
    if previous is None:return uncross(base,target,tri,released_from=(base,None))
    if not len(intersections(target,tri)):return False
    if len(intersections(previous,tri)):return uncross(base,target,tri,released_from=(base,previous))
    # A paid-out loop can transfer only the motion which the surrounding
    # cloth can accommodate. Keep the remaining supply in the stitch ledger.
    # The limit acts on displacement since the last accepted open state.
    travel=target-previous
    for factor in (.75,.5,.25,.125,.0625,.03125):
        trial=previous+factor*travel
        if not len(intersections(trial,tri)):
            target[:]=trial
            return True
    target[:]=previous
    return True


def uncross(reference,points,tri,*,released_from=None,initial_pairs=None):
    corrected=False
    verified=False
    active=None
    for attempt in range(7):
        if attempt==0 and initial_pairs is not None:
            pairs=initial_pairs
        elif active is None:
            pairs=intersections(points,tri)
        else:
            # Only a changed triangle can create a new intersection. Keep its
            # stationary neighbours in the same refitted material-space tree.
            pairs=intersections(points,tri,active)
        seed=np.unique(tri[pairs])
        if corrected:
            a,b,c=points[tri].transpose(1,0,2)
            signed=np.einsum('ij,ij->i',np.cross(b-a,c-a),axes)
            seed=np.union1d(seed,tri[(area2>1e-24)&(signed<.004*area2)])
        if not len(seed):break
        if not corrected:
            if released_from is not None:
                material,previous=released_from
                reference,verified=paid_out_reference(material,points,previous,tri)
            a,b,c=reference[tri].transpose(1,0,2)
            axes=np.cross(b-a,c-a);area2=np.einsum('ij,ij->i',axes,axes)
        corrected=True
        active=np.zeros(len(points),dtype=bool)
        if repair_fans(points,reference,seed,attempt,active):continue
        active[seed]=True
        # Each group returns to a translated copy of its original small
        # patch. The banks outside it keep their full opening displacement.
        while len(seed):
            group=np.array([seed[0]]);other=seed[1:]
            for _ in range(12):
                if not len(other):break
                center=points[group].mean(0)
                radius=np.linalg.norm(points[group]-center,axis=1).max(initial=0.)+.004
                select=np.linalg.norm(points[other]-center,axis=1)<radius
                if not select.any():break
                group=np.r_[group,other[select]];other=other[~select]
            seed=other
            d=points-reference
            shift=np.median(d[group],axis=0)
            x=points[group].mean(0);y=reference[group].mean(0)
            rx=np.linalg.norm(points[group]-x,axis=1).max(initial=0.)
            ry=np.linalg.norm(reference[group]-y,axis=1).max(initial=0.)
            distance=np.maximum(np.linalg.norm(points-x,axis=1)-rx,
                                np.linalg.norm(reference-y,axis=1)-ry)
            weight=1.-smooth((distance-.0008)/(.006+.001*attempt))
            active|=weight>0.
            points+=weight[:,None]*(shift-d)
    if corrected and released_from is not None and len(intersections(points,tri)):
        # Anchor the remaining fold to the last admissible position. This also
        # bounds a recovery substep when two independently moving fans cross.
        safe=reference
        if verified or not len(intersections(safe,tri)):
            delta=points-safe
            for factor in (.75,.5,.25,.125):
                trial=safe+factor*delta
                if not len(intersections(trial,tri)):
                    points[:]=trial
                    break
            else:points[:]=safe
    return corrected


def correct(sheet,*,released,move_held=True):
    cut=sheet.plan
    if cut is None or not cut.copies:return
    if getattr(cut,'overlap_stage',None)!=cut.reached:
        ids,local=np.unique(sheet.faces[cut.face_ids],return_inverse=True)
        source=ids.copy()
        for k,new in cut.copies.items():source[source==new]=cut.path[k]
        tri=local.reshape(-1,3);base=cut.base[source]
        a,b,c=base[tri].transpose(1,0,2)
        normal=np.cross(b-a,c-a)
        support=sheet.support_normals[source][tri].mean(1)
        cut.overlap_domain=(ids,tri,base,normal,support,sheet.support_normals[source])
        cut.overlap_stage=cut.reached
    ids,tri,before,normal,support,vertex_normals=cut.overlap_domain
    if released:
        neutral=sheet.rest[ids]
        change=uncross(before,neutral,tri)
        # Completed cuts have already checked the whole skin contact band
        # on private arrays. Recheck the triangles touched by uncrossing.
        contact_tri=tri
        if cut.done:
            affected=np.linalg.norm(neutral-sheet.rest[ids],axis=1)>.000002
            contact_tri=tri[affected[tri].any(1)]
        if len(contact_tri) and clear_skin(sheet,before,neutral,contact_tri,vertex_normals):
            change=True
            if uncross(before,neutral,tri):clear_skin(sheet,before,neutral,tri,vertex_normals)
        if change:
            if move_held:sheet.pos[ids]+=neutral-sheet.rest[ids]
            sheet.rest[ids]=neutral
            sheet.normals=surface_normals(sheet.rest,sheet.faces)
        cut.checked_rest=cut.reached
    else:
        held=sheet.pos[ids]
        # On a moving hand only a folded/strongly compressed fan or nearby
        # geometry can newly intersect the same side of the stocking. Test
        # the triangles overlapping those fans' spatial boxes. Release
        # still checks the full band including both free banks.
        p=held[tri];a,b,c=p.transpose(1,0,2)
        cross=np.cross(b-a,c-a)
        projected=np.einsum('ij,ij->i',normal,support)
        outward=np.einsum('ij,ij->i',cross,support)
        oriented=np.einsum('ij,ij->i',cross,normal)
        hot=((projected>1e-11)&(outward<.21*projected))|(oriented<.16*(normal*normal).sum(1))
        if not hot.any():return
        lo=p.min(1);hi=p.max(1);active=np.zeros(len(tri),bool)
        group,inverse=np.unique(np.floor(p[hot].mean(1)/.012).astype('i'),axis=0,return_inverse=True)
        suspect=np.flatnonzero(hot)
        for i in range(len(group)):
            f=suspect[inverse==i]
            low=lo[f].min(0)-.003
            high=hi[f].max(0)+.003
            active|=((lo<high)&(hi>low)).all(1)
        sub,faces=np.unique(tri[active],return_inverse=True)
        compact=held[sub]
        if uncross(sheet.rest[ids[sub]],compact,faces.reshape(-1,3)):sheet.pos[ids[sub]]=compact
