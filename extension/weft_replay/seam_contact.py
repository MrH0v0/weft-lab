"""Read-only skin geometry for a dry fixed pose. All queries use owned arrays."""
import numpy as np
from .contact_index import ContactIndex


class FixedSkin:
    def __init__(self,triangles):
        self.tri=np.ascontiguousarray(triangles,dtype=float)
        self.low=self.tri.min(1);self.high=self.tri.max(1)
        points=self.tri.reshape(-1,3)
        self.index=ContactIndex.create(points,np.arange(len(points),dtype='i').reshape(-1,3))
        self.bins={}

    def ray_gaps(self,points,normals):
        # Exact first front hit for tied or opposing nearest faces.
        # The small spatial bins are shared by adjacent courses of this grip.
        if self.index is not None:
            distance=self.index.first_hits(points+.014*normals,-normals,.025)
            return np.where(np.isfinite(distance),distance-.014,.025)
        values=np.full(len(points),.025)
        keys,inv=np.unique(np.floor(points/.020).astype('i'),axis=0,return_inverse=True)
        for i,key in enumerate(keys):
            slot=tuple(key)
            if slot not in self.bins:
                lo=key*.020-.015;hi=(key+1)*.020+.015
                self.bins[slot]=np.flatnonzero(((self.high>=lo)&(self.low<=hi)).all(1))
            tiles=self.bins[slot]
            if not len(tiles):continue
            a,b,c=self.tri[tiles].transpose(1,0,2)
            ab=b-a;ac=c-a
            group=np.flatnonzero(inv==i)
            for ix in np.array_split(group,max(1,(len(group)+39)//40)):
                d=-normals[ix,None];p=points[ix,None]-.014*d
                h=np.cross(d,ac);den=np.einsum('tq,btq->bt',ab,h)
                ok=abs(den)>1e-15
                f=np.divide(1.,den,out=np.zeros_like(den),where=ok)
                s=p-a;q=np.cross(s,ab)
                u=np.einsum('btq,btq->bt',s,h)*f
                v=np.einsum('bi,bti->bt',d[:,0],q)*f
                t=np.einsum('ti,bti->bt',ac,q)*f
                ok&=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1.0000001)&(t>=0)&(t<=.025)
                first=np.min(np.where(ok,t,np.inf),axis=1)
                values[ix]=np.where(np.isfinite(first),first-.014,.025)
        return values

    def clearance(self,points,normals):
        if not len(points):return np.empty(0)
        if self.index is None:return self.ray_gaps(points,normals)
        faces,data=self.index.nearest(points)
        alignment=np.einsum('ij,ij->i',data[:,3:6],normals)
        hit=(faces>=0)&(abs(alignment)>.72)&(data[:,6]<.013)
        gap=np.empty(len(points));gap[hit]=data[hit,7]/alignment[hit]
        if not np.all(hit):gap[~hit]=self.ray_gaps(points[~hit],normals[~hit])
        return gap

    def seat(self,rest,proposed,normals):
        before,after=self.clearance(rest,normals),self.clearance(proposed,normals)
        mask=(before<.024)&(after<.024)
        result=proposed.copy()
        result[mask]+=(np.maximum(before[mask],.00038)-after[mask]).clip(-.006,.006)[:,None]*normals[mask]
        return result


def clear_skin(sheet,ref,points,tri,normals):
    """Recheck actual skin after a compressed fan has been moved as a unit."""
    if not hasattr(sheet,'clearance'):return False
    moved=np.linalg.norm(points-ref,axis=1)>.000025
    local=tri[moved[tri].any(1)]
    if not len(local):return False
    vertices=np.unique(local)
    p=points[local]
    coarse=local[np.linalg.norm(p-np.roll(p,1,axis=1),axis=2).max(1)>.0027]
    a=coarse.ravel()
    b=np.roll(coarse,-1,axis=1).ravel()
    edge_keys=np.unique((np.minimum(a,b).astype(np.int64)<<32)|np.maximum(a,b))
    edges=np.column_stack((edge_keys>>32,edge_keys&0xffffffff)).astype(np.int32)
    touched=False
    for _ in range(3):
        samples=[points[vertices]];axes=[normals[vertices]]
        for faces in (coarse,edges):
            samples.append(points[faces].mean(1))
            direction=normals[faces].mean(1)
            axes.append(direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12))
        gap=sheet.clearance(np.vstack(samples),np.vstack(axes))
        need=np.where(gap<.00015,(.00048-gap).clip(0.,.007),0.)
        if not np.any(need):break
        touched=True
        single,center,middle=np.split(need,[len(vertices),len(vertices)+len(coarse)])
        lift=np.zeros(len(points));lift[vertices]=single
        for ids,stock in ((coarse,center),(edges,middle)):
            for col in ids.T:np.maximum.at(lift,col,stock)
        spread=lift[local].max(1)*.62
        for col in local.T:np.maximum.at(lift,col,spread)
        points+=lift[:,None]*normals
    # Reopen a compressed triangle in the tangent plane: a correction
    # across the skin normal would undo the contact clearance again.
    from .contact_index import tangent_fans
    repaired=tangent_fans(points,ref,local,normals)
    if repaired is not None:return touched or repaired
    a,b,c=ref[local].transpose(1,0,2)
    old=np.cross(b-a,c-a);area=np.linalg.norm(old,axis=1)
    unit=old/np.maximum(area[:,None],1e-14)
    for _ in range(24):
        q=points[local];a,b,c=q.transpose(1,0,2)
        signed=np.einsum('ij,ij->i',np.cross(b-a,c-a),unit)
        bad=(area>1e-12)&(signed<.009*area)
        if not bad.any():break
        touched=True;ids=local[bad];q=q[bad]
        g=np.cross(np.roll(q,-1,axis=1)-np.roll(q,-2,axis=1),unit[bad,None])
        n=normals[ids]
        g-=np.einsum('ijk,ijk->ij',g,n)[:,:,None]*n
        amount=(.055*area[bad]-signed[bad])/np.maximum((g*g).sum((1,2)),1e-16)
        v,inv=np.unique(ids,return_inverse=True)
        adjustment=np.zeros((len(v),3))
        np.add.at(adjustment,inv.ravel(),(g*amount[:,None,None]).reshape(-1,3))
        points[v]+=adjustment/np.bincount(inv.ravel())[:,None]
    return touched


def held_face_groups(rest,faces):
    # Closed bank duplicates have equal positions. The material faces
    # therefore share one sampling layout through successive loop breaks.
    # Releasing yarn can move an existing corner well outside the
    # pinched patch. Its adjacent 40 mm triangles still need contact.
    from .contact_index import face_probe_groups
    groups_at=face_probe_groups(rest,faces)
    if groups_at is None:
        p=rest[faces]
        sizes=np.linalg.norm(p-np.roll(p,1,axis=1),axis=2).max(1)
        groups_at=np.select((sizes>.018,sizes>.009,sizes>.004),(7,4,2),0)
    # Sample long faces across their full interior. Inside a curved
    # shin the worst crossing can lie between its edge and centroid.
    groups=[]
    for steps in (2,4,7):
        mask=groups_at==steps
        if mask.any():
            bary=np.array([(i,j,steps-i-j) for i in range(steps+1) for j in range(steps-i+1)
                           if max(i,j,steps-i-j)<steps],dtype=float)/steps
            groups.append((faces[mask],bary))
    return groups


def seat_held_faces(sheet,normals,clearance,topology,banks=None,*,face_needs=None,grip_vertices=None):
    # The strip can extend past the grab influence. Check the committed
    # triangles after both the bank constraints and the fracture recoil.
    cache=getattr(sheet,'held_contact',None)
    if cache is None or cache[0] is not topology:
        faces=sheet.faces
        if grip_vertices is not None:
            touched=np.zeros(len(sheet.rest),bool)
            touched[grip_vertices]=True
            faces=faces[touched[faces].any(1)]
        groups=held_face_groups(sheet.rest,faces)
        cache=sheet.held_contact=(topology,groups)
    groups=cache[1]
    pos=sheet.pos
    moved=np.einsum('ij,ij->i',pos-sheet.rest,pos-sheet.rest)>1e-10
    if not moved.any():return
    for _ in range(3):
        lift=np.zeros(len(pos))
        def correction(q,n):
            gap=clearance(q,n)
            return np.where(gap<.00011,np.minimum(.006,.00034-gap),0.)
        ids=np.flatnonzero(moved)
        lift[ids]=correction(pos[ids],normals[ids])
        for tiles,bary in groups:
            f=tiles[moved[tiles].any(1)]
            if not len(f):continue
            if face_needs is None:
                xyz=np.einsum('sc,fcj->fsj',bary,pos[f])
                axes=np.einsum('sc,fcj->fsj',bary,normals[f])
                axes/=np.maximum(np.linalg.norm(axes,axis=2)[:,:,None],1e-12)
                needs=correction(xyz.reshape(-1,3),axes.reshape(-1,3)).reshape(len(f),len(bary)).max(1)
            else:
                needs=face_needs(pos,normals,f,bary)
            affected=needs>0.
            if affected.any():
                for col in f[affected].T:np.maximum.at(lift,col,needs[affected])
        if banks is not None:
            shared=np.maximum(lift[banks[:,0]],lift[banks[:,1]])
            lift[banks[:,0]]=shared
            lift[banks[:,1]]=shared
        raised=lift>0.
        if not raised.any():break
        pos[raised]+=lift[raised,None]*normals[raised]
        if face_needs is not None:
            # Only a moved corner can invalidate the preceding exact face test.
            moved=raised.copy()
            if banks is not None:
                different=np.any(pos[banks[:,1]]!=pos[banks[:,0]],axis=1)
                moved[banks[different,1]]=True
        else:
            moved|=raised
        if banks is not None:
            pos[banks[:,1]]=pos[banks[:,0]]
