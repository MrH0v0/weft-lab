"""Refine the actual material under a continued grip, preserving the original loop UV on each face."""
import numpy as np
from .open_seam import chart_offset, normalized

# Corners 3, 4, 5 are the midpoints of AB, BC and CA.
CHILDREN=(((0,1,2),),((0,3,2),(3,1,2)),((1,4,0),(4,2,0)),
          ((3,1,4),(0,3,4),(0,4,2)),((2,5,1),(5,0,1)),
          ((5,0,3),(3,1,2),(3,2,5)),((4,2,5),(0,1,4),(0,4,5)),
          ((0,3,5),(3,1,4),(5,4,2),(3,4,5)))


def retarget(face, point, positions, new_faces, parent):
    children=np.flatnonzero(parent==face)
    if len(children)==1:return int(children[0])
    p=positions[new_faces[children]]
    a=p[:,1]-p[:,0];b=p[:,2]-p[:,0];d=point-p[:,0]
    cross=np.cross(a,b);den=np.einsum('ij,ij->i',cross,cross)
    u=np.einsum('ij,ij->i',np.cross(d,b),cross)/np.maximum(den,1e-30)
    v=np.einsum('ij,ij->i',np.cross(a,d),cross)/np.maximum(den,1e-30)
    return int(children[np.argmax(np.minimum(np.minimum(u,v),1-u-v))])


def refine(sheet,face,point,center,radius=.067,spacing=.0023):
    changed=False
    for _ in range(6):
        tri=sheet.faces
        rest=sheet.rest
        near=(np.abs(rest-center)<radius+.012).all(1)
        local=np.flatnonzero(near[tri].any(1))
        edges=tri[local][:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
        keys=edges.min(1).astype(np.int64)*len(rest)+edges.max(1)
        _,idx,inv=np.unique(keys,return_index=True,return_inverse=True)
        a,b=edges[idx].T
        length=np.linalg.norm(rest[a]-rest[b],axis=1)
        distances=np.linalg.norm((rest[a]+rest[b])*.5-center,axis=1)
        bisect=(distances<radius)&(length>spacing)
        if not bisect.any():break
        a,b=a[bisect],b[bisect]
        mid=np.arange(len(rest),len(rest)+len(a),dtype='i')
        mapping=np.full(len(idx),-1,dtype='i');mapping[bisect]=mid
        mids=np.full(tri.shape,-1,dtype='i');mids[local]=mapping[inv].reshape(-1,3)
        mask=(mids>=0)@np.array([1,2,4])
        table=np.column_stack((tri,mids));uv=sheet.corner_uv.reshape(-1,3,2)
        corners=np.concatenate((uv,(uv+np.roll(uv,-1,axis=1))*.5),axis=1)
        faces=[];knit=[];parents=[]
        for kind,template in enumerate(CHILDREN):
            rows=np.flatnonzero(mask==kind)
            for order in template:
                faces.append(table[rows][:,order]);knit.append(corners[rows][:,order]);parents.append(rows)
        sheet.faces=np.concatenate(faces);sheet.corner_uv=np.concatenate(knit).reshape(-1,2)
        sheet.rest=np.vstack((rest,(rest[a]+rest[b])*.5))
        if getattr(sheet,'bank_weave',None) is not None:
            data=sheet.bank_weave;sheet.bank_weave=np.vstack((data,.5*(data[a]+data[b])))
        if hasattr(sheet,'material_reference'):
            old=sheet.material_reference;sheet.material_reference=np.vstack((old,.5*(old[a]+old[b])))
        sheet.pos=np.vstack((sheet.pos,(sheet.pos[a]+sheet.pos[b])*.5))
        sheet.knit=np.vstack((sheet.knit,sheet.knit[a]+.5*chart_offset(sheet.knit[b],sheet.knit[a])))
        sheet.support_normals=np.vstack((sheet.support_normals,normalized(sheet.support_normals[a]+sheet.support_normals[b])))
        if sheet.toe is not None:sheet.toe=np.r_[sheet.toe,.5*(sheet.toe[a]+sheet.toe[b])]
        # Only edges actually on the factory opening inherit cuff membership.
        if sheet.cuff:
            both=np.isin(a,list(sheet.cuff))&np.isin(b,list(sheet.cuff))
            if both.any():
                border_keys,uses=np.unique(keys,return_counts=True)
                border=border_keys[uses==1]
                both&=np.isin(np.minimum(a,b).astype('int64')*len(rest)+np.maximum(a,b),border)
                sheet.cuff.update(map(int,mid[both]))
        face=retarget(face,point,sheet.pos,sheet.faces,np.concatenate(parents))
        changed=True
    if changed:
        sheet.refresh_layout();sheet.identify()
    return face,changed
