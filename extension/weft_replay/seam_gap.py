"""Keep the free edges of an unloaded, paid-out slit on their own fabric banks."""
import numpy as np
from .seam_overlap import intersections


def retain_gap(points,faces,pairs,profile,required,sheet,normals):
    """Retract each free lip into its attached courses, tapering to the intact tips."""
    left,right=pairs.T
    lengths=np.linalg.norm(points[right]-points[left],axis=1)
    if required<=0. or np.max(lengths)>=required:return False
    center=points[pairs].mean(1)
    tangent=np.gradient(center,axis=0)
    tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1.e-12)
    boundary=np.zeros(len(points),bool);boundary[pairs]=True
    local=boundary.copy()
    for _ in range(5):
        selected=local[faces].any(1)
        local[faces[selected]]=True
    region=np.flatnonzero(local);remap=np.full(len(points),-1,int);remap[region]=np.arange(len(region))
    tri=remap[faces[selected]]
    edges=np.unique(np.sort(tri[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
    a,b=edges.T;src=np.r_[a,b];dst=np.r_[b,a];ref=points[region]
    vec=ref[src]-ref[dst]
    weights=1./np.maximum(np.linalg.norm(vec,axis=1),.00025)
    interior=~boundary[region[src]]
    count=len(region)
    inward=np.column_stack([np.bincount(dst[interior],(weights*vec[:,j])[interior],minlength=count)
                            for j in range(3)])
    l,r=remap[left],remap[right]
    axis=inward[r]-inward[l]
    axis-=tangent*(axis*tangent).sum(1)[:,None]
    axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1.e-12)
    residual=np.maximum(0.,required*profile-np.einsum('ij,ij->i',points[right]-points[left],axis))
    delta=np.zeros_like(ref);delta[l]=-.5*residual[:,None]*axis;delta[r]=-delta[l]
    # Fixed outer ring leaves the surrounding tube and any distant apertures in place.
    outside=np.zeros(len(points),bool);outside[faces[~selected]]=True
    free=~(outside[region]|boundary[region])
    mass=np.maximum(np.bincount(dst,weights,minlength=count),1.e-12)
    for _ in range(25):
        mean=np.column_stack([np.bincount(dst,weights*delta[src,j],minlength=count) for j in range(3)])/mass[:,None]
        delta[free]=mean[free]
    base_norm=np.cross(ref[tri[:,1]]-ref[tri[:,0]],ref[tri[:,2]]-ref[tri[:,0]])
    area=np.einsum('ij,ij->i',base_norm,base_norm)
    for fraction in (1.,.75,.5,.25,.125,.0625):
        candidate=ref+fraction*delta
        if normals is not None and hasattr(sheet,'clearance'):
            from .seam_contact import clear_skin
            clear_skin(sheet,ref,candidate,tri,normals[region])
        norm=np.cross(candidate[tri[:,1]]-candidate[tri[:,0]],candidate[tri[:,2]]-candidate[tri[:,0]])
        if np.any(np.einsum('ij,ij->i',norm,base_norm)<.012*area):continue
        if len(intersections(candidate,tri)):continue
        points[region]=candidate
        return True
    return False


def restore_saved(sheet):
    """Recover an earlier sealed snapshot with intact topology and coincident free banks."""
    changed=False
    for crack in sheet.cracks:
        if crack['notch'] or len(crack['members'])!=1:continue
        cycle=crack['rim']
        coords=sheet.knit[cycle].copy();coords[:,0]%=144.
        _,which,counts=np.unique(np.round(coords,4),return_inverse=True,return_counts=True,axis=0)
        paired=[cycle[which==k] for k in np.flatnonzero(counts==2)]
        if len(paired)<3:continue
        pairs=np.array(paired)
        if np.linalg.norm(sheet.rest[pairs[:,0]]-sheet.rest[pairs[:,1]],axis=1).max()>.00035:continue
        # Order along the old boundary, including a seam crossing the chart wrap.
        mid=sheet.rest[pairs].mean(1)
        direction=np.linalg.svd(mid-mid.mean(0),full_matrices=False)[2][0]
        order=np.argsort(mid@direction);pairs=pairs[order];mid=mid[order]
        profile=np.sin(np.pi*(np.arange(len(pairs))+.5)/len(pairs))**.72
        span=np.ptp(mid@direction)
        before=sheet.rest.copy()
        if retain_gap(sheet.rest,sheet.faces,pairs,profile,min(.009,.24*span),sheet,sheet.support_normals):
            sheet.pos+=sheet.rest-before;changed=True
    if changed:sheet.refresh_layout()
    return changed
