"""Extend a slit across an existing triangle sheet and through a second free edge."""
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np
from .tear_surface import strain_frame, limit_strain, limit_held_travel


def normalized(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)


def smooth(t):
    t=np.clip(t,0.,1.)
    return t*t*(3.-2.*t)


def chart_offset(uv,center):
    offset=uv-center
    offset[...,0]-=144.*np.round(offset[...,0]/144.)
    return offset


class ClothEdges:
    def __init__(self,faces,vertex_count):
        edges=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
        keys=edges.min(1).astype(np.int64)*vertex_count+edges.max(1)
        _,first,count=np.unique(keys,return_index=True,return_counts=True)
        self.edges=edges[first]
        self.used=np.zeros(vertex_count,bool);self.used[self.edges.ravel()]=True
        border=self.edges[count==1]
        nxt=defaultdict(list)
        for a,b in border:
            nxt[int(a)].append(int(b))
        self.border=np.zeros(vertex_count,bool)
        self.border[border.ravel()]=True
        self.cycles=[]
        seen=set()
        for v in nxt:
            if v in seen:
                continue
            chain=[];i=v
            while i not in seen and len(nxt.get(i,()))==1:
                seen.add(i);chain.append(i);i=nxt[i][0]
            if i==v and len(chain)>=3:
                self.cycles.append(np.asarray(chain,dtype='i'))


def closest_tips(points,uv,a,b):
    delta=chart_offset(uv[a,None],uv[b][None,:])
    world=points[a,None]-points[b][None,:]
    dist=np.einsum('ijk,ijk->ij',world,world)+.0000028*np.einsum('ijk,ijk->ij',delta,delta)
    i,j=np.unravel_index(dist.argmin(),dist.shape)
    return int(a[i]),int(b[j])


def bridge_at(sheet,point):
    best=None
    for i,first in enumerate(sheet.cracks):
        for other in sheet.cracks[i+1:]:
            a,b=closest_tips(sheet.rest,sheet.knit,first['rim'],other['rim'])
            p,q=sheet.rest[[a,b]]
            line=q-p
            length=np.linalg.norm(line)
            # After a partial join the last interlock can be smaller than one
            # screen pixel. A hit on cloth immediately beside this real edge
            # still pulls it. Unrelated overlapping rims have no such edge.
            if length<.003 and not hasattr(sheet,'contact_pose'):
                edges=sheet.layout.edges
                adjacent=edges[(edges[:,0]==a)|(edges[:,1]==a)]
                if (adjacent==b).any():
                    center=.5*(p+q)
                    distance=np.linalg.norm(point-center)
                    if distance<.0045:
                        pair=(first,other,a,b) if (point-center)@line<=0. else (other,first,b,a)
                        if best is None or distance<best[0]:best=distance,pair
                    continue
            if not .0004<length<.142:
                continue
            u=(point-p)@line/length**2
            distance=np.linalg.norm(point-p-np.clip(u,0.,1.)*line)
            if not -.16<=u<=1.16 or distance>min(.020,.010+.14*length):
                continue
            # A longer chord passing under an intermediate eye cannot carry
            # load through that already empty stretch of stocking.
            blocked=False
            for h in sheet.cracks:
                if h is first or h is other:continue
                rel=sheet.rest[h['rim']]-p
                t=rel@line/length**2
                if ((t>.06)&(t<.94)&(np.linalg.norm(rel-t[:,None]*line,axis=1)<.014)).any():
                    blocked=True;break
            if blocked:continue
            score=distance+.045*length+.012*max(-u,u-1.,0.)
            pair=(first,other,a,b) if u<=.5 else (other,first,b,a)
            if best is None or score<best[0]:
                best=score,pair
    return best[1] if best else None


def material_path(sheet,a,b):
    p,q=sheet.rest[[a,b]]
    line=q-p
    length=np.linalg.norm(line)
    if length<1e-6:return None
    uv_delta=chart_offset(sheet.knit,sheet.knit[a])
    du=uv_delta[b]
    along=uv_delta@du/max(du@du,1e-12)
    lateral=np.linalg.norm(uv_delta-along[:,None]*du,axis=1)
    rel=sheet.rest-p
    progress=rel@line/length**2
    clearance=np.linalg.norm(rel-np.clip(progress,0,1)[:,None]*line,axis=1)
    allow=(clearance<.017+.20*length)&(lateral<8.)&(progress>-.24)&(progress<1.24)
    allow[sheet.layout.border]=False
    allow[[a,b]]=True
    edges=sheet.layout.edges
    edge=edges[allow[edges].all(1)]
    v,w=edge.T
    lengths=np.linalg.norm(sheet.rest[v]-sheet.rest[w],axis=1)
    # Following the stitch lane avoids straight chords across a curved leg.
    cost=lengths*(1.+.30*(lateral[v]+lateral[w])**2+5.*(clearance[v]+clearance[w])/max(length,.012))
    adj=defaultdict(list)
    for (v,w),d in zip(edge.tolist(),cost.tolist()):
        adj[v].append((w,d));adj[w].append((v,d))
    todo=[(0.,a)];distance={a:0.};prev={}
    while todo:
        d,v=heappop(todo)
        if v==b:
            path=[b]
            while path[-1]!=a:path.append(prev[path[-1]])
            return np.array(path[::-1],dtype='i')
        if d>distance[v]:continue
        for w,c in adj[v]:
            alt=d+c
            if alt<distance.get(w,np.inf):
                distance[w]=alt;prev[w]=v;heappush(todo,(alt,w))
    return None


def cut_fans(faces,path,border):
    """Faces on the same oriented side of each cut edge, including its two ends."""
    mask=np.zeros(int(faces.max())+1,bool);mask[path]=True
    local=np.flatnonzero(mask[faces].any(1))
    directed={};incident=defaultdict(set);around=defaultdict(list)
    for fid,tri in zip(local.tolist(),faces[local].tolist()):
        for v,w in zip(tri,tri[1:]+tri[:1]):
            directed[v,w]=fid
            around[min(v,w),max(v,w)].append(fid)
            if mask[v]:incident[v].add(fid)
    fans=[]
    for i,v in enumerate(path.tolist()):
        if (i==0 or i==len(path)-1) and not border[v]:
            fans.append(None);continue
        before=int(path[i-1]) if i else None
        after=int(path[i+1]) if i+1<len(path) else None
        seed=directed.get((v,after) if after is not None else (before,v))
        if seed is None:return None
        todo=[seed];visited={seed}
        while todo:
            f=todo.pop()
            for w in faces[f].tolist():
                if w in (v,before,after):continue
                for nxt in around[min(v,w),max(v,w)]:
                    if nxt not in visited:
                        visited.add(nxt);todo.append(nxt)
        if len(visited)==len(incident[v]):return None
        fans.append(np.asarray(sorted(visited),dtype='i'))
    return fans


class OpenSeam:
    def __init__(self,sheet,path,fans,widths,joining):
        self.path,self.fans,self.joining=path,fans,joining
        self.base=sheet.rest.copy()
        self.dry_bridge=False
        # A dry cut pays yarn out of the two boundaries it actually reaches.
        # Other holes may flex under the hand, but their relaxed lips should
        # not shrink on every nearby bridge cut and free-edge smoothing pass.
        self.other_rims=np.empty(0,dtype=int)
        if not hasattr(sheet,'contact_pose'):
            touch=[np.isin(path[[0,-1]],c['rim']) for c in sheet.cracks]
            self.dry_bridge=bool(joining and any(a and not b for a,b in touch) and any(b and not a for a,b in touch))
            other=[c['rim'] for c,ends in zip(sheet.cracks,touch) if not ends.any()]
            if other:self.other_rims=np.unique(np.concatenate(other))
        line=self.base[path]
        edge=np.diff(line,axis=0)
        self.stations=np.r_[0.,np.cumsum(np.linalg.norm(edge,axis=1))]
        self.length=self.stations[-1]
        self.reached=0
        self.done=False
        self.copies={}
        self.widths=np.asarray(widths)
        self.normal=normalized(sheet.normals[path].mean(0))
        self.tangent=normalized(line[-1]-line[0])
        self.across=normalized(np.cross(self.normal,self.tangent))
        reach=min(.006,self.length)
        projection=(line-line[0])@self.tangent
        self.rolled_start=self.dry_bridge and np.interp(reach,self.stations,projection)<.35*reach
        self.phase=float(sheet.knit[path[0]]@[.71,.23])
        # Nearest position on the seam in the original cloth. Extend the frame
        # a short distance into each old eye so the join opens at the former tip.
        radius=.065+max(widths)
        active=(self.base>=line.min(0)-radius).all(1)&(self.base<=line.max(0)+radius).all(1)
        self.ids=np.flatnonzero(active)
        points=self.base[self.ids]
        guide=line.copy()
        if len(line)>2:
            guide[1:-1]=.25*line[:-2]+.5*line[1:-1]+.25*line[2:]
        seg=np.diff(guide,axis=0)
        rel=points[:,None]-guide[:-1]
        u=np.einsum('ijk,jk->ij',rel,seg)/np.maximum((seg*seg).sum(1),1e-15)
        u=np.clip(u,0.,1.)
        offsets=rel-u[:,:,None]*seg
        index=np.argmin(np.einsum('ijk,ijk->ij',offsets,offsets),axis=1)
        ar=np.arange(len(points))
        s=self.stations[index]+u[ar,index]*np.diff(self.stations)[index]
        first=(index==0)&(u[ar,index]<1e-7);last=(index==len(seg)-1)&(u[ar,index]>1.-1e-7)
        s[first]+=np.minimum(0.,(points[first]-line[0])@self.tangent)
        s[last]+=np.maximum(0.,(points[last]-line[-1])@self.tangent)
        normals=sheet.support_normals[self.ids]
        across=normalized(self.across-(normals@self.across)[:,None]*normals)
        offset=offsets[ar,index]
        self.s=s;self.across_field=across
        self.side=np.sign(np.einsum('ij,ij->i',offset,across))
        self.side[np.searchsorted(self.ids,path)]=-1.
        dist=abs(np.einsum('ij,ij->i',offset,across))
        depth=abs(np.einsum('ij,ij->i',offset,normals))
        self.fall=smooth(1.-dist/radius)*smooth(1.-depth/.038)
        order=np.searchsorted(self.ids,path)
        self.s[order]=self.stations
        self.fall[order]=1.
        self.path_local=order
        eligible=np.zeros(len(self.base),bool)
        eligible[self.ids]=(dist<radius)&(depth<.039)
        self.face_ids=np.flatnonzero(eligible[sheet.faces].any(1))
        averaged=np.stack([np.interp(self.stations+d,self.stations,line[:,j])
                           for d in (-.004,-.002,0.,.002,.004) for j in range(3)]).reshape(5,3,-1).mean(0).T
        self.center_bias=np.einsum('ij,ij->i',averaged-line,across[order])
        self.center_bias=np.clip(self.center_bias,-.0014,.0014)
        self.center_bias[[0,-1]]=0.

    def relax_banks(self,sheet,width):
        # Extend the bank displacement through connected cloth. Opposite
        # sides can be coincident in Knit coordinates but have no spring
        # between them after their interlock has broken.
        original=np.arange(len(sheet.rest),dtype='i')
        for i,dup in self.copies.items():original[dup]=self.path[i]
        ids,tri=np.unique(sheet.faces[self.face_ids],return_inverse=True)
        tri=tri.reshape(-1,3);ref=self.base[original[ids]]
        # A duplicated bank starts at its parent's current position. All earlier
        # pairs already have a collision-tested, unloaded position of their own.
        accepted=sheet.rest[ids] if self.dry_bridge else ref
        pts=ref[tri]
        ab,ac=(pts[:,1:]-pts[:,:1]).transpose(1,0,2)
        area=np.linalg.norm(np.cross(ab,ac),axis=1)
        edges=tri[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
        corner=pts[:,[2,0,1]]
        dots=np.einsum('ijk,ijk->ij',pts-corner,pts[:,[1,2,0]]-corner)
        weights=(.5*dots/np.maximum(area[:,None],1e-13)).ravel().clip(.035,12.)
        ia,ib=edges.T
        source=np.r_[ia,ib];dest=np.r_[ib,ia];weights=np.tile(weights,2)
        diag=np.bincount(dest,weights,minlength=len(ids))
        mass=np.bincount(tri.ravel(),np.repeat(area/6.,3),minlength=len(ids))
        diag+=mass/.050**2
        sidx=np.searchsorted(self.ids,original[ids])
        inside=sidx<len(self.ids)
        inside[inside]=self.ids[sidx[inside]]==original[ids[inside]]
        fixed=~inside.copy()
        other=np.isin(original[ids],self.other_rims)
        fixed|=other
        valid=np.flatnonzero(inside)
        fixed[valid]|=self.fall[sidx[valid]]<.022
        if self.dry_bridge:
            from .bridge_relax import retract
            released=retract(self,ids,ref,tri,fixed,source,dest,weights,diag)
            x=np.linalg.norm(released-ref,axis=1)
            self.previous_displacement=None
        else:
            previous=getattr(self,'previous_displacement',None)
            x=np.zeros(len(ids))
            if previous is not None:
                old_ids,old_x=previous
                _,old_i,new_i=np.intersect1d(old_ids,ids,assume_unique=True,return_indices=True)
                x[new_i]=old_x[old_i]
                x[fixed]=0.
            if self.copies:
                steps=np.fromiter(self.copies,dtype=int)
                new=np.fromiter(self.copies.values(),dtype=int)
                magnitude=width[self.path_local[steps]]
                bias=self.center_bias[steps]+.065*np.sin(self.stations[steps]*80.+self.phase)*magnitude
                for vertices,value in ((self.path[steps],bias-magnitude),(new,bias+magnitude)):
                    known=np.searchsorted(ids,vertices)
                    x[known]=value;fixed[known]=True
            tip=np.searchsorted(ids,self.path[self.reached])
            if not self.done or self.fans[-1] is None:fixed[tip]=True;x[tip]=0.
            def multiply(v):
                res=diag*v-np.bincount(dest,weights*v[source],minlength=len(ids))
                res[fixed]=0.
                return res
            residual=-multiply(x)
            z=residual/np.maximum(diag,1e-12);direction=z.copy();rz=residual@z
            for _ in range(100):
                if rz<2e-14 or np.max(abs(z))<.000012:break
                ad=multiply(direction)
                step=rz/max(direction@ad,1e-25)
                x+=step*direction;residual-=step*ad
                z=residual/np.maximum(diag,1e-12)
                nxt=residual@z;direction=z+(nxt/rz)*direction;rz=nxt
            self.previous_displacement=(ids,x)
            axes=np.broadcast_to(self.across,ref.shape).copy()
            axes[valid]=self.across_field[sidx[valid]]
            released=ref+x[:,None]*axes
        # Round a turn from an old eye into the newly released bank. Extend
        # this small correction into its attached courses to keep Knit smooth.
        if self.copies:
            all_edges=np.sort(edges,axis=1)
            keys=all_edges[:,0].astype(np.int64)*len(ref)+all_edges[:,1]
            _,idx,count=np.unique(keys,return_index=True,return_counts=True)
            outer=edges[idx[count==1]]
            near=(np.abs(x)>.001)&~other
            adjustment=np.zeros_like(ref);pins=np.zeros(len(ref),bool)
            v=np.unique(outer[near[outer].any(1)])
            v=v[near[v]]
            pos_sum=np.zeros((len(v),3));total=np.zeros(len(v))
            for a,b in (outer.T,outer[:,::-1].T):
                nxt=np.arange(len(ref));nxt[a]=b
                i=v.copy();arc=np.zeros(len(v))
                for _ in range(18):
                    j=nxt[i];arc+=np.linalg.norm(released[j]-released[i],axis=1)
                    use=(j!=v)&(j!=i)&(arc<.009)
                    if not use.any():break
                    w=np.exp(-.5*(arc/.0033)**2)*use
                    pos_sum+=w[:,None]*released[j];total+=w;i=j
            use=total>.1;v=v[use]
            d=pos_sum[use]/total[use,None]-released[v]
            adjustment[v]=d*np.minimum(.82,.0029/np.maximum(np.linalg.norm(d,axis=1),1e-12))[:,None]
            pins[v]=True
            pins[other]=True
            free=~pins
            for _ in range(12):
                smooth_delta=np.column_stack([np.bincount(dest,weights*adjustment[source,i],minlength=len(ref)) for i in range(3)])
                adjustment[free]=.91*smooth_delta[free]/np.maximum(diag[free,None],1e-12)
            released+=adjustment
        if getattr(sheet,'seat',None):
            moved=np.flatnonzero(abs(x)>.00012)
            if len(moved):
                released[moved]=sheet.seat(ref[moved],released[moved],sheet.support_normals[original[ids[moved]]])
        frame=strain_frame(ref,tri)
        limit_strain(released,frame)
        limit_held_travel(released,accepted,frame)
        if hasattr(sheet,'clearance'):
            surface_normals=sheet.support_normals[original[ids]]
            moved=np.linalg.norm(released-ref,axis=1)>.00004
            local_tri=tri[moved[tri].any(1)]
            p=released[local_tri]
            coarse=(np.linalg.norm(p-np.roll(p,1,axis=1),axis=2).max(1)>.0032)
            local_tri=local_tri[coarse]
            # A coarse face may cut the shin at its centre even when all
            # its vertices clear the skin. Correct those actual contact points.
            for _ in range(2):
                lift=np.zeros(len(ids))
                for weights in ((1/3,1/3,1/3),(.5,.5,0.),(0.,.5,.5),(.5,0.,.5)):
                    w=np.asarray(weights)
                    sample=np.einsum('j,njk->nk',w,released[local_tri])
                    normals=normalized(np.einsum('j,njk->nk',w,surface_normals[local_tri]))
                    gap=sheet.clearance(sample,normals)
                    need=np.where(gap<.00023,np.clip(.00039-gap,0.,.0035),0.)
                    for k in np.flatnonzero(w):np.maximum.at(lift,local_tri[:,k],need)
                if not lift.any():break
                released+=lift[:,None]*surface_normals
        # A very oblique inherited stitch fan has little room on its
        # compressed side. Share the remaining displacement deficit along
        # its connected stars, then line-search the last fraction if needed.
        ref_cross=np.cross(ab,ac)
        minimum=.016*area**2
        if self.dry_bridge:
            p,q,r=accepted[tri].transpose(1,0,2)
            previous_area=np.einsum('ij,ij->i',np.cross(q-p,r-p),ref_cross)
            minimum=np.minimum(minimum,previous_area-.001*area**2)
        def compressed(points):
            p,q,r=points[tri].transpose(1,0,2)
            return (np.einsum('ij,ij->i',np.cross(q-p,r-p),ref_cross)<minimum)&(area>1e-12)
        displacement=released-accepted
        factor=np.ones(len(ref))
        for _ in range(16):
            bad=compressed(released)
            if not bad.any():break
            loss=np.zeros(len(ref));loss[tri[bad].ravel()]=.34
            for spread in range(7):
                adjacent=np.zeros_like(loss)
                np.maximum.at(adjacent,dest,loss[source])
                loss=np.maximum(loss,.80*adjacent)
            factor*=1.-loss
            released=accepted+(displacement*factor[:,None])
        if self.dry_bridge:
            for _ in range(12):
                bad=compressed(released)
                if not bad.any():break
                pins=np.unique(tri[bad])
                released[pins]=accepted[pins]
        if compressed(released).any():
            low,high=0.,1.
            delta=released-accepted
            for _ in range(13):
                t=(low+high)*.5
                if compressed(accepted+t*delta).any():high=t
                else:low=t
            released=accepted+low*delta
        # Every paid station can become the reference for another pull or release.
        # Correct the two free banks while the candidate is still private.
        from .seam_overlap import uncross
        from .seam_contact import clear_skin
        support=sheet.support_normals[original[ids]]
        if self.dry_bridge:
            from .seam_overlap import settle_paid_out
            from .bridge_relax import preserve_open, unfold
            clear_skin(sheet,accepted,released,tri,support)
            if not unfold(released,tri,source,dest,sheet,support):
                settle_paid_out(ref,released,tri,accepted)
            steps=np.fromiter(self.copies,dtype=int)
            left=np.searchsorted(ids,self.path[steps])
            right=np.searchsorted(ids,np.fromiter(self.copies.values(),dtype=int))
            if preserve_open(accepted,released,left,right):
                clear_skin(sheet,accepted,released,tri,support)
                if not unfold(released,tri,source,dest,sheet,support):
                    settle_paid_out(ref,released,tri,accepted)
        else:
            uncross(accepted,released,tri)
            if clear_skin(sheet,accepted,released,tri,support):
                if uncross(accepted,released,tri):
                    clear_skin(sheet,accepted,released,tri,support)
        sheet.rest[ids]=released
        self.checked_rest=self.reached

    def grow(self,sheet,distance):
        stop=min(len(self.path)-1,int(np.searchsorted(self.stations,distance,side='right')-1))
        if stop<=self.reached:return False
        final=stop==len(self.path)-1
        nodes=[i for i in range(self.reached,stop+int(final)) if self.fans[i] is not None]
        # An interior slit first opens when a pair of distinct bank vertices exists.
        if not nodes and not self.copies:return False
        if nodes:
            original=self.path[nodes]
            dup=sheet.duplicate(original)
            for i,src,dst in zip(nodes,original,dup):
                fs=self.fans[i]
                field=sheet.faces[fs]
                field[field==src]=dst
                sheet.faces[fs]=field
                self.copies[i]=dst
        self.reached=stop;self.done=final
        front=self.stations[stop]
        t=np.clip(self.s/self.length,0.,1.)
        width=(1.-t)*self.widths[0]+t*self.widths[1]
        width*=1.+.08*np.sin(self.s*115.+self.phase)+.035*np.sin(self.s*280.-self.phase)
        tail=.014 if self.joining and final else 0.
        head=.015 if self.fans[0] is not None else 0.
        fade_in=smooth((self.s+head)/max(.009,head)) if head else smooth(self.s/.010)
        fade_out=smooth((front+tail-self.s)/max(.009,tail))
        width*=fade_in*fade_out*smooth(front/.014)*self.fall
        self.relax_banks(sheet,width)
        return True
