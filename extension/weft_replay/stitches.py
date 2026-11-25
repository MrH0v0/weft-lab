"""Two instrumented centrelines, reconstructed in the garment material chart."""
import numpy as np
from .cache import assets

def centrelines(frame):
    rows=np.asarray(frame["loopPatch"]).reshape(2,4,4)
    result=[]
    for j in (1,2):
        shoulder,lean,crown,feed=rows[0,j];nxt=rows[1,j]
        neck=np.clip(.135-.20*shoulder+.16*crown,.09,.19)
        wide=np.clip(.255+shoulder,.10,.39);top=1.25+.55*crown;mid=.65+.32*crown
        upper=np.clip(.135-.20*nxt[0]+.16*nxt[2],.09,.19)
        l=(.5-neck+.08*lean,.045);r=(.5+neck+.08*lean,.045)
        l2=(max(.04,.5-wide+.46*lean),mid);r2=(min(.96,.5+wide+.46*lean),mid)
        cl=(.5-upper+.08*nxt[1],1.045);cr=(.5+upper+.08*nxt[1],1.045)
        apex=(.5+lean,top);capl=min(cl[0],apex[0]-.04);capr=max(cr[0],apex[0]+.04)
        footl=(0,-.22+(rows[0,j-1,3]+feed)*.5);footr=(1,-.22+(rows[0,j+1,3]+feed)*.5)
        knots=np.asarray([footl,l,l2,cl,apex,cr,r2,r,footr])
        handles=np.asarray([(l[0]-.015,footl[1]),(l[0]+.02,mid*.48),(2*cl[0]-capl,.88),(capl,top),
                            (capr,top),(2*cr[0]-capr,.88),(r[0]-.02,mid*.48),(r[0]+.015,footr[1])])
        t=np.linspace(0,1,6,endpoint=False)[None,:,None]
        points=(1-t)**2*knots[:-1,None]+2*(1-t)*t*handles[:,None]+t*t*knots[1:,None]
        chart=np.vstack((points.reshape(-1,2),knots[-1]))
        result.append((chart+[41+j,170])/[164,510])
    return result

class StitchPatch:
    def __init__(self):
        _,faces,uv,_=assets()["Stocking"];p=uv[faces]
        selected=(p[:,:,0].max(1)>=41/164)&(p[:,:,0].min(1)<=45/164)&(p[:,:,1].max(1)>=169/510)&(p[:,:,1].min(1)<=172/510)
        self.faces=faces[selected];p=p[selected]
        self.origin=p[:,0];self.inverse=np.linalg.inv(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0]],axis=2))

    def lift(self,coords,uv):
        ab=np.einsum("tij,tnj->tni",self.inverse,uv[None,:,:]-self.origin[:,None,:])
        w=np.concatenate([1-ab.sum(axis=2,keepdims=True),ab],axis=2)
        index=w.min(axis=2).argmax(axis=0)
        weights=w[index,np.arange(len(uv))]
        if weights.min()<-.0001:raise ValueError("Stitch left its material patch")
        return np.einsum("ni,nij->nj",weights,coords[self.faces[index]])

