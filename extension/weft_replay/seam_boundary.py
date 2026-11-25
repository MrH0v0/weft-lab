"""Separate the factory opening from torn bank segments that meet that edge."""
import numpy as np


def factory_cuff(cycles,knit,used=()):
    skip=set(used)
    candidates=[c for c in cycles if not any(int(v) in skip for v in c)]
    if not candidates:return np.empty(0,dtype='i')
    def coverage(c):return len(np.unique(np.floor((knit[c,0]%144.)/4.)))
    return max(candidates,key=coverage)


def nearest_edge(points,ring,at,natural=None):
    if not len(ring):return float('inf'),None
    a=points[ring];b=np.roll(a,-1,axis=0)
    ab=b-a
    t=np.clip(np.einsum('ij,ij->i',at-a,ab)/np.maximum((ab*ab).sum(1),1e-20),0,1)
    d=np.linalg.norm(a+t[:,None]*ab-at,axis=1)
    if natural is not None:
        ends=np.isin(ring,list(natural))
        d[~(ends&np.roll(ends,-1))]=np.inf
    k=int(d.argmin())
    return float(d[k]),int(ring[(k+int(t[k]>.5))%len(ring)])


def torn_arcs(cycle,cuff):
    natural=np.isin(cycle,list(cuff))
    if not natural.any():
        yield cycle,False
        return
    # The closing chord across a cuff notch is air. Keep only its two torn
    # banks, including their independent attachment points on the factory edge.
    starts=np.flatnonzero(natural&~np.roll(natural,-1))
    for i in starts:
        j=(int(i)+1)%len(cycle);arc=[int(cycle[i])]
        while not natural[j]:
            arc.append(int(cycle[j]));j=(j+1)%len(cycle)
        arc.append(int(cycle[j]))
        yield np.array(arc,dtype='i'),True
