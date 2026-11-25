"""Material-space gauges for individual loops on the currently grasped patch."""
import numpy as np
from .yarn_path import yarn_length
from .contact_index import measure_gauges


class LoopGauges:
    def __init__(self, uv, triangles, rest, ledger, first_row, failed=None):
        chart=uv[triangles].copy()
        chart[:,:,0]-=np.round(chart[:,:,0]-chart[:,:1,0])
        chart[:,:,0]*=144
        sides=(chart[:,1:]-chart[:,:1]).transpose(0,2,1)
        valid=abs(np.linalg.det(sides))>1e-8
        chart,triangles,inverse=chart[valid],triangles[valid],np.linalg.inv(sides[valid])
        low=np.floor(chart.min(axis=1)).astype(int)
        span=np.floor(chart.max(axis=1)).astype(int)-low+1
        count=span.prod(axis=1)
        face=np.repeat(np.arange(len(chart)),count)
        within=np.arange(len(face))-np.repeat(count.cumsum()-count,count)
        cell=low[face]+np.column_stack((within%span[face,0],within//span[face,0]))
        # A scalar key has the same U-then-V order as the chart rows. Keep
        # the same last-face selection when several triangles meet a probe.
        low_row=cell[:,1].min(initial=first_row)
        rows=cell[:,1].max(initial=first_row)-low_row+1
        keys,slot=np.unique((cell[:,0]%144)*rows+cell[:,1]-low_row,return_inverse=True)
        ids=np.column_stack((keys//rows,keys%rows+low_row))
        corners=np.zeros((len(ids),4,3),int)
        weights=np.zeros_like(corners,dtype=float)
        found=np.zeros((len(ids),4),bool)
        for k,offset in enumerate(((.15,.5),(.85,.5),(.5,.10),(.5,.90))):
            ab=np.einsum('nij,nj->ni',inverse[face],cell+offset-chart[face,0])
            bary=np.column_stack((1-ab.sum(1),ab))
            inside=bary.min(1)>=-1e-7
            corners[slot[inside],k]=triangles[face[inside]]
            weights[slot[inside],k]=bary[inside]
            found[slot[inside],k]=True
        complete=found.all(1)
        self.cells=ids[complete]
        self.corners=corners[complete].astype(np.int32)
        self.weights=weights[complete]
        self.reference,self.rest_angle=self.measure(rest)
        self.rest_length=yarn_length(self.reference,self.rest_angle)
        self.ledger=ledger
        x,y=self.cells.T
        self.failed=failed if failed is not None else np.zeros_like(ledger,dtype=bool)
        self.outgoing=(y-first_row,x)
        self.incoming=(y-first_row,(x-1)%144)
        # Pad the V ends and wrap only the knitted circumference in U.
        # Missing adjacent cells refer to the current cell, as before.
        indices=np.arange(len(x))
        first=y.min(initial=first_row)
        row=y-first+1
        lookup=np.full((int(row.max(initial=0))+2,144),-1,int)
        lookup[row,x]=indices
        adjacent=np.column_stack((lookup[row,(x-1)%144],lookup[row,(x+1)%144],
                                  lookup[row-1,x],lookup[row+1,x]))
        self.adjacent=np.where(adjacent>=0,adjacent,indices[:,None])
        self.neighbours=self.adjacent[:,:2].copy()
        self.right=self.neighbours[:,1]
        self.linked=self.right!=np.arange(len(self.cells))
        self.capacity=np.minimum(self.rest_length,self.rest_length[self.right])
        self.update(rest,0)

    def measure(self,points):
        measured=measure_gauges(self,points)
        if measured is not None:return measured
        sample=np.einsum('nkq,nkqd->nkd',self.weights,points[self.corners])
        across=sample[:,1]-sample[:,0]
        up=sample[:,3]-sample[:,2]
        self.centers=sample.mean(1)
        self.vectors=np.stack((across,up),axis=1)
        lengths=np.maximum(np.linalg.norm(self.vectors,axis=2),1e-9)
        return lengths,np.sum(across*up,axis=1)/lengths.prod(1)

    def update(self,pos,dt,*,substeps=None):
        lengths,angle=self.measure(pos)
        self.stretch=lengths/self.reference
        course=np.maximum(self.stretch[:,0],.02)
        wale=np.maximum(self.stretch[:,1],.02)
        self.bias=np.log(course/wale)
        self.load=np.log(course)+.35*np.log(wale)
        self.lean=np.clip(.32*(angle-self.rest_angle),-.18,.18)
        self.slide(dt,substeps=substeps)

    def slide(self,dt,*,substeps=None):
        """Advance yarn at the last measured stitch geometry."""
        self.intact=~self.failed[self.outgoing]
        # Locked junctions retain their signed transfer at failure. Both stocks
        # still account for that transfer, but a free end cannot pull on its neighbour.
        connected=self.linked & self.intact & self.intact[self.right]
        # One signed material length per junction. Increasing it removes the
        # exact same length from the left loop as it adds to the right loop.
        count=substeps if substeps is not None else max(1,int(np.ceil(dt/.010)))
        for _ in range(count):
            flow=self.ledger[self.outgoing]
            stock=self.rest_length+self.ledger[self.incoming]-flow
            tension=self.load-1.35*np.log(np.maximum(stock/self.rest_length,.02))
            drive=tension[self.right]-tension-.85*flow/self.capacity
            # Rate-dependent sliding with a dry-friction dead band at rest.
            speed=np.sign(drive)*np.maximum(abs(drive)-.005,0)*self.capacity/.18
            proposed=np.clip(flow+speed*(dt/count),-.12*self.capacity,.12*self.capacity)
            self.ledger[self.outgoing]=np.where(connected,proposed,flow)
        self.flow=self.ledger[self.outgoing]
        self.stock=self.rest_length+self.ledger[self.incoming]-self.flow
        feed=(self.stock-self.rest_length)/self.rest_length
        self.sliding=bool(np.any(connected & (abs(proposed-flow)>1e-9)))
        self.feed=feed
        self.head=np.clip(1+1.9*self.bias+3.8*feed,.65,1.95)
        self.neck=np.clip(1-.95*self.bias-2.4*feed,.54,1.44)
