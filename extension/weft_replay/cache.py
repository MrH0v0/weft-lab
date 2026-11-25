"""Sample the packaged replay in metres. Browser Y-up -> Blender Z-up."""
import json
import struct
from functools import lru_cache
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data"

def xyz(points):
    a = np.asarray(points, dtype=np.float32)
    return np.stack((a[...,0], -a[...,2], a[...,1]), axis=-1)

@lru_cache(maxsize=1)
def frames():
    return json.loads((DATA / "frames.json").read_text())["frames"]

@lru_cache(maxsize=1)
def assets():
    raw = (DATA / "leg.glb").read_bytes()
    n = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20+n])
    def value(index, dims):
        ac = doc["accessors"][index]
        v = doc["bufferViews"][ac["bufferView"]]
        return np.frombuffer(raw, "<u4" if ac["componentType"]==5125 else "<f4",
                             ac["count"]*dims, 28+n+v["byteOffset"]).reshape(-1,dims)
    result = {}
    for m in doc["meshes"]:
        p=m["primitives"][0]; a=p["attributes"]
        result[m["name"]]=(xyz(value(a["POSITION"],3)),value(p["indices"],1).reshape(-1,3),
                           value(a["TEXCOORD_0"],2), value(a.get("TEXCOORD_1",a["TEXCOORD_0"]),2))
    return result

@lru_cache(maxsize=1)
def displacement_table():
    samples=frames()
    return np.memmap(DATA/"geometry.i16",dtype="<i2",mode="r",
                     shape=(len(samples),samples[0]["count"],3))

@lru_cache(maxsize=2)
def delta(i):
    return np.multiply(displacement_table()[i],np.float32(.000002),dtype=np.float32)

@lru_cache(maxsize=1)
def sdf():
    return np.fromfile(DATA/"leg.sdf",dtype="<i2").reshape(111,266,71)

def grip_gap(point):
    p=(np.array(point)+[.105,.015,.10])/.003
    x,y,z=np.floor(p).astype(int)
    a,b,c=p-[x,y,z]
    return float(np.einsum('ijk,i,j,k',sdf()[z:z+2,y:y+2,x:x+2],[1-c,c],[1-b,b],[1-a,a])*.01)

class Section:
    def __init__(self,rest,faces,meta,moving):
        self.o=xyz(meta["origin"]);self.n=xyz(meta["normal"]);self.t=xyz(meta["tangent"])
        self.side=np.cross(self.t,self.n)
        # All-frame per-vertex motion bounds prune triangles without assuming
        # that a particular patch stays still on this specimen.
        pad=np.fromfile(DATA/"reach.f32",dtype="<f4")[faces].max(axis=1)+.000002 if moving else 0
        rel=rest-self.o
        keep=np.ones(len(faces),dtype=bool)
        for axis,low,high in ((self.t,-.024,.024),(self.n,-.042,.063),(self.side,0,0)):
            z=(rel@axis)[faces]
            keep &= (z.min(axis=1)<=high+pad)&(z.max(axis=1)>=low-pad)
        self.faces=faces[keep]

    def depths(self,points):
        # Independent ray/triangle intersections, evaluated on Blender mesh coordinates.
        p=points[self.faces]-self.o
        lateral=p@self.side
        hit=(lateral.min(axis=1)<=0)&(lateral.max(axis=1)>=0)
        p=p[hit]; lateral=lateral[hit]
        x,y,z=p@self.t,lateral,p@self.n
        den=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
        ok=np.abs(den)>1e-14
        x,y,z,den=x[ok],y[ok],z[ok],den[ok,None]
        dx=((np.arange(257)+.5)/257-.5)*.048-x[:,:1]
        dy=-y[:,:1]
        u=(dx*(y[:,2:3]-y[:,0:1])-dy*(x[:,2:3]-x[:,0:1]))/den
        v=(dy*(x[:,1:2]-x[:,0:1])-dx*(y[:,1:2]-y[:,0:1]))/den
        depth=z[:,:1]+u*(z[:,1:2]-z[:,:1])+v*(z[:,2:3]-z[:,:1])
        good=(u>=-1e-6)&(v>=-1e-6)&(u+v<=1.000001)&(depth>=-.0419)&(depth<=.063)
        out=np.where(good,depth,np.inf).min(axis=0,initial=np.inf)*1000
        out[~np.isfinite(out)]=np.nan
        return out

