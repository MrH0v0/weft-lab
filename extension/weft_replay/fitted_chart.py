"""Knit coordinates for the trial stock fitted to Lisa's shin and foot."""
import numpy as np

from .binding import coordinates
from .live_membrane import knit_chart


def leg_sections(source):
    xyz = coordinates(source.data)
    indices = np.empty((len(source.data.edges),2),np.int32)
    source.data.edges.foreach_get("vertices",indices.ravel())
    first,second = xyz[indices].transpose(1,0,2)
    z0,z1 = first[:,2],second[:,2]
    levels = np.linspace(xyz[:,2].min(),xyz[:,2].max(),128)[1:-1]
    centers = []
    for level in levels:
        crossing = (np.minimum(z0,z1)<=level)&(level<np.maximum(z0,z1))
        fraction = (level-z0[crossing])/(z1[crossing]-z0[crossing])
        section = first[crossing,:2]+fraction[:,None]*(second[crossing,:2]-first[crossing,:2])
        centers.append((section.min(axis=0)+section.max(axis=0))*.5)
    return levels,np.asarray(centers)


def fitted_chart(source, points, cross_sections=True):
    ankle = np.asarray(source["weft_candidate_ankle"])
    forward = np.asarray(source["weft_candidate_toe"])-ankle
    forward /= np.linalg.norm(forward)
    side = np.cross(forward,(0,0,1))
    side /= np.linalg.norm(side)
    up = np.cross(side,forward)
    if cross_sections:
        levels,centers = leg_sections(source)
        z = points[:,2]
        x,y = (points[:,:2]-np.column_stack([np.interp(z,levels,c) for c in centers.T])).T
        uv = np.column_stack((np.arctan2(y,x)/(2*np.pi),z/.002))
    else:
        uv = knit_chart(points)
    foot = points-ankle
    along = foot@forward
    levels = np.linspace(0,along.max(),30)
    radial = foot@np.column_stack((side,up))
    centers = []
    for level in levels:
        ring = radial[abs(along-level)<.008]
        centers.append((ring.min(axis=0)+ring.max(axis=0))*.5 if len(ring) else (np.nan,np.nan))
    centers = np.asarray(centers)
    valid = np.isfinite(centers[:,0])
    radial -= np.column_stack([np.interp(along,levels[valid],c) for c in centers[valid].T])
    turn = np.arctan2(radial[:,1],radial[:,0])/(2*np.pi)-uv[:,0]
    blend = np.clip((ankle[2]+.024-points[:,2])/.075,0,1)
    blend = blend**2*(3-2*blend)
    uv[:,0] += blend*(turn-np.round(turn))
    uv[:,1] += blend*((ankle[2]-along)/.002-uv[:,1])
    return uv
