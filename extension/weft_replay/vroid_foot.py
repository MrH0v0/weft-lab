"""Dry garment fit and material coordinates for the project's VRoid foot."""
import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon
from mathutils.kdtree import KDTree

from .live_membrane import unique_edges


def is_vroid_foot(source, rest):
    rig = source.find_armature()
    return (rig is not None and "J_Bip_R_Foot" in rig.data.bones
            and rest[:, 2].max() > .30
            and np.any(abs(rest[:, 2]-.14) < .012)
            and np.any((rest[:, 2] < .08) & (rest[:, 1] > .09)))


def smoothstep(value):
    t = np.clip(value, 0, 1)
    return t*t*(3-2*t)


def sew_foot_boundary(points, triangles, rest, source_ids=None):
    """Weld near-coincident duplicated UV borders in the posed VRoid foot."""
    half_edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    edges, use_count = np.unique(half_edges, axis=0, return_counts=True)
    rim = edges[(use_count == 1) & (rest[edges, 2] < .208).all(axis=1)]
    border = np.unique(rim)
    if not len(border):
        return (points, triangles, rest) if source_ids is None else (points, triangles, rest, source_ids)
    tree = KDTree(len(border))
    for i in border:
        tree.insert(points[i], int(i))
    tree.balance()
    merged = np.arange(len(points))
    tolerance = .000012

    def root(index):
        while merged[index] != index:
            index = merged[index]
        return index

    for i in border:
        for _, other, _ in tree.find_range(points[i], tolerance):
            if other >= i or np.linalg.norm(rest[i]-rest[other]) > tolerance:
                continue
            a, b = root(i), root(other)
            merged[max(a, b)] = min(a, b)
    for i in border:
        merged[i] = root(i)
    keep, indices = np.unique(merged, return_inverse=True)
    faces = indices[triangles]
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    result = points[keep], faces[valid], rest[keep]
    return result if source_ids is None else (*result, source_ids[keep])


def close_toe_boundaries(triangles, rest):
    """Bridge the nail rims and cover the four gaps between the toes."""
    directed = triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    _, inverse, uses = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    toe = (rest[:, 1] > .072) & (rest[:, 2] < .082)
    rim = directed[(uses[inverse] == 1) & toe[directed].all(axis=1)]
    following = dict(rim)
    rings = []
    while following:
        start = current = next(iter(following))
        ring = []
        while current in following:
            ring.append(current)
            current = following.pop(current)
        if current == start and len(ring) >= 3:
            rings.append(np.array(ring))

    caps, nails = [], []
    for ring in rings:
        if np.ptp(rest[ring, 1]) > .015:
            reverse = ring[::-1]
            faces = tessellate_polygon([[Vector(position) for position in rest[reverse]]])
            if faces:
                caps.append(reverse[np.array(faces)])
        else:
            nails.append((ring, rest[ring].mean(0)))

    while nails:
        ring, center = nails.pop()
        matches = [(np.linalg.norm(other_center[:2]-center[:2]), index)
                   for index, (other, other_center) in enumerate(nails)
                   if len(other) == len(ring) and .001 < abs(other_center[2]-center[2]) < .004]
        if not matches:
            continue
        distance, index = min(matches)
        if distance > .002:
            continue
        other = nails.pop(index)[0][::-1]
        start = min(range(len(other)),
                    key=lambda shift: np.square(rest[ring]-rest[np.roll(other, shift)]).sum())
        other = np.roll(other, start)
        far = np.roll(ring, -1)
        caps.append(np.column_stack((far, ring, other)))
        caps.append(np.column_stack((far, other, np.roll(other, -1))))
    return np.concatenate((triangles, *caps))


def foot_chart(rest, leg_chart):
    """Follow the foot's length, turning the courses around the heel."""
    x, y, z = rest.T
    # The quarter turn is tangent to the ankle axis and the forefoot axis.
    pivot_y, pivot_z, radius = .037, .136, .075
    angle = np.arctan2(np.maximum(pivot_z-z, 0), np.maximum(pivot_y-y, 0))
    cross = radius - np.hypot(y-pivot_y, z-pivot_z)
    length = pivot_z - radius*angle
    leg = z >= pivot_z
    cross[leg], length[leg] = y[leg] - (pivot_y-radius), z[leg]
    fore = (y >= pivot_y) & ~leg
    dorsal = pivot_z - radius - .013*smoothstep((y[fore]-pivot_y)/.084)
    cross[fore] = z[fore] - dorsal
    length[fore] -= y[fore] - pivot_y
    ankle_x = np.median(x[abs(z-.14)<.012])
    toe_x = np.median(x[(y>.082) & (z<.08)])
    center_x = ankle_x + (toe_x-ankle_x)*smoothstep((y+.025)/.105)
    around = np.arctan2(cross, x-center_x)/(2*np.pi)
    blend = smoothstep((.192-z)/.047)
    turn = around - leg_chart[:, 0]
    turn -= np.round(turn)
    return np.column_stack((leg_chart[:, 0] + blend*turn,
                            leg_chart[:, 1] + blend*(length/.002 - leg_chart[:, 1])))


def round_foot_surface(points, triangles, rest, skin_tree):
    """Fair the closed toe and ankle locally while retaining the skin clearance."""
    z = rest[:, 2]
    foot = z < .193
    edges = unique_edges(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), len(points))
    edges = edges[foot[edges].any(axis=1)]
    a, b = np.concatenate((edges, edges[:, ::-1])).T
    weights = 1 / np.maximum(.0006, np.linalg.norm(points[b]-points[a], axis=1))
    total = np.bincount(a, weights, minlength=len(points))
    weight = smoothstep((.193-z)/.061)
    toe = smoothstep((rest[:, 1]-.065)/.037) * smoothstep((.095-z)/.022)
    limit = (.00085 + .00155*toe) * weight
    smoothed = points.copy()
    for factor in (.72, .65, .62, .6*toe, .6*toe, .6*toe, -.38):
        differences = (smoothed[b]-smoothed[a])*weights[:, None]
        mean = np.column_stack([np.bincount(a, axis, minlength=len(points)) for axis in differences.T])
        smoothed += (factor*weight/np.maximum(total, 1))[:, None]*mean
    shift = smoothed - points
    length = np.linalg.norm(shift, axis=1)
    shift *= np.minimum(1., limit/np.maximum(length, 1e-12))[:, None]
    result = points.copy()
    for i in np.flatnonzero(length > .000008):
        start = Vector(points[i])
        nearest, normal, _, distance = skin_tree.find_nearest(start)
        if distance > .004 or any(normal.dot(other)<-.8 for _, other, _, _ in
                                  skin_tree.find_nearest_range(start, distance+.000002)):
            continue
        moved = start + Vector(shift[i])
        hit, outward, _, _ = skin_tree.find_nearest(moved)
        minimum = min((start-nearest).dot(normal), .00015)
        new_gap = (moved-hit).dot(outward)
        if new_gap < minimum:
            moved += (minimum-new_gap)*outward
            hit, outward, _, _ = skin_tree.find_nearest(moved)
            if (moved-hit).dot(outward) < minimum-1e-6:
                continue
        # Preserve the bounded fairing step and ambiguous skin faces.
        if (moved-start).length <= limit[i]+1e-7:
            result[i] = moved
    return result
