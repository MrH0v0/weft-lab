"""Water surface reconstruction shared by 04 live and render previews."""

import numpy as np

from .wet_film import GRAVITY, PULSE, pulse


CONTACT_DEPTH_M = .00016
FILM_CLOSURE_OFFSET_M = -.000028
LANE_RANGES = ((0, 1), (1, 4), (4, 9), (9, 12), (12, 14), (14, 16), (16, 17))
EDGE_PULSES = ((0., .28), (.37, .63), (.73, 1.))
FRINGE_PULSES = ((0., .38), (.45, 1.))


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def air_paths(pour, start, end, part):
    """Integrate flux along the part of each lane still in flight."""
    lanes = slice(start, end)
    low, high = part
    age = pour.age - pour.delay[lanes]
    low_t = np.maximum(0., age - PULSE * high)
    high_t = np.clip(age - PULSE * low, 0., pour.flight[lanes])
    span = np.maximum(0., high_t - low_t)
    if not np.any(span > 0):
        return None
    sample = np.linspace(0., 1., 39)
    travel = np.minimum(low_t, high_t) + sample[:, None] * span
    initial_flux = pour.volume * 1e-6 * pour.shares[lanes]
    phase = np.clip((age - travel) / PULSE, 0., 1.)
    mass_per_sample = (1. - np.cos(2. * np.pi * phase)) / PULSE * initial_flux * span
    volume = float(np.sum(initial_flux * (pulse(age - low_t) - pulse(age - high_t)) * (span > 0)))
    if volume < 2e-12:
        return None
    weight = mass_per_sample + 1e-8 * (initial_flux * span)
    weight /= weight.sum(axis=1, keepdims=True)
    ray = (pour.origin[lanes] + travel[..., None] * pour.velocity[lanes]
           + .5 * travel[..., None]**2 * GRAVITY)
    spine = np.einsum("ij,ijk->ik", weight, ray)
    motion = np.einsum("ij,ijk->ik", weight, (pour.velocity[lanes] + travel[..., None] * GRAVITY) * span[:, None])
    approach = np.einsum("ij,ij->i", weight, pour.flight[lanes] - travel)
    return spine, motion, mass_per_sample.sum(axis=1), volume, approach


def tube(piece, crosswise, age, index, edge):
    centre, motion, rate, volume, approach = piece
    along = unit(motion)
    across = unit(crosswise - (along @ crosswise)[:, None] * along)
    normal = np.cross(along, across)
    length = np.linalg.norm(np.diff(centre, axis=0), axis=1).sum()
    natural_radius = (3. * volume / (4. * np.pi))**(1. / 3.)
    if length < 2.2 * natural_radius:
        mid = len(centre) // 2
        # A short packet rounds over instead of ending as a wide flat cap.
        t = np.linspace(-1., 1., len(centre))
        half = max(.5 * length, natural_radius)
        centre = centre[mid] + (t * half)[:, None] * along[mid]
        area = np.pi * natural_radius**3 / half * np.maximum(1e-5, 1. - t**2)
    else:
        area = rate / np.maximum(np.linalg.norm(motion, axis=1), length * .3)
        phase = np.linspace(0., 1., len(area))
        cap = np.minimum(1., np.minimum(phase, 1. - phase) / (.18 if edge else .085))
        area = np.maximum(1e-12 * area.max(), area * cap)
    phase = np.linspace(0., 1., len(area))
    head = np.clip(approach / .019, 0., 1.)
    azimuth = index * 2.39 + 115. * age
    ripple = np.sin(phase * 12. + azimuth) + .3 * np.sin(phase * 21. - 84. * age + index)
    bow = np.sin(phase * (6.5 + .56 * index) - azimuth) + .26 * np.sin(phase * 22. + .7 * azimuth)
    centre = centre + head[:, None] * (normal * (.00076 * ripple[:, None]) + across * (.00086 * bow[:, None]))
    pinch = .62 + .38 * np.sin(phase * (13.5 + index * .65) + azimuth)
    area *= (.28 + .72 * pinch) * (1. + .19 * np.sin(31. * phase - .5 * azimuth))
    ds = np.linalg.norm(np.diff(centre, axis=0), axis=1)
    area *= volume / np.maximum(np.dot(.5 * (area[:-1] + area[1:]), ds), 1e-18)
    radius = np.maximum(np.sqrt(np.maximum(area, 1e-20) / np.pi), .000009)
    aspect = (1.04 if edge else 1.26) + .28 * np.sin(phase * 9. - 67. * age + index * .9)
    radial = np.linspace(0., 2. * np.pi, 16, endpoint=False)
    rings = centre[:, None, :] + radius[:, None, None] * (
        (aspect[:, None] * np.cos(radial))[:, :, None] * across[:, None, :]
        + (np.sin(radial) / aspect[:, None])[:, :, None] * normal[:, None, :])
    offsets = np.arange(rings.shape[0] * rings.shape[1]).reshape(rings.shape[:2])
    adjacent = np.roll(offsets, -1, axis=1)
    faces = np.stack((offsets[:-1], adjacent[:-1], adjacent[1:], offsets[1:]), axis=-1).reshape(-1, 4).tolist()
    faces += [offsets[0, ::-1].tolist(), offsets[-1].tolist()]
    surface = np.full(len(faces), .73 if edge else .81, dtype=np.float32)
    roughness = np.repeat(.045 + .075 * (.5 + .5 * ripple[:-1] / 1.3), rings.shape[1])
    roughness = np.r_[roughness, .12, .12].astype(np.float32)
    return rings.reshape(-1, 3), faces, surface, roughness


def ligaments(pour):
    """Keep the 04 lane timing and render a rounded stream with broken edges."""
    targets = pour.points(pour.flight)
    across = unit(targets[-1] - targets[0])
    pieces = []
    for i, (start, end) in enumerate(LANE_RANGES):
        edge = end - start == 1
        parts = EDGE_PULSES if edge else FRINGE_PULSES if i in (1, 5) else ((0., 1.),)
        packets = [piece for part in parts if (piece := air_paths(pour, start, end, part)) is not None]
        if not packets:
            continue
        reference = sum(packet[3] for packet in packets)
        if len(parts) > 1:
            original = air_paths(pour, start, end, (0., 1.))
            gain = min(1.45, original[3] / reference)
        else:
            gain = 1.
        for points, motion, rate, volume, approach in packets:
            pieces.append(tube((points, motion, rate, volume * gain, approach), across, pour.age, i, edge))
    return merge(pieces)


def merge(pieces):
    pieces = [model for model in pieces if model is not None]
    if not pieces:
        return None
    vertices, faces, surface, roughness = [], [], [], []
    count = 0
    for points, polys, alpha, rough in pieces:
        faces.extend(tuple(count + index for index in face) for face in polys)
        vertices.append(points)
        surface.append(alpha)
        roughness.append(rough)
        count += len(points)
    return np.concatenate(vertices), faces, np.concatenate(surface), np.concatenate(roughness)


def triangles(shape):
    ids = np.arange(np.prod(shape)).reshape(shape)
    a,b,c,d = (i.ravel() for i in (ids[:-1,:-1], ids[:-1,1:], ids[1:,:-1], ids[1:,1:]))
    return np.vstack((np.column_stack((a,b,c)), np.column_stack((b,d,c))))


def upsample(grid):
    """Bilinear 2× refinement that preserves original samples."""
    rows,cols,*tail = grid.shape
    out = np.empty((2*rows-1,2*cols-1,*tail),dtype=grid.dtype)
    out[::2,::2] = grid
    out[::2,1::2] = (grid[:,:-1]+grid[:,1:])*.5
    out[1::2] = (out[:-1:2]+out[2::2])*.5
    return out


def smooth_depth(depth, valid, area):
    """Smooth the reconstructed height on valid sites and preserve its integral."""
    before = (depth*area).sum()
    result = depth.copy()
    for _ in range(2):
        weights = valid.astype(float).copy()*2
        acc = result*weights
        for axis in (0,1):
            a,b = (slice(None),slice(1,None)),(slice(None),slice(None,-1))
            if axis==0:
                a,b = a[::-1],b[::-1]
            local = valid[a]&valid[b]
            acc[a] += result[b]*local
            acc[b] += result[a]*local
            weights[a] += local
            weights[b] += local
        result = acc/np.maximum(weights,1.)
    return result*(before/max((result*area).sum(),1e-20))


class PackedFaces:
    def __init__(self,sizes,corners):
        self.sizes=np.ascontiguousarray(sizes,dtype=np.int32)
        self.corners=np.ascontiguousarray(corners,dtype=np.int32)

    def __len__(self):
        return len(self.sizes)


def hull(points, depth, valid, *, threshold, offset=0., center=False, fallback=None, grounded=False, packed=False, domain_feather=0.):
    """Clip the liquid footprint, cap its perimeter and supply real ray lengths."""
    if depth.max() <= threshold:
        return None
    points = upsample(points)
    valid = upsample(valid.astype(float)) >= .999
    depth = upsample(depth) * valid
    row,col = np.gradient(points, axis=(0,1))
    normal = np.cross(col,row)
    normal /= np.maximum(np.linalg.norm(normal,axis=2,keepdims=True), 1e-10)
    if fallback is not None:
        use = normal@fallback > 0.
        normal[~use] = fallback
    xyz = points.reshape(-1,3)
    n = normal.reshape(-1,3)
    v = depth.ravel()
    base = xyz+n*offset
    if center:
        base -= n*(.5*v)[:,None]
    cap = base + n*v[:,None]
    tri = triangles(depth.shape)
    inside = v[tri] > threshold
    keep = inside.any(axis=1) & valid.ravel()[tri].all(axis=1)
    tri, inside = tri[keep], inside[keep]
    if not len(tri):
        return None

    # Walk each triangle in winding order, alternating original vertices
    # and clipped edges. Keep the first occurrence of each shared edge.
    next_vertex = np.roll(tri, -1, axis=1)
    crossing = inside != np.roll(inside, -1, axis=1)
    mask = np.stack((inside, crossing), axis=-1).reshape(-1, 6)
    a = np.repeat(tri, 2, axis=1)[mask]
    b = np.stack((tri, next_vertex), axis=-1).reshape(-1, 6)[mask]
    keys = np.minimum(a, b) * len(v) + np.maximum(a, b)
    _, first, corners = np.unique(keys, return_index=True, return_inverse=True)
    order = first.argsort()
    indices = np.empty_like(order)
    indices[order] = np.arange(len(order))
    corners = indices[corners]
    a, b = a[first[order]], b[first[order]]
    t = np.divide(threshold-v[a], v[b]-v[a], out=np.zeros(len(a)), where=a!=b)
    top = cap[a] + t[:, None] * (cap[b]-cap[a])
    bottom = base[a] + t[:, None] * (base[b]-base[a])
    verts = np.stack((bottom, top), axis=1).reshape(-1, 3)

    sizes = mask.sum(axis=1)
    stops = sizes.cumsum()
    starts = stops - sizes
    # Blender takes flat loop indices. Interleave the top fan with its
    # reversed bottom fan in the original polygon order.
    group_start=np.repeat(starts,sizes)
    offset_in_face=np.arange(len(corners))-group_start
    cap_corners=np.empty(2*len(corners),dtype=np.int32)
    cap_corners[2*group_start+offset_in_face]=2*corners+1
    cap_corners[2*np.repeat(stops,sizes)-1-offset_in_face]=2*corners

    next_corner = np.roll(corners, -1)
    next_corner[stops-1] = corners[starts]
    edge_key = np.minimum(corners, next_corner) * len(a) + np.maximum(corners, next_corner)
    _, edge, count = np.unique(edge_key, return_index=True, return_counts=True)
    boundary = np.sort(edge[count == 1])
    start_side,end_side=2*corners[boundary],2*next_corner[boundary]
    sides=np.column_stack((start_side+1,start_side,end_side,end_side+1))
    caps=2*len(sizes)
    outer=PackedFaces(np.r_[np.repeat(sizes,2),np.full(len(sides),4)],np.r_[cap_corners,sides.ravel()])
    if not packed:
        outer=[tuple(face.tolist()) for face in np.split(outer.corners,outer.sizes.cumsum()[:-1])]

    surface = np.ones(len(outer), 'f')
    roughness = np.full_like(surface, .045)
    if grounded:
        # The bottom belongs to the yarn / pore environment.
        height = (1-t)*v[a] + t*v[b]
        top_depth = np.add.reduceat(height[corners], starts) / sizes
        surface[1:caps:2] = 0.
        surface[:caps:2] = np.clip(top_depth/.00025, .17, 1.)
        surface[caps:] = .22
        if domain_feather:
            # Free water leaving the measured patch loses its visible cap
            # gradually. All physical heights, positions and fluxes are retained.
            iy,ix=np.indices(depth.shape)
            margin=np.minimum(np.minimum(ix,depth.shape[1]-1-ix),np.minimum(iy,depth.shape[0]-1-iy)).ravel()
            w=np.clip(((1-t)*margin[a]+t*margin[b])/domain_feather,0.,1.)
            w=w*w*(3.-2.*w)
            surface[:caps:2]*=np.add.reduceat(w[corners],starts)/sizes
            surface[caps:]*=.5*(w[corners[boundary]]+w[next_corner[boundary]])
        roughness[:] = .13
    return verts, outer, surface, roughness


FILM_CUTOFF_M = .000020


def impact_beads(pour, normal, free_volume):
    """Use a small volume from the displayed film for the unresolved splash rim."""
    targets = pour.points(pour.flight)
    drops = []
    for j, lane in enumerate((2, 5, 7, 11, 13, 4, 9, 15, 6, 12, 3)):
        emission = .009 + .0054 * j
        age = pour.age - pour.flight[lane] - pour.delay[lane] - emission
        if not 0. <= age < .062:
            continue
        side = unit(pour.tangent[lane])
        speed = .095 + .041 * (j % 4)
        velocity = normal * speed + side * (.13 + .04 * (j % 3))
        velocity[2] += .18 + .03 * ((j * 7) % 5)
        taper = np.clip(age / .006, 0., 1.) * np.clip((.062 - age) / .020, 0., 1.)
        radius = (.00033 + .000055 * ((j * 3) % 6)) * taper**.4
        if radius < .000045:
            continue
        centre = targets[lane] + age * velocity + .5 * age**2 * GRAVITY + normal * (radius + .00023)
        drops.append((centre, velocity + age * GRAVITY, radius))

    total = sum(4. * np.pi * radius**3 / 3. for _, _, radius in drops)
    if not drops or free_volume <= 0.:
        return None, 0.
    scale = min(1., (.028 * free_volume / total)**(1. / 3.))
    pieces = []
    for index, (centre, velocity, radius) in enumerate(drops):
        radius *= scale
        along = unit(velocity)
        crosswise = unit(np.cross(along, normal))
        param = np.linspace(-1., 1., 11)
        arc = centre + (param * radius * 1.21)[:, None] * along
        motion = np.broadcast_to(along * (2. * radius), arc.shape)
        vol = 4. * np.pi * radius**3 / 3.
        model = tube((arc, motion, 1. - param**2, vol, np.zeros(len(param))), crosswise, pour.age, index, True)
        points, faces, _, _ = model
        pieces.append((points, faces, np.full(len(faces), .87), np.full(len(faces), .045)))
    return merge(pieces), total * scale**3


def attached_film(points, depth, area, normal, elapsed, *, spray_volume=0., valid=None, packed=False):
    """Resolve free water into uneven shallow bands on the moving Knit lattice."""
    free = np.maximum(0., depth - CONTACT_DEPTH_M)
    if free.max() <= FILM_CUTOFF_M:
        return None
    volume = max(0., float(np.sum(free * area)) - spray_volume)
    if valid is None:
        valid = np.ones(free.shape, dtype=bool)
    smooth = smooth_depth(free, valid, area)
    height, cell_area = upsample(smooth), upsample(area) * .25
    cell_area[[0, -1]] *= .5
    cell_area[:, [0, -1]] *= .5
    y, x = np.indices(height.shape, dtype=float) * .5
    bend = x + .29 * np.sin(y * .49 + .6) + .16 * np.cos(y * 1.24)
    separation = .60 * np.sin(2. * np.pi * bend / 1.8 + y * .16)
    separation += .40 * np.sin(2. * np.pi * bend / 1.13 - y * .15 + elapsed * 9.)
    height *= 1. + .46 * separation
    height *= volume / np.sum(height * cell_area)
    valid = upsample(valid.astype(float)) >= .999
    model = hull(upsample(points), height, valid, threshold=FILM_CUTOFF_M,
                 offset=FILM_CLOSURE_OFFSET_M, fallback=normal, grounded=True, packed=packed, domain_feather=16.)
    if model is None:
        return None
    vertices, faces, air, _ = model
    # Preserve real shallow and thick regions while reducing the low-resolution
    # cap's single strong reflection. The ground-facing interface is still zero.
    roughness = np.asarray(.085 + .055 * (1. - air), dtype=np.float32)
    return vertices, faces, air * .78, roughness
