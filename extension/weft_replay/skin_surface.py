"""Exact skin queries with a small working BVH around the moving garment."""
from mathutils.bvhtree import BVHTree
import numpy as np


class SkinBlocks:
    """Group nearby triangles once, then bound each group in the current pose."""
    size = .026

    def __init__(self, vertices, indices):
        centers = vertices[indices].mean(axis=1)
        offset = centers - centers.min(0)
        cells = np.floor(offset / max(self.size, float(offset.max()) / 10000.)).astype(np.int64)
        shape = cells.max(0) + 1
        _, groups, self.counts = np.unique(
            np.ravel_multi_index(cells.T, shape), return_inverse=True, return_counts=True)
        self.faces = np.argsort(groups, kind='stable').astype(np.int32)
        # Each vertex is read only once per block, including on a pose change.
        keys = np.unique((groups[:, None] * len(vertices) + indices).ravel())
        block_ids, self.vertices = np.divmod(keys, len(vertices))
        self.starts = np.r_[0, np.flatnonzero(block_ids[1:] != block_ids[:-1]) + 1]
        self.representatives = self.vertices[self.starts]


class PosedSkinBlocks:
    def __init__(self, layout, vertices, indices):
        self.layout, self.vertices, self.indices = layout, vertices, indices
        self.bounds = None
        self.tiles = {}
        self.close_tiles = {}

    def faces_in(self, lower, upper):
        if self.bounds is None:
            points = self.vertices[self.layout.vertices]
            starts = self.layout.starts
            self.bounds = (np.minimum.reduceat(points, starts, axis=0),
                           np.maximum.reduceat(points, starts, axis=0))
        lo, hi = self.bounds
        active = ((hi >= lower) & (lo <= upper)).all(axis=1)
        return self.layout.faces[np.repeat(active, self.layout.counts)]

    def tile_for(self, point):
        x, y, z = point
        size = self.layout.size
        key = (int(x // size), int(y // size), int(z // size))
        if key not in self.tiles:
            center = (np.asarray(key) + .5) * size
            delta = self.vertices[self.layout.representatives] - center
            # Every triangle that can be closest intersects the search box.
            # Bound the distance for all points in this tile by an actual
            # skin vertex and the tile's half diagonal.
            radius = np.sqrt(np.einsum('ij,ij->i', delta, delta).min()) + .866025404 * size + 2e-6
            bounds = np.asarray((center - .5 * size, center + .5 * size))
            self.tiles[key] = SkinSurface(self.vertices, self.indices, bounds, padding=radius, blocks=self)
        return self.tiles[key]

    def close_tile_for(self, point):
        # A short local query need not include every skin triangle that may
        # be nearest to a distant point elsewhere in the cell.
        size=self.layout.size
        key=tuple(int(value//size) for value in point)
        if key not in self.close_tiles:
            lower=np.asarray(key)*size
            bounds=np.asarray((lower,lower+size))
            self.close_tiles[key]=SkinSurface(self.vertices,self.indices,bounds,padding=.003,blocks=self)
        return self.close_tiles[key]


class TrianglePositions:
    """Keep triangle coordinates in the shared vertex array until queried."""
    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices

    def __getitem__(self, index):
        return self.vertices[self.indices[index]]


class SkinSurface:
    def __init__(self, vertices, indices, garment, padding=.012, fallback=None, exact=False,
                 blocks=None, search_tiles=False, close_first=False):
        self.vertices, self.indices = vertices, indices
        self.positions = TrianglePositions(vertices, indices)
        self.lower = tuple(map(float, garment.min(0) - padding))
        self.upper = tuple(map(float, garment.max(0) + padding))
        self._full = None
        self._near = None
        self._face_ids = None
        self._built_near = exact
        self._fallback = fallback
        self._exact = exact
        self._blocks = blocks
        self._search_tiles = search_tiles
        self._close_first = close_first and blocks is not None and not exact

    def _build_near(self):
        if self._built_near:
            return
        vertices,indices=self.vertices,self.indices
        # Rounding outwards retains any skin triangle crossing the box, even
        # when all three corners are outside it.
        selected = (self._blocks.faces_in(np.asarray(self.lower) - 1e-7, np.asarray(self.upper) + 1e-7)
                    if self._blocks else np.arange(len(indices), dtype=np.int32))
        compact_region = self._blocks is not None and len(selected) < len(vertices) // 8
        if compact_region:
            nearby = vertices[indices[selected]]
            lo = np.asarray(self.lower) - 1e-7
            hi = np.asarray(self.upper) + 1e-7
            selected = selected[~(((nearby < lo).all(axis=1) | (nearby > hi).all(axis=1)).any(axis=1))]
        else:
            for axis, (low, high) in enumerate(zip(self.lower, self.upper)):
                a,b,c = indices[selected].T
                below = vertices[:,axis] < low - 1e-7
                above = vertices[:,axis] > high + 1e-7
                selected = selected[~(below[a]&below[b]&below[c] | above[a]&above[b]&above[c])]
        if 0<len(selected)<len(indices):
            if self._blocks:
                selected.sort()
            faces = indices[selected]
            if compact_region:
                used, compact = np.unique(faces.ravel(), return_inverse=True)
                compact = compact.reshape(faces.shape)
            else:
                used = np.zeros(len(vertices), dtype=bool)
                used[faces.ravel()] = True
                compact = (np.cumsum(used, dtype=np.int32) - 1)[faces]
            self._near = BVHTree.FromPolygons(vertices[used].tolist(), compact.tolist(), all_triangles=True)
            self._face_ids = selected.tolist()
        self._built_near = True

    def full(self):
        if self._full is None:
            self._full = BVHTree.FromPolygons(self.vertices.tolist(), self.indices.tolist(), all_triangles=True)
        return self._full

    def _available(self, point):
        x,y,z = point
        a,b,c = self.lower
        d,e,f = self.upper
        return min(x-a,y-b,z-c,d-x,e-y,f-z)-1e-7

    def find_nearest(self, point, distance=None, *, clear_at=None):
        if self._fallback is not None and self._available(point)<=0.:
            return self._fallback.find_nearest(point,distance,clear_at=clear_at)
        if self._close_first and not self._built_near:
            tile=self._blocks.close_tile_for(point)
            limit=tile._available(point)
            if distance is not None:
                limit=min(distance,limit)
            tile._build_near()
            if tile._near is not None:
                hit,n,index,gap=_nearest(tile._near,point,limit)
                # The complete region closer than limit is inside the tile.
                # The same Blender closest-face query can resolve a fold here.
                if index is not None and gap<limit:
                    return hit,n,tile._face_ids[index],gap
        self._build_near()
        if self._near is None:
            return self._nearest_outside(point,distance,clear_at)
        limit = self._available(point)
        if limit > 0.:
            if distance is not None:
                limit = min(distance, limit)
            hit, normal, face, gap = _nearest(self._near, point, limit)
            if face is not None:
                return hit, normal, self._face_ids[face], gap
            if distance == limit:
                return hit, normal, face, gap
        return self._nearest_outside(point,distance,clear_at)

    def _nearest_outside(self, point, distance, clear_at):
        if self._fallback is not None:
            return self._fallback.find_nearest(point,distance,clear_at=clear_at)
        if self._exact:
            return self.full().find_nearest(point) if distance is None else self.full().find_nearest(point,distance)
        if self._search_tiles and self._full is None:
            tile = self._blocks.tile_for(point)
            tile._build_near()
            if tile._near is not None:
                limit = tile._available(point)
                if distance is not None:
                    limit = min(distance, limit)
                hit, normal, index, gap = _nearest(tile._near, point, limit)
                if index is not None:
                    # Preserve the original tree's choice of face on a tie.
                    # That choice also anchors the grab and skin diagnostics.
                    neighbors = tile._near.find_nearest_range(point, min(limit, gap + 1e-7))
                    ties = [(p,n) for p,n,i,d in neighbors if i != index and abs(d - gap) <= 1e-7]
                    # If every tied face clears the caller's contact floor,
                    # the choice is immaterial to that contact decision.
                    if not ties or (clear_at is not None and (point-hit).dot(normal) >= clear_at
                                    and all((point-p).dot(n) >= clear_at for p,n in ties)):
                        return hit, normal, tile._face_ids[index], gap
                elif limit == distance:
                    return hit, normal, index, gap
        return _nearest(self.full(),point,distance)

    def find_nearest_range(self, point, distance):
        if self._close_first and not self._built_near:
            tile=self._blocks.close_tile_for(point)
            if distance<=tile._available(point):
                tile._build_near()
                if tile._near is not None:
                    return [(p,n,tile._face_ids[i],d) for p,n,i,d in tile._near.find_nearest_range(point,distance)]
        if distance <= self._available(point):
            self._build_near()
            if self._near is not None:
                return [(p,n,self._face_ids[i],d) for p,n,i,d in self._near.find_nearest_range(point,distance)]
        if self._fallback is not None:
            return self._fallback.find_nearest_range(point,distance)
        if self._search_tiles and self._full is None:
            tile = self._blocks.tile_for(point)
            if distance <= tile._available(point):
                tile._build_near()
                if tile._near is not None:
                    return [(p,n,tile._face_ids[i],d) for p,n,i,d in tile._near.find_nearest_range(point,distance)]
        return self.full().find_nearest_range(point,distance)

    def ray_cast(self, *args):
        return self.full().ray_cast(*args)

    def overlap_garment(self, tree, points):
        self._build_near()
        if (self._near is not None and (points.min(0)>=self.lower).all()
                and (points.max(0)<=self.upper).all()):
            return tree.overlap(self._near)
        return tree.overlap(self.full())


def _nearest(tree, point, distance):
    # Most queries start less than a millimetre from the skin. A miss
    # expands back to the exact query the caller requested.
    if distance is None or distance>.0005:
        near = tree.find_nearest(point,.0005)
        if near[2] is not None and near[3]<.0005:
            return near
    return tree.find_nearest(point) if distance is None else tree.find_nearest(point,distance)
