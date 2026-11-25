"""Bind wet reservoirs to evaluated garment/skin triangles and publish one step."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .live_contact import barycentric, frames
from .contact_audit import opposing_face
from .live_membrane import unit
from .posed_contact import CLEARANCE
from .wet_film import AREAL_MASS, DT, FRONT_MM, GRAVITY, ROUGHNESS, Film, Pour


PITCH = .0025
SIZE = 35
OFFSET = (SIZE - 1) / 2
LOWER_ROWS = 18
SHAPE = (SIZE + LOWER_ROWS, SIZE)
ORIGIN = np.array((OFFSET, OFFSET + LOWER_ROWS))
CENTER = int(ORIGIN[1])*SIZE + int(ORIGIN[0])
OUTER_MM = 'weft_outer_film_mm'
SATURATION = 'weft_knit_saturation'
SKIN_MM = 'weft_skin_film_mm'


def tree_for(points, triangles):
    return BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)


def surface_axes(normal):
    across = np.cross((0., 0., 1.), normal)
    if np.linalg.norm(across) < 1e-5:
        across = np.cross((0., 1., 0.), normal)
    across = unit(across)
    return across, unit(np.cross(normal, across))


def gather(points, ids, bary):
    return np.einsum('ni,nik->nk', bary, points[ids])


def project_sites(points, tree, triangles, normal):
    ids, weights, valid = [], [], []
    for p in points:
        hit, n, face, _ = tree.ray_cast(Vector(p + .027 * normal), Vector(-normal), .054)
        ok = face is not None and np.dot(n, normal) > .14
        ids.append(face if ok else 0)
        weights.append(barycentric(triangles[face], np.asarray(hit)) if ok else (1., 0., 0.))
        valid.append(ok)
    return np.array(ids), np.array(weights), np.array(valid)


def mesh_faces():
    grid = np.arange(np.prod(SHAPE)).reshape(SHAPE)
    a, b, c, d = (v.ravel() for v in (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]))
    return np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))


TRIANGLES = mesh_faces()
EDGES = np.unique(np.sort(np.vstack((TRIANGLES[:, :2], TRIANGLES[:, 1:], TRIANGLES[:, ::2])), axis=1), axis=0)


class LayerCoverage:
    """The triangulated domain shared by the wet maps, probes and cursor input."""
    def __init__(self, points, valid):
        self.points = points
        self.faces = TRIANGLES[valid[TRIANGLES].all(axis=1)]
        self.tree = tree_for(points, self.faces) if len(self.faces) else None

    def locate(self, point):
        if self.tree is None:
            return None
        hit, _, face, distance = self.tree.find_nearest(Vector(point), PITCH*.35)
        if face is None:
            return None
        ids = self.faces[face]
        tri = self.points[ids]
        # The BVH clamps to edges. Reject points whose own projection is outside.
        if barycentric(tri, point).min() < -1e-4:
            return None
        weights = np.maximum(barycentric(tri, np.asarray(hit)), 0.)
        return ids, weights/weights.sum()


class WetSurface:
    def __init__(self, sim, face, point, liquid):
        self.sim = sim
        self.pose = getattr(sim,'contact_pose',None) or sim.pose
        self.reference = getattr(sim,'material_reference',sim.rest).copy()
        self.recipe_cursor = len(self.pose.recipes)
        q, n, skin_face, _ = self.pose.tree.find_nearest(Vector(point))
        normal = np.asarray(n, dtype=float)
        self.normal = normal
        self.across, self.up = surface_axes(normal)
        self.skin_mark = (skin_face, barycentric(self.pose.triangles[skin_face], np.asarray(q)))
        face_ids = sim.faces[face]
        self.mark_rest = barycentric(sim.pos[face_ids], np.asarray(point)) @ self.reference[face_ids]
        x, y = np.meshgrid((np.arange(SIZE) - ORIGIN[0])*PITCH,
                           (np.arange(SHAPE[0]) - ORIGIN[1])*PITCH)
        plane = np.asarray(point) + x.ravel()[:, None] * self.across + y.ravel()[:, None] * self.up
        self.sock_face, self.sock_bary, sv = project_sites(plane, tree_for(sim.pos, sim.faces), sim.pos[sim.faces], normal)
        self.skin_face, self.skin_bary, kv = project_sites(plane, self.pose.tree, self.pose.triangles, normal)
        if not sv[CENTER] or not kv[CENTER]:
            raise ValueError('泼水点缺少同侧的袜面和皮肤')
        self.skin_valid = kv
        self.sock_valid = sv
        self.sock_ids = sim.faces[self.sock_face]
        self.rest = gather(self.reference, self.sock_ids, self.sock_bary)
        self.reference_plane = np.column_stack((self.rest @ self.across, self.rest @ self.up))
        self.faces = sim.faces
        self.area = self.area_at(self.sock_points(), self.sock_valid)
        skin_area = self.area_at(self.skin_points(), kv)
        self.film = Film(SHAPE, PITCH, (sv, kv), self.area, skin_area, liquid)
        self.contact = {}
        self.inlets = []
        self.exchange = (0., 0., 0.)
        self.probe_contact = False
        self.cached_contact_tree = None
        self.refresh_sampling()

    def sock_points(self):
        return gather(self.sim.pos, self.sock_ids, self.sock_bary)

    def skin_points(self):
        return np.einsum('ni,nik->nk', self.skin_bary, self.pose.triangles[self.skin_face])

    def refresh_sampling(self):
        self.sock_uv = gather(self.sim.uv, self.sock_ids, self.sock_bary)
        u = self.sim.uv[self.sock_ids, 0]
        self.sock_uv[:, 0] = (u[:, 0] + np.sum(self.sock_bary * ((u - u[:, :1]) - np.round(u-u[:, :1])), axis=1)) % 1
        self.updates = None
        self._sock_coverage = None
        area = self.area_at(self.sim.rest, np.ones(len(self.sim.rest), bool), self.sim.faces, 1e-12)
        self.dual_area = area
        rest = self.reference - self.rest[CENTER]
        xy = np.column_stack((rest @ self.across, rest @ self.up)) / PITCH + ORIGIN
        self.shader_ids, self.shader_weights = self.stencil(xy, self.sock_valid)
        self.shader_weights[np.abs(rest @ self.normal) > .028] = 0.

    @staticmethod
    def area_at(points, valid, faces=TRIANGLES, minimum=PITCH**2/6):
        t = points[faces]
        a = .5 * np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1)
        a *= valid[faces].all(axis=1)
        return np.maximum(minimum,
                          np.bincount(faces.ravel(), np.repeat(a/3, 3), minlength=len(points)))

    def rebind(self):
        if self.faces is self.sim.faces and len(self.shader_ids)==len(self.sim.rest):
            return
        self.faces = self.sim.faces
        # The pose updates sim.rest in world space. Extend the wet material
        # reference with the same subdivisions and duplicated tear vertices.
        rest = self.sim.rest
        if hasattr(self.sim,'material_reference'):
            self.reference=self.sim.material_reference
        elif np.array_equal(rest[:len(self.reference)], self.reference):
            self.reference = rest.copy()
        else:
            self.reference = self.pose.extend_samples(self.reference,self.recipe_cursor)
        self.recipe_cursor = len(self.pose.recipes)
        tree = tree_for(self.reference, self.sim.faces)
        sites,feet=[],[]
        for i in np.flatnonzero(self.sock_valid):
            q, _, face, distance = tree.find_nearest(Vector(self.rest[i]))
            if face is None or distance > .00008:
                self.sock_valid[i] = False
                continue
            self.sock_face[i]=face
            sites.append(i)
            feet.append(tuple(q))
        if sites:
            ids=self.sim.faces[self.sock_face[sites]]
            self.sock_ids[sites]=ids
            triangles=self.reference[ids]
            a,b,c=triangles.transpose(1,0,2)
            u,v,p=b-a,c-a,np.asarray(feet)-a
            normal=np.cross(u,v)
            denom=np.einsum('ij,ij->i',normal,normal)
            valid=denom>1e-20*np.einsum('ij,ij->i',u,u)*np.einsum('ij,ij->i',v,v)
            weights=np.zeros((len(sites),3))
            for col,area in ((1,np.cross(p,v)),(2,np.cross(u,p))):
                weights[valid,col]=np.einsum('ij,ij->i',area[valid],normal[valid])/denom[valid]
            weights[:,0]=1.-weights[:,1:].sum(axis=1)
            for j in np.flatnonzero(~valid):
                weights[j]=barycentric(triangles[j],feet[j])
            self.sock_bary[sites]=weights
        self.refresh_sampling()

    @staticmethod
    def stencil(xy, valid):
        base = np.floor(xy).astype(int)
        ax, ay = np.clip(xy - base, 0., 1.).T
        x, y = base.T
        i = x + y * SIZE
        ids = np.column_stack((i, i+1, i+SIZE, i+SIZE+1)).clip(0, SHAPE[0]*SIZE - 1)
        weights = np.column_stack(((1-ax)*(1-ay), ax*(1-ay), (1-ax)*ay, ax*ay))
        weights *= ((x >= 0) & (y >= 0) & (x < SIZE-1) & (y < SHAPE[0]-1))[:, None] * valid[ids]
        return ids, weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    def layout(self):
        skin = self.skin_points()
        face, bary = self.skin_mark
        origin = bary @ self.pose.triangles[face]
        x = unit(skin[CENTER+1]-skin[CENTER-1])
        y = unit(skin[CENTER+SIZE]-skin[CENTER-SIZE])
        n = unit(np.cross(x, y))
        # Dual basis preserves the lattice coordinates when skin shears.
        local = np.linalg.pinv(np.stack((x, y), axis=1)).T
        return skin, origin, n, local

    def match(self, points, layout):
        skin, origin, n, basis = layout
        xy = (points - skin[CENTER]) @ basis / PITCH + ORIGIN
        ids, w = self.stencil(xy, self.skin_valid)
        foot = np.einsum('ni,nik->nk', w, skin[ids])
        good = w.sum(axis=1) > .5
        sites = ids[np.arange(len(ids)), np.argmax(w, axis=1)]
        return np.where(good, sites, -1), (points-foot) @ n - CLEARANCE

    def landing_stencil(self, face, point):
        """Resolve the rendered hit to the same material domain as a probe."""
        coverage = self.sock_coverage()
        ids = self.sim.faces[face]
        rest = barycentric(self.sim.pos[ids], point) @ self.reference[ids]
        sample = coverage.locate(rest)
        if sample is None:
            return np.full(4, -1), np.zeros(4)
        sites, weights = sample
        # The archived first inlet uses four sites per lane.
        return np.r_[sites, sites[-1]], np.r_[weights, 0.]

    def sock_coverage(self):
        self.rebind()
        if self._sock_coverage is None:
            points = gather(self.reference, self.sock_ids, self.sock_bary)
            self._sock_coverage = LayerCoverage(points, self.sock_valid)
        return self._sock_coverage

    def dose(self, side, point=None):
        points = self.skin_points() if side == 'skin' else self.sock_points()
        if point is None:
            point = points[CENTER]
        elif side == 'skin':
            point = np.asarray(self.pose.tree.find_nearest(Vector(point))[0])
        self.film.dose(side, np.linalg.norm(points - point, axis=1))

    def pour(self, sign, point=None, *, face=None):
        self.rebind()
        sim = self.sim
        sock = self.sock_points()
        tree = tree_for(sim.pos, sim.faces)
        a = np.linspace(-1., 1., 17)
        w = .91 * np.sqrt(1-a*a) + .09
        w[[0, -1]] *= .5
        w /= w.sum()
        delay = .016*a*a + .003*(1+a)
        width = self.film.liquid.radius
        across, up, normal_at_hit = self.across, self.up, self.normal
        # Keep the inlet strip on the original 04 rows so the 17 lanes share
        # the same surface impacts with the comparison capture.
        if point is None:
            incoming_column = sock[LOWER_ROWS*SIZE+SIZE//2::SIZE]
            targets = np.column_stack([np.interp(a*width, np.arange(-OFFSET, OFFSET+1)*PITCH,
                                                  incoming_column[:, k]) for k in range(3)])
        else:
            if face is None:
                _, _, face, _ = tree.find_nearest(Vector(point))
            ids = sim.faces[face]
            bind, posed = frames(np.stack((self.reference[ids], sim.pos[ids])))
            across, up, normal_at_hit = (posed @ bind.T @ np.stack((across, up, normal_at_hit)).T).T
            targets = point + (a*width)[:, None]*up
        # Reproject the interpolated targets to the actual curved stocking.
        normal = np.tile(normal_at_hit, (len(a), 1))
        for i, target in enumerate(targets):
            q, n, face, _ = tree.ray_cast(Vector(target + normal_at_hit*.008), Vector(-normal_at_hit), .016)
            if face is not None:
                targets[i] = q
                normal[i] = n
        launch_time = .050 - .003*a*a
        terminal = -1.40*normal + sign*.85*across
        launch = terminal - launch_time[:, None]*GRAVITY
        start = targets - launch * launch_time[:, None] - .5*launch_time[:, None]**2*GRAVITY
        p = Pour(start, launch, launch_time.copy(), delay, w, np.full((len(a), 4), -1),
                 np.tile((1., 0., 0., 0.), (len(a), 1)), np.zeros_like(start), self.film.liquid.volume)
        t = np.linspace(0., .060, 31)
        path = p.points(t[:, None])
        for i in range(len(a)):
            for j in range(len(t)-1):
                d = Vector(path[j+1, i]-path[j, i])
                q, n, face, dist = tree.ray_cast(Vector(path[j, i]), d.normalized(), d.length)
                if face is None:
                    continue
                p.flight[i] = t[j] + (t[j+1]-t[j])*dist/d.length
                if d.dot(n) >= 0:
                    break
                if point is None:
                    rel = np.asarray(q) - sock[CENTER]
                    xy = np.array([[rel @ self.across, rel @ self.up]]) / PITCH + ORIGIN
                    sites, weights = self.stencil(xy, self.sock_valid)
                    sites, weights = sites[0], weights[0]
                else:
                    sites, weights = self.landing_stencil(face, np.asarray(q))
                if weights.any():
                    p.sites[i], p.weights[i] = sites, weights
                v = p.velocity[i] + GRAVITY*p.flight[i]
                p.tangent[i] = v - (v @ n)*np.asarray(n)
                break
        self.inlets.append(dict(sign=sign, volume_ml=p.volume, lanes=[dict(
            origin_m=o.tolist(), velocity_m_s=v.tolist(), flight_s=float(t), delay_s=float(d),
            share=float(w), sites=[int(k) for k in ids], weights=ws.tolist(), tangent_m_s=tangent.tolist())
            for o, v, t, d, w, ids, ws, tangent in zip(
                p.origin, p.velocity, p.flight, p.delay, p.shares, p.sites, p.weights, p.tangent)]))
        self.film.add_pour(p)

    def skin_depth(self, point):
        if self.updates is None:
            return 0.
        skin, _, _, axes = self.updates
        rel = np.asarray(point) - skin[CENTER]
        xy = rel @ axes / PITCH + ORIGIN
        x, y = xy.astype(int)
        if not (0 <= xy[0] < SIZE-1 and 0 <= xy[1] < SHAPE[0]-1):
            return 0.
        ax, ay = xy - (x, y)
        i = x + y*SIZE
        idx = [i, i+1, i+SIZE, i+SIZE+1]
        return float(np.dot(((1-ax)*(1-ay), ax*(1-ay), (1-ax)*ay, ax*ay), self.contact_depth[idx]))

    def skin_depths(self, points):
        if self.updates is None:
            return np.zeros(len(points))
        skin, _, _, axes = self.updates
        xy = (points-skin[CENTER]) @ axes / PITCH + ORIGIN
        use = ((xy >= 0.) & (xy < (SIZE-1, SHAPE[0]-1))).all(axis=1)
        x, y = xy[use].astype(int).T
        a, b = xy[use].T - (x, y)
        i = x + y*SIZE
        indices = np.column_stack((i, i+1, i+SIZE, i+SIZE+1))
        weights = np.column_stack(((1-a)*(1-b), a*(1-b), (1-a)*b, a*b))
        depths = np.zeros(len(points))
        depths[use] = np.sum(weights*self.contact_depth[indices], axis=1)
        return depths

    def mechanics(self, vertex, depth, normal_correction, tangent, dt):
        mixed = min(1., depth / ROUGHNESS)
        mu = .28 + .8 * self.film.liquid.wet_gain * mixed
        shear_ratio = .001 * self.film.liquid.viscosity * dt * mixed / (AREAL_MASS * max(ROUGHNESS, depth))
        stick = min(.35, mu * (1-mixed) * normal_correction / max(tangent, 1e-12))
        viscous = shear_ratio / (1 + shear_ratio)
        solid_force = 1000 * AREAL_MASS * self.dual_area[min(vertex, len(self.dual_area)-1)] / dt**2
        self.contact[vertex] = (solid_force*normal_correction, solid_force*normal_correction*mixed,
                                solid_force*tangent*min(1., stick+viscous))
        return min(1., stick + viscous)

    def prepare_contact(self):
        self.rebind()
        self.contact = {}
        self.prepare_probe_depth()

    def advance(self):
        self.rebind()
        sock = self.sock_points()
        skin, origin, n, axes = self.layout()
        f = self.film
        active = self.sock_valid & ((f.outer > 0) | (f.knit > 0) | bool(f.skin.any()) | bool(f.pours))
        tree = self.pose.tree
        if self.cached_contact_tree is not tree:
            self.cached_contact_tree = tree
            self.cached_sock = np.full_like(sock, np.nan)
            self.cached_sites = np.full(len(sock), -1, dtype=int)
            self.cached_gaps = np.full(len(sock), np.inf)
        # A stationary site in the same evaluated skin has the same contact.
        # Recheck every moved site, with no positional tolerance.
        changed = active & (sock != self.cached_sock).any(axis=1)
        self.cached_sock[changed] = sock[changed]
        self.cached_sites[changed] = -1
        self.cached_gaps[changed] = np.inf
        candidates=np.flatnonzero(changed)
        indices, feet, signed = [], [], []
        if len(candidates):
            index=self.pose.wet_skin_index()
            if index is not None:
                faces,records=index.nearest(sock[candidates],check_faces=True)
                clear=faces>=0
                use=clear&(records[:,3:6]@n>=.1)
                indices.extend(candidates[use])
                feet.extend(records[use,:3])
                signed.extend(records[use,7]-CLEARANCE)
                # Ambiguous and coincident sides still use Blender's exact choice.
                candidates=candidates[~clear]
        for i in candidates:
            p = Vector(sock[i])
            q, normal, _, dist = tree.find_nearest(p)
            if q is None or np.dot(normal, n) < .1 or opposing_face(p, tree, q, normal, dist) is not None:
                continue
            indices.append(i)
            feet.append(q)
            signed.append((p-q).dot(normal)-CLEARANCE)
        if indices:
            paired, _ = self.match(np.asarray(feet), (skin, origin, n, axes))
            self.cached_sites[indices], self.cached_gaps[indices] = paired, signed
        sites = np.where(active, self.cached_sites, -1)
        gaps = np.where(active, self.cached_gaps, np.inf)
        area = self.area_at(sock, self.sock_valid)
        skin_area = self.area_at(skin, self.skin_valid)
        self.remove_failed_sites()
        a, b, c, touching = self.film.advance(sock, area, skin_area, sites, gaps)
        self.exchange = a, b, c
        self.probe_contact = bool(touching[CENTER])
        self.prepare_probe_depth()

    def remove_failed_sites(self):
        if self.sim.cracks:
            u, v = self.sock_uv.T
            row = np.floor(v).astype(int) - self.sim.yarn_first_row
            col = (u*144).astype(int) % 144
            rows = (row >= 0) & (row < len(self.sim.failed_cells))
            failed = np.flatnonzero(rows)[self.sim.failed_cells[row[rows], col[rows]]]
            if self.sock_valid[failed].any():
                self.sock_valid[failed] = False
                self._sock_coverage = None
        self.film.remove_sock_sites(self.sock_valid)

    def prepare_probe_depth(self):
        self.updates = self.layout() if self.film.skin.any() else None
        self.contact_depth = self.film.depth(self.film.skin, self.film.skin_area)
        # skin_depth interpolates with nonnegative weights summing to one.
        self.contact_ceiling = float(self.contact_depth.max())

    def fields(self):
        """Owned arrays for 05's layer maps. Invalid sites have no probe value."""
        f = self.film
        return dict(step=f.step, time_s=f.step*DT, lattice_shape=f.shape,
                    sock=dict(xyz_m=self.sock_points(), valid=self.sock_valid.copy(), uv=self.sock_uv.copy(),
                              triangles=self.sock_ids.copy(), barycentric=self.sock_bary.copy(),
                              outer_ml=f.outer.copy(), outer_mm=1000*f.depth(f.outer, f.area),
                              knit_ml=f.knit.copy(), saturation=f.knit/f.capacity),
                    skin=dict(xyz_m=self.skin_points(), valid=self.skin_valid.copy(),
                              triangles=self.pose.skin_ids[self.skin_face].copy(), barycentric=self.skin_bary.copy(),
                              film_ml=f.skin.copy(), film_mm=1000*f.depth(f.skin, f.skin_area)))

    def reading(self):
        f = self.film
        depth = 1000*f.depth(f.outer, f.area)
        wet = f.knit / np.maximum(f.capacity, 1e-12)
        a, b = EDGES.T
        crossing = ((depth[a] >= FRONT_MM) != (depth[b] >= FRONT_MM)) & self.sock_valid[a] & self.sock_valid[b]
        a, b = a[crossing], b[crossing]
        plane = self.reference_plane - self.reference_plane[CENTER]
        contour = plane[a] + ((FRONT_MM-depth[a])/(depth[b]-depth[a]))[:, None]*(plane[b]-plane[a])
        fronts = (1000*float(contour[:, 0].min()), 1000*float(contour[:, 0].max()),
                  -1000*float(contour[:, 1].min())) if len(contour) else (None, None, None)
        velocity = f.flow * (1000 / (f.routed_ml*DT)) if f.routed_ml > .00001 else None
        speed = (float(velocity @ self.across), -float(velocity @ self.up)) if velocity is not None else (None, None)
        p = self.sock_points()[CENTER]
        skin, k, _, _ = self.layout()
        skin_sample = self.pose.contact_at(p)
        uv = self.sock_uv[CENTER]
        near = np.linalg.norm(self.sim.pos - p, axis=1) < .006
        forces = np.array([value for i, value in self.contact.items() if near[i] and i not in self.pose.conflicts])
        normal, fluid, shear = forces.sum(axis=0) if len(forces) else (0., 0., 0.)
        to_knit, to_skin, squeeze = self.exchange
        thick = depth > .5
        return dict(step=f.step, time_s=f.step*DT, dosed_ml=f.dosed, **f.inventory(),
                    first_contact_step=f.first_contact, outer_peak_mm=float(depth.max()),
                    outer_over_05mm_ml=float(f.outer[thick].sum()),
                    outer_over_05mm_cm2=float(1e4*f.area[thick].sum()),
                    outer_runoff_ml=float(f.outer_runoff), downslope_runoff_ml=float(f.downslope_runoff),
                    outer_to_knit_ml=float(f.outer_to_knit),
                    wetted_knit_cm2=float(1e4*f.area[wet >= .05].sum()),
                    front_left_mm=fronts[0], front_right_mm=fronts[1], front_down_mm=fronts[2],
                    cross_velocity_mm_s=speed[0], down_velocity_mm_s=speed[1],
                    skin_to_knit_ml=to_knit, knit_to_skin_ml=to_skin, squeezed_ml=squeeze,
                    outer_probe_mm=float(depth[CENTER]), knit_probe_saturation=float(wet[CENTER]),
                    skin_marker_film_mm=float(1000*f.depth(f.skin, f.skin_area)[CENTER]),
                    contact_film_mm=1000*self.skin_depth(skin_sample['skin_xyz']),
                    probe_contact=self.probe_contact, gap_mm=skin_sample['gap_mm'],
                    normal_mn=float(normal), fluid_load_mn=float(fluid), shear_mn=float(shear),
                    material_uv=uv.tolist(), sock_xyz_m=p.tolist(), skin_xyz_m=k.tolist(),
                    sock_marker_xyz_m=self.sock_points()[CENTER+SIZE].tolist(),
                    skin_marker_xyz_m=skin[CENTER-SIZE].tolist(),
                    held=self.sim.grip is not None, fabric_failures=len(self.sim.fracture_events))
