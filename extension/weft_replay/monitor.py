"""Read one committed live stocking state in material coordinates."""
from collections import deque
from dataclasses import dataclass
from math import fsum

import numpy as np
from mathutils import Vector

from .contact_audit import opposing_face
from .fracture import chart_delta
from .live_contact import barycentric, frames
from .wet_scene import LayerCoverage


INVENTORY = ('queued_ml', 'airborne_ml', 'outer_ml', 'knit_ml', 'skin_ml',
             'outflow_ml', 'evaporated_ml')
COURSE_EDGES = np.array([[[x, y], [x+1, y]] for y in (-1, 0, 1) for x in (-1, 0)])
SINGLE_EDGE = 3


@dataclass
class Anchor:
    ids: np.ndarray
    weights: np.ndarray
    uv: np.ndarray
    reference: np.ndarray
    witness: np.ndarray
    face: int | None

    def point(self, points):
        return self.weights @ points[self.ids]


class MaterialSurface:
    def __init__(self, sim):
        self.sim = sim
        self.reference = sim.rest.copy()
        self.faces = None

    def interpolate(self, a, b, fraction):
        pos = self.reference
        fraction = np.broadcast_to(fraction, len(a))
        self.reference = np.vstack((pos, pos[a]+(pos[b]-pos[a])*fraction[:, None]))

    def interpolate_faces(self, parents, weights):
        new=np.einsum('ni,nij->nj',weights,self.reference[parents])
        self.reference=np.vstack((self.reference,new))

    def refresh(self):
        sim = self.sim
        if self.faces is sim.faces:
            return False
        self.faces = sim.faces
        self.face_indices = {tuple(ids): i for i, ids in enumerate(self.faces)}
        self.chart = sim.uv[self.faces]
        self.triangles = self.reference[self.faces]
        basis = chart_delta(self.chart[:, 1:], self.chart[:, :1]).transpose(0, 2, 1)
        self.valid = np.abs(np.linalg.det(basis)) > 1e-8
        self.inverse = np.zeros_like(basis)
        self.inverse[self.valid] = np.linalg.inv(basis[self.valid])
        self.lengths = self.direction_lengths(self.reference)
        return True

    def direction_lengths(self, points, face=slice(None)):
        xyz = points[self.faces[face]]
        slopes = (xyz[..., 1:, :]-xyz[..., :1, :]).swapaxes(-1, -2) @ self.inverse[face]
        return np.linalg.norm(slopes, axis=-2)

    def stretch(self, points, face=slice(None)):
        length = self.lengths[face]
        return 100*(np.divide(self.direction_lengths(points, face), length,
                              out=np.full_like(length, np.nan), where=length > 1e-9)-1)

    def failed(self, uv):
        sim = self.sim
        x, y = np.moveaxis(np.floor(uv*[144, 1]).astype(int), -1, 0)
        row = y-sim.yarn_first_row
        ok = (row >= 0) & (row < len(sim.failed_cells))
        return ok & sim.failed_cells[np.clip(row, 0, len(sim.failed_cells)-1), x % 144]

    def stretch_map(self):
        uv = self.chart[:, 0]+chart_delta(self.chart, self.chart[:, :1]).mean(1)/[144, 1]
        values = self.stretch(self.sim.pos)
        values[self.failed(uv)] = np.nan
        return values

    def pick(self, face, weights):
        ids = self.faces[face]
        uv = self.chart[face, 0]+(weights @ chart_delta(self.chart[face], self.chart[face, 0]))/[144, 1]
        uv[0] %= 1.
        tri = self.reference[ids]
        return Anchor(ids.copy(), weights.copy(), uv, weights @ tri, tri.mean(0), face)

    def candidates(self, uv, near):
        delta = chart_delta(self.chart, uv)
        mask = self.valid & (delta.min(1) <= 1e-7).all(1) & (delta.max(1) >= -1e-7).all(1)
        idx = np.flatnonzero(mask)
        ab = np.einsum('nij,nj->ni', self.inverse[idx], -delta[idx, 0])
        w = np.column_stack((1-ab.sum(1), ab))
        p = np.einsum('ni,nij->nj', w, self.triangles[idx])
        return idx, w, (w.min(1) >= -1e-7) & (np.linalg.norm(p-near, axis=1) < .012)

    def gauge(self, anchor, offset):
        uv = anchor.uv + np.array(offset)/[144, 1]
        i = anchor.face
        slopes = (self.triangles[i, 1:]-self.triangles[i, 0]).T @ self.inverse[i]
        near = anchor.reference + slopes @ offset
        ids, w, inside = self.candidates(uv, near)
        if not inside.any():
            return None
        idx, weights = ids[inside], w[inside]
        error = np.linalg.norm(np.einsum('ni,nij->nj', weights, self.triangles[idx])-near, axis=1)
        k = error.argmin()
        return self.pick(idx[k], weights[k])

    def rebind(self, anchor):
        if anchor is None:
            return
        anchor.face = self.face_indices.get(tuple(anchor.ids))
        if anchor.face is not None:
            return
        ids, w, inside = self.candidates(anchor.uv, anchor.reference)
        error = np.linalg.norm(np.einsum('ni,nij->nj', w, self.triangles[ids])-anchor.reference, axis=1)
        keep = inside & (error < 2e-7)
        ids, w = ids[keep], w[keep]
        if not len(ids):
            return
        # A point on the prospective seam belongs to the side of its original
        # face interior. Compare in the reference chart, before the banks move.
        if len(ids) > 1:
            old = anchor.witness-anchor.reference
            side = (self.triangles[ids].mean(1)-anchor.reference) @ old
            same = side > 1e-12
            if same.any():
                ids, w, side = ids[same], w[same], side[same]
            posed = np.einsum('ni,nij->nj', w, self.sim.pos[self.faces[ids]])
            j = int(np.argmax(side))
            ambiguous = (abs(side-side[j]) < 1e-12) & (np.linalg.norm(posed-posed[j], axis=1) > 1e-6)
            if ambiguous.any():
                return
        else:
            j = 0
        anchor.face, anchor.ids, anchor.weights = int(ids[j]), self.faces[ids[j]].copy(), w[j].copy()


def cross_opening(sim, a, b):
    for crack in sim.cracks:
        axis = crack['load_axis']
        p, q = chart_delta(np.array([a.uv, b.uv]), crack['uv'])
        if p[axis]*q[axis] < 0:
            run = p[1-axis]-p[axis]*(q[1-axis]-p[1-axis])/(q[axis]-p[axis])
            if crack['fronts'][0] <= run <= crack['fronts'][1]:
                return True
    return False


def sample_layer(points, coordinates, valid, values):
    """Interpolate only inside the projectable 04 lattice, including its dry cells."""
    coverage = LayerCoverage(coordinates, valid)
    samples = []
    for point in points:
        sample = coverage.locate(np.asarray(point))
        if sample is None:
            samples.append(None)
        else:
            ids, weights = sample
            samples.append(weights @ values[ids])
    return samples


class Monitor:
    def __init__(self, sim, source, fixed_contact=None, frame=None):
        self.sim = sim
        self.source = source
        self.fixed_contact = fixed_contact
        self.frame = frame
        self.surface = MaterialSurface(sim)
        sim.monitor_interpolate = self.surface.interpolate
        sim.monitor_interpolate_faces = self.surface.interpolate_faces
        self.anchor = None
        self.edges = []
        self.skin_anchor = None
        self.history = deque(maxlen=3600)
        self.row = None
        self.wet = None
        self.wet_fields = None
        self._gaps = None
        self.capture(wet=None)

    def contact(self):
        pose = self.sim.pose
        return (pose.tree, pose.triangles, pose.poly) if pose is not None else self.fixed_contact

    def select(self, face, weights):
        self.surface.refresh()
        self.anchor = self.surface.pick(face, np.asarray(weights, dtype=float))
        self.edges = [[self.surface.gauge(self.anchor, offset) for offset in edge] for edge in COURSE_EDGES]
        # The single forward course edge begins exactly at the picked point.
        self.edges[SINGLE_EDGE][0] = self.anchor
        self.skin_anchor = None
        skin = self.contact()
        if skin is not None:
            p = self.anchor.point(self.sim.pos)
            hit, _, i, _ = skin[0].find_nearest(Vector(p))
            tri = skin[1][i]
            self.skin_anchor = (i, barycentric(tri, np.asarray(hit)),
                                (p-np.asarray(hit)) @ frames(tri[None])[0])
        self.history.clear()
        self.capture(self.wet)

    def gaps(self):
        if self._gaps is None and (skin := self.contact()) is not None:
            tree = skin[0]
            points = self.sim.pos
            values = np.empty(len(points))
            for i, xyz in enumerate(points):
                p = Vector(xyz)
                q, n, _, _ = tree.find_nearest(p)
                values[i] = 1000*(p-q).dot(n)
            self._gaps = values
        return self._gaps

    def whole_sock(self):
        g = self.gaps()
        if g is not None:
            return dict(min_mm=float(g.min()), inside=int((g < -.1).sum()), vertices=len(g))
        return None

    def edge_lengths(self):
        result = []
        for a, b in self.edges:
            if a is None or b is None or a.face is None or b.face is None or cross_opening(self.sim, a, b):
                result.append(None)
                continue
            rest = float(np.linalg.norm(b.reference-a.reference))
            posed = float(np.linalg.norm(a.point(self.sim.pos)-b.point(self.sim.pos)))
            result.append((rest, posed) if rest > 1e-9 else None)
        return result

    def sample(self):
        a = self.anchor
        if a is None or a.face is None:
            return {}
        sim = self.sim
        p = a.point(sim.pos)
        failed = bool(self.surface.failed(a.uv))
        stretch = self.surface.stretch(sim.pos, a.face)
        course, wale = (float(v) if not failed and np.isfinite(v) else None for v in stretch)
        lengths = self.edge_lengths()
        single = lengths[SINGLE_EDGE]
        patch = 100*(fsum(e[1] for e in lengths)/fsum(e[0] for e in lengths)-1) if all(lengths) else None
        data = dict(uv=a.uv.tolist(), face=int(a.face), vertices=a.ids.tolist(), barycentric=a.weights.tolist(),
                    reference_xyz_m=a.reference.tolist(), sock_xyz_m=p.tolist(),
                    displayed_xyz_m=a.point(sim.display_positions()).tolist(),
                    course_pct=course, wale_pct=wale, cell_failed=failed,
                    single_course_pct=100*(single[1]/single[0]-1) if single else None,
                    patch_course_pct=patch, course_edges_m=lengths)
        if (skin := self.contact()) is not None:
            if sim.pose is not None:
                contact = sim.pose.contact_at(p)
                q = np.asarray(contact['skin_xyz'])
            else:
                point = Vector(p)
                hit, n, face, d = skin[0].find_nearest(point)
                other = opposing_face(point, skin[0], hit, n, d)
                q = np.asarray(hit)
                contact = dict(gap_mm=1000*(point-hit).dot(n), normal=list(n),
                               skin_xyz=list(hit), polygon=int(skin[2][face]),
                               opposing_polygon=None if other is None else int(skin[2][other]))
            data.update(gap_mm=contact['gap_mm'], nearest_skin_xyz_m=q.tolist(), contact=contact)
            if self.skin_anchor is not None:
                face, weights, original = self.skin_anchor
                tri = skin[1][face]
                marker = weights @ tri
                rel = (p-marker) @ frames(tri[None])[0] - original
                data.update(slip_mm=float(1000*np.linalg.norm(rel[:2])), slip_components_mm=(1000*rel[:2]).tolist(),
                            skin_anchor_xyz_m=marker.tolist(), skin_barycentric=weights.tolist(),
                            skin_anchor_triangle=face)
            if self.wet_fields is not None:
                self.sample_wet(a, q, data)
        cell = np.floor(a.uv*[144, 1]).astype(int)
        x, y = cell
        row = y-sim.yarn_first_row
        data['cell'] = cell.tolist()
        if 0 <= row < len(sim.failed_cells):
            loops = sim.patch.loops if sim.patch is not None else None
            if loops is not None:
                match = np.flatnonzero((loops.cells==cell).all(axis=1))
                if len(match):
                    i = match[0]
                    data['yarn_transfer_mm'] = float(1000*loops.flow[i]) if loops.linked[i] else None
                    data['loop_chord_pct'] = (100*(loops.stretch[i]-1)).tolist()
        banks = []
        for crack in sim.cracks:
            axis = crack['load_axis']
            delta = chart_delta(a.uv, crack['uv'])
            if abs(delta[axis]) <= 2 and crack['fronts'][0]-.5 <= delta[1-axis] <= crack['fronts'][1]+.5:
                side = chart_delta(sim.uv[a.ids], crack['uv'])[:, axis].mean()
                banks.append(dict(hole_id=crack['id'], side='+' if side > 0 else '-',
                                  cut_coordinate='course' if axis==0 else 'wale'))
        data['banks'] = banks
        return data

    def sample_wet(self, anchor, skin_point, data):
        fields = self.wet_fields
        sock, skin = fields['sock'], fields['skin']
        coverage = self.wet.sock_coverage()
        sample = coverage.locate(anchor.point(self.wet.reference))
        skin_points = {'skin_mm': skin_point}
        if 'skin_anchor_xyz_m' in data:
            skin_points['skin_anchor_mm'] = data['skin_anchor_xyz_m']
        films = sample_layer(list(skin_points.values()), skin['xyz_m'], skin['valid'], skin['film_mm'])
        if sample is not None:
            ids, weights = sample
            data.update(outer_mm=float(weights @ sock['outer_mm'][ids]),
                        knit_pct=float(weights @ sock['saturation'][ids])*100.)
        for key, value in zip(skin_points, films):
            if value is not None:
                data[key] = float(value)

    def capture(self, wet, *, store=True):
        sim = self.sim
        if self.surface.refresh():
            self.surface.rebind(self.anchor)
            for a, b in self.edges:
                if a is not self.anchor:
                    self.surface.rebind(a)
                self.surface.rebind(b)
        self.wet = wet
        self.wet_fields = wet.fields() if wet is not None else None
        self._gaps = None
        pose = sim.pose
        if pose is None:
            step, t_s, frame = sim.step_number, sim.elapsed, self.frame
        else:
            step, t_s, frame = pose.step_number, pose.elapsed, pose.pose_key
        wet_ok = (self.wet_fields is not None and self.wet_fields['step'] == step
                  and abs(self.wet_fields['time_s']-t_s) < 1e-9)
        if not wet_ok:
            self.wet_fields = None
        fabric = sim.fracture_readings()
        self.row = dict(object=self.source, step=step, time_s=t_s, pose_frame=frame,
                        reference='fitted pose at monitor start', held=sim.grip is not None,
                        probe=self.sample(), holes=fabric['holes'], failed_cells=fabric['failed_cells'],
                        wet=wet.reading() if wet_ok else None)
        if store and self.anchor is not None:
            if self.history and self.history[-1]['step'] == step:
                self.history.pop()
            self.history.append(self.row)
        return self.row
