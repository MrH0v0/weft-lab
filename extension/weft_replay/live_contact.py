"""One evaluated pose for garment transport, grip and skin contact. Metres and seconds."""
from functools import partial

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .contact_audit import collision_mesh, opposing_face, sample
from .contact_index import ContactIndex
from .live_membrane import unit
from .posed_contact import CLEARANCE, world_vertices
from .skin_surface import PosedSkinBlocks, SkinBlocks, SkinSurface
from . import motion_fit


def barycentric(triangle, point):
    ab = np.linalg.lstsq((triangle[1:] - triangle[0]).T, point - triangle[0], rcond=None)[0]
    return np.r_[1 - ab.sum(), ab]


def frames(triangles):
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]
    x = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    z = np.cross(a, b)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return np.stack((x, np.cross(z, x), z), axis=-1)


def connectivity(mesh):
    return motion_fit.corners_of(mesh)


class LiveContact:
    def __init__(self, source, skin, slot, graph, source_ids, initial, faces, *, prepared_motion=True, wet_contact=False):
        if skin is None:
            raise ValueError("动作抓取需要已绑定的皮肤接触网格")
        self.source, self.skin, self.slot = source, skin, slot
        self.source_ids = source_ids
        source_mesh = source.evaluated_get(graph).data
        self.source_count = len(source_mesh.vertices)
        self.source_corners = connectivity(source_mesh)
        self.native = source.find_armature() is not None or 'weft_length_skin_vertices' in source
        self.fitted = 'weft_length_skin_vertices' in source
        self.prepared_motion = self.fitted and prepared_motion
        if not self.native and skin.find_armature() is None:
            raise ValueError("动作抓取需要袜面或接触皮肤具有骨架")
        self.initial = initial.copy()
        self.pose_key = None
        self.recipes = []
        self.step_number = 0
        self.marker = None
        self.elapsed = 0.
        self.grip_compression = 0.
        self.projected = {}
        self.conflicts = {}
        self.unresolved = {}
        self.return_patch = None
        # Lisa's trial asset has a dedicated, independently renderable right-leg skin.
        self.responds_to_pressure = source.get('weft_candidate_ankle') is not None
        original = source.get('weft_length_source', source)
        self.separate_leg_skin = original.get('weft_candidate_ankle') is not None
        self.pressure_anchor = None
        self.indent_depth = 0.
        self.skin_deformation_mm = 0.
        self.skin_deformed_vertices = 0
        self._settled_contact = None
        self.prepared=None
        self.skin_blocks = None
        self.posed_skin_blocks = None
        self.refresh_skin(graph)
        self.contact_index = (ContactIndex.create(self.base_skin_vertices, self.skin_ids)
                              if self.prepared_motion and not self.responds_to_pressure else None)
        self.wet_index = None
        self.wet_contact = wet_contact
        self.wet_contact_tree = None
        self.bind_face = np.empty(len(initial), np.int32)
        nearest = np.empty_like(initial)
        self.bind_conflict = np.zeros(len(initial), bool)
        for i, vertex in enumerate(initial):
            p = Vector(vertex)
            q, n, face, distance = self.tree.find_nearest(p)
            self.bind_face[i], nearest[i] = face, q
            if distance <= .004:
                self.bind_conflict[i] = opposing_face(p, self.tree, q, n, distance) is not None
        triangles = self.triangles[self.bind_face]
        edges = triangles[:, 1:] - triangles[:, :1]
        u, v = edges[:, 0], edges[:, 1]
        normal = np.cross(u, v)
        denom = np.maximum(np.sum(normal * normal, axis=1), 1e-30)
        rel = nearest - triangles[:, 0]
        a = np.sum(np.cross(rel, v) * normal, axis=1) / denom
        b = np.sum(np.cross(u, rel) * normal, axis=1) / denom
        self.bind_bary = np.column_stack((1 - a - b, a, b))
        basis = frames(triangles)
        raw = world_vertices(source.evaluated_get(graph))[source_ids]
        offset = initial - (raw if self.native else nearest)
        self.bind_offset = np.einsum('nki,nk->ni', basis, offset)
        self.bind_frames = basis
        self.fitted_triangle_key=motion_fit.array_key(self.skin_ids) if self.prepared_motion and not self.bind_offset.any() else None
        self.set_topology(faces, initial)

    def refresh_skin(self, graph, prepared_rest=None):
        evaluated = self.skin.evaluated_get(graph)
        if self.fitted:
            corners,_=motion_fit.skin_topology(self.skin,evaluated,graph,self.slot)
        else:
            corners=connectivity(evaluated.data)
        if not hasattr(self, 'skin_ids'):
            reference=motion_fit.reference_collision(self.source,graph) if self.prepared_motion else None
            if reference is None:
                vertices, self.skin_ids, self.poly = collision_mesh(self.skin, self.slot, graph)
            else:
                vertices=world_vertices(evaluated)
                self.skin_ids,self.poly=reference
            self.skin_corners = corners
            self.skin_count = len(vertices)
        else:
            vertices = world_vertices(evaluated)
            if len(vertices) != self.skin_count or not np.array_equal(corners,self.skin_corners):
                raise ValueError("动作抓取期间皮肤网格拓扑发生变化")
        # Blender may flip the internal diagonal of a deformed n-gon.
        # Reuse the initial triangulation so a skin landmark keeps its corners.
        self.skin_transform = evaluated.matrix_world.copy()
        if hasattr(self,'base_skin_vertices') and np.array_equal(vertices,self.base_skin_vertices):
            return
        self.base_skin_vertices = vertices
        self._settled_contact = None
        if self.fitted:
            from .sock_length import fitted_collision
            self.base_tree=fitted_collision(self.source,vertices,self.skin_ids)
            if self.prepared_motion:
                if self.skin_blocks is None:
                    self.skin_blocks = SkinBlocks(vertices, self.skin_ids)
                self.posed_skin_blocks = PosedSkinBlocks(self.skin_blocks, vertices, self.skin_ids)
            if self.base_tree is None:
                garment = prepared_rest if prepared_rest is not None else world_vertices(self.source.evaluated_get(graph))
                self.base_tree=SkinSurface(vertices,self.skin_ids,garment,
                                           blocks=self.posed_skin_blocks, search_tiles=self.prepared_motion,
                                           close_first=getattr(self,'contact_index',None) is not None)
            self.base_triangles=self.base_tree.positions
            self.unfocused_tree=self.base_tree
        else:
            self.base_triangles = vertices[self.skin_ids]
            self.base_tree = self.skin_tree(vertices)
        self.skin_vertices,self.triangles,self.tree = vertices,self.base_triangles,self.base_tree

    def skin_tree(self, vertices):
        return BVHTree.FromPolygons(vertices.tolist(), self.skin_ids.tolist(), all_triangles=True)

    def interpolate(self, a, b, fraction):
        self.recipes.append((a.copy(), b.copy(), np.asarray(fraction).reshape(-1, 1)))
        self.vertex_faces = None

    def interpolate_faces(self, parents, weights):
        self.recipes.append((parents.copy(), None, weights.copy()))
        self.vertex_faces = None

    def extend_samples(self, values, start=0):
        """Carry the same edge or triangle coordinates through every topology change."""
        for a, b, weights in self.recipes[start:]:
            if b is None:
                extra = np.einsum('ni,nij->nj', weights, values[a])
            else:
                extra = values[a] + (values[b]-values[a])*weights
            values = np.vstack((values, extra))
        return values

    def set_topology(self, faces, rest):
        # Use the largest adjacent triangle to transport local offsets with the pose.
        tri = rest[faces]
        areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
        largest_area = np.full(len(rest), -np.inf)
        for corners in faces.T:
            np.maximum.at(largest_area, corners, areas)
        # Equal areas still select the last adjacent face in input order.
        largest = np.full(len(rest), -1, np.int32)
        for corners in faces.T:
            matches = areas == largest_area[corners]
            np.maximum.at(largest, corners[matches], np.flatnonzero(matches))
        largest[largest < 0] = areas.argmin()
        self.vertex_faces = faces[largest]

    def update_pose(self, sim, graph, key, prepared_rest=None):
        if key == self.pose_key:
            return False
        self.refresh_skin(graph, prepared_rest)
        canonical = self.triangles[self.bind_face]
        guide = np.einsum('ni,nik->nk', self.bind_bary, canonical)
        self.bind_frames=frames(self.base_triangles[self.bind_face])
        offset = np.einsum('nki,ni->nk', self.bind_frames, self.bind_offset)
        if self.native:
            evaluated = self.source.evaluated_get(graph)
            raw = world_vertices(evaluated) if prepared_rest is None else prepared_rest
            if len(raw) != self.source_count or not np.array_equal(connectivity(evaluated.data),self.source_corners):
                raise ValueError("动作抓取期间袜面网格拓扑发生变化")
            next_rest = raw[self.source_ids] + offset
            if self.fitted_triangle_key is not None:
                self.prepared=motion_fit.samples_for(self.source,key,self.fitted_triangle_key,motion_fit.coordinate_key(raw))
        else:
            next_rest = guide + offset
        next_rest = self.extend_samples(next_rest)
        self.guide = guide
        self.pose_key = key
        history=getattr(sim,'neutral_history',None) if self.wet_contact and sim.cracks else None
        previous=history[1:] if history is not None and history[0] is sim.faces else ()
        ids,rotation = self.transport(sim,next_rest,previous)
        if self.wet_contact and sim.cracks:
            # The paid-out bank is a material shape. Carry its accepted
            # corrections with the same local cloth frames as the live vertices,
            # then recheck against this pose's curved skin and other bank.
            if previous:
                def carry(sheet):
                    moved=next_rest.copy()
                    moved[ids]+=np.einsum('nij,nj->ni',rotation,sheet[ids]-sim.rest[ids])
                    return moved
                sim.neutral_history=(sim.faces,carry(previous[0]),carry(previous[1]))
            else:sim.neutral_history=None
            sim.open_rest=sim.contact_normals=sim.tear_support_faces=sim.strain_cache=None
            sim.material_frame=sim.held_contact=sim.contact_band=sim.contact_material_band=None
            sim.bridge_supports.clear()
        sim.rest[:] = next_rest
        if self.fitted:
            self.base_tree=self.unfocused_tree
            if self.prepared is not None and sim.patch is not None and len(sim.patch.ids):
                self.base_tree=SkinSurface(self.base_skin_vertices,self.skin_ids,next_rest[sim.patch.ids],
                                           padding=.016,fallback=self.unfocused_tree,blocks=self.posed_skin_blocks)
            self.tree=self.base_tree
        if sim.grip is not None:
            sim.anchor = sim.bary @ sim.rest[sim.grip]
            tri = sim.rest[sim.grip]
            sim.normal = unit(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
            sim.weights()
            sim.update_axes()
        if sim.patch is not None:
            patch = sim.patch
            patch.normal = np.einsum('nij,nj->ni', rotation[np.searchsorted(ids,patch.ids)], patch.normal)
            patch.loops.measure(sim.rest[patch.ids])
            patch.rest_centers = patch.loops.centers.copy()
            patch.directions = patch.loops.vectors.copy()
            if sim.grip is not None:
                patch.offset = sim.rest[patch.ids] - sim.anchor
                patch.distance = np.linalg.norm(patch.offset, axis=1)
            self.set_patch_planes(patch, sim.rest[patch.ids])
            if patch is self.return_patch:
                self.refresh_seating(sim)
        for crack in sim.cracks:
            crack_rotation=rotation[np.searchsorted(ids,crack['ids'])]
            for name in ('direction','tip_direction','cap_tangents','course_tangents','bias_tangents'):
                if name in crack:
                    crack[name] = np.einsum('nij,nj->ni', crack_rotation, crack[name])
            crack['offset'] = np.einsum('nij,nj->ni', crack_rotation, crack['offset'])
            transform=rotation[np.searchsorted(ids,crack['banks'][0,0])]
            crack['along'] = transform @ crack['along']
            crack['across'] = transform @ crack['across']
            crack['normal'] = transform @ crack['normal']
            crack['center'] = sim.rest[crack['banks']].mean(axis=(0,1))
        return True

    def transport(self, sim, next_rest, accepted=()):
        # A neutral vertex has no material offset to rotate. Still transport
        # reference directions at a grab patch or an existing tear.
        active=np.any(sim.pos!=sim.rest,axis=1)|np.any(sim.opening!=0.,axis=1)|np.any(sim.seating!=0.,axis=1)
        for sheet in accepted:active|=np.any(sheet!=sim.rest,axis=1)
        if sim.patch is not None:
            active[sim.patch.ids]=True
        for crack in sim.cracks:
            active[crack['ids']]=True
            active[crack['banks'][0,0]]=True
        ids=np.flatnonzero(active)
        if self.vertex_faces is None and len(ids):
            self.set_topology(sim.faces,sim.rest)
        rotation=np.empty((0,3,3))
        if len(ids):
            faces=self.vertex_faces[ids]
            old_frame=frames(sim.rest[faces])
            new_frame=frames(next_rest[faces])
            rotation=new_frame@old_frame.transpose(0,2,1)
            degenerate=((np.linalg.norm(old_frame[:,:,2],axis=1)<.5)
                        | (np.linalg.norm(new_frame[:,:,2],axis=1)<.5))
            rotation[degenerate]=np.eye(3)
            sim.pos[ids]=next_rest[ids]+np.einsum('nij,nj->ni',rotation,sim.pos[ids]-sim.rest[ids])
            sim.opening[ids]=np.einsum('nij,nj->ni',rotation,sim.opening[ids])
            sim.seating[ids]=np.einsum('nij,nj->ni',rotation,sim.seating[ids])
        sim.pos[~active]=next_rest[~active]
        return ids,rotation

    def set_patch_planes(self, patch, points):
        if self.prepared_motion and patch is not self.return_patch:
            # A moving grip uses live collision in step(). Seating uses the
            # planes on release, with the same pose and surface sampled before.
            samples = None if self.indent_depth else self.prepared
            patch.defer_skin_plane(partial(self._sample_skin_planes, points, patch.normal,
                                           patch.ids, self.tree, samples))
        else:
            patch.skin_plane = self.skin_planes(points, patch.normal, patch.ids)

    def skin_planes(self, points, normals, indices=None):
        samples = None if self.indent_depth else self.prepared
        return self._sample_skin_planes(points, normals, indices, self.tree, samples)

    def _sample_skin_planes(self, points, normals, indices, tree, prepared):
        origin = points.copy()
        normal = np.zeros_like(points)
        pending=range(len(points))
        if prepared is not None and indices is not None:
            original=indices<len(self.initial)
            samples=prepared.planes[self.source_ids[indices[original]]]
            valid=np.sum(samples[:,:3]*normals[original],axis=1)>=.65
            available=np.flatnonzero(original)[valid]
            normal[available]=samples[valid,:3]
            origin[available]-=np.maximum(0.,samples[valid,3].astype(float)-CLEARANCE)[:,None]*normal[available]
            pending=np.flatnonzero(~original)
        for i in pending:
            row=points[i]
            p = Vector(row)
            q,n,_,d = tree.find_nearest(p, .004)
            if q is not None and np.dot(n, normals[i]) >= .65 and opposing_face(p,tree,q,n,d) is None:
                normal[i] = n
                origin[i] = row - max(0.,(p-q).dot(n)-CLEARANCE)*normal[i]
        return origin, normal

    def pick(self, sim, face_index):
        self.return_patch = None
        p = sim.bary @ sim.pos[sim.grip]
        hit, _, face, _ = self.tree.find_nearest(Vector(p))
        self.marker = (sim.grip.copy(), sim.bary.copy(), face,
                       barycentric(self.triangles[face], np.asarray(hit)))
        self.pick_face = face_index
        self.pick_relation = p - np.asarray(hit)
        self.pick_basis = frames(self.base_triangles[face:face+1])[0]
        self.pick_gap = float(self.pick_relation @ self.pick_basis[:,2])
        self.marker_edges = np.linalg.norm(np.roll(sim.rest[sim.grip], -1, axis=0)-sim.rest[sim.grip],axis=1)

    def seat_released_patch(self, sim):
        self.return_patch = sim.patch
        ids = sim.patch.ids
        moved = np.linalg.norm(sim.pos[ids]-sim.rest[ids]-sim.opening[ids]-sim.seating[ids],axis=1)
        # Use the area actually disturbed by the drag. A radial grip mask can
        # leave the outer compression ridges floating at their import offset.
        weight = np.clip((moved-.00005)/.00015,0.,1.)
        self.return_weight = weight**2*(3-2*weight)
        self.refresh_seating(sim)

    def refresh_seating(self, sim):
        ids = self.return_patch.ids
        origin,normal = self.return_patch.skin_plane
        valid = (self.return_weight>0.) & (np.linalg.norm(normal,axis=1)>.5)
        desired = (origin-sim.rest[ids])*self.return_weight[:,None]
        sim.moving |= bool(np.any(abs(sim.seating[ids[valid]]-desired[valid])>1e-6))
        # Leave ambiguous skin samples at the pose-transported position.
        sim.seating[ids[valid]] = desired[valid]

    def deform_skin(self, dt, load):
        if not self.responds_to_pressure:
            return
        desired = 0.
        if load is not None:
            face,hit = load
            self.pressure_anchor = face,barycentric(self.base_triangles[face],np.asarray(hit))
            desired = min(.00035,.25*self.grip_compression)
        self.indent_depth += (desired-self.indent_depth)*(-np.expm1(-dt/.04))
        if self.indent_depth < .000001:
            self.indent_depth = 0.
            self.skin_vertices,self.triangles,self.tree = self.base_skin_vertices,self.base_triangles,self.base_tree
            self.skin_deformation_mm = 0.
            self.skin_deformed_vertices = 0
            return
        face,weights = self.pressure_anchor
        center = weights@self.base_triangles[face]
        normal = frames(self.base_triangles[face:face+1])[0,:,2]
        delta = self.base_skin_vertices-center
        radius_squared = .008**2
        influence = np.maximum(0.,1.-np.einsum('ni,ni->n',delta,delta)/radius_squared)**2
        offsets = self.indent_depth*influence
        self.skin_vertices = self.base_skin_vertices-offsets[:,None]*normal
        self.triangles = self.skin_vertices[self.skin_ids]
        self.tree = self.skin_tree(self.skin_vertices)
        self.skin_deformation_mm = 1000*float(offsets.max())
        self.skin_deformed_vertices = int(np.count_nonzero(offsets))

    def step(self, sim, dt, target=None, elapsed=None, *, driven_input=None, wet=None, fabric_dt=None):
        self.step_number += 1
        self.elapsed = self.elapsed + dt if elapsed is None else elapsed
        self.grip_compression = 0.
        load = None
        index = self.contact_index if wet is None and not self.wet_contact else None
        if index is not None:
            index.update(self.base_skin_vertices)
        if sim.grip is not None and target is not None:
            p = Vector(target)
            fast_hit = None
            clear = False
            if index is not None:
                faces, data = index.nearest(target, check_faces=True, clear_at=CLEARANCE)
                clear = faces[0]==-2
                if faces[0]>=0:
                    row = data[0]
                    fast_hit = (Vector(row[:3]), Vector(row[3:6]), int(faces[0]), float(row[6]))
            if not clear:
                base_hit = (fast_hit if fast_hit is not None else
                            self.nearest_contact(self.base_tree,p,CLEARANCE if wet is None else None))
                hit,n,face,d = base_hit
                if (p-hit).dot(n)<CLEARANCE and (fast_hit is not None or opposing_face(p,self.base_tree,hit,n,d) is None):
                    self.grip_compression = CLEARANCE-(p-hit).dot(n)
                    load = face,hit
        self.deform_skin(dt,load)
        if sim.grip is not None and target is not None and not clear:
            hit,n,_,d = (base_hit if self.tree is self.base_tree
                         else self.nearest_contact(self.tree,p,CLEARANCE if wet is None else None))
            floor = CLEARANCE + (wet.skin_depth(hit) if wet is not None else 0.)
            if (p-hit).dot(n)<floor and (fast_hit is not None or opposing_face(p,self.tree,hit,n,d) is None):
                target = np.asarray(hit+n*floor)
        previous = sim.pos.copy()
        recipe_count=len(self.recipes)
        sim.step(dt if fabric_dt is None else fabric_dt,target,driven_input=driven_input)
        previous=self.extend_samples(previous,recipe_count)
        same_dry_rest = self.fitted and wet is None and sim.grip is None and not sim.moving
        if same_dry_rest and self._settled_contact is not None and np.array_equal(self._settled_contact,sim.pos):
            return False
        self._settled_contact = None
        self.projected = {}
        self.conflicts = {}
        self.unresolved = {}
        count = len(self.initial)
        plane_normal = self.bind_frames[:, :, 2]
        relative = sim.pos[:count] - self.guide
        signed = np.sum(relative * plane_normal, axis=1)
        # Separately skinned layers can drift across a fold while a stocking
        # transported by the skin itself follows its binding triangle.
        search_band = .002 if self.native else 0.
        near = (signed<CLEARANCE+search_band)&(np.linalg.norm(relative,axis=1)<.015)
        samples=self.prepared if wet is None and not self.indent_depth else None
        travel=None
        if samples is not None:
            travel=np.linalg.norm(sim.pos[:count]-sim.rest[:count],axis=1)
            near &= travel>=samples.margins[self.source_ids]
        candidates = np.flatnonzero(near | self.bind_conflict)
        moved = None
        if len(sim.pos)>count and sim.cracks:
            # Refined stitches outside the grip patch have no original bind plane.
            # Their permanent opening still needs contact after recoil has stopped.
            travel_added=sim.pos[count:]-sim.rest[count:]
            moved_added=np.einsum('ij,ij->i',travel_added,travel_added)>.000025**2
            candidates=np.union1d(candidates,count+np.flatnonzero(moved_added))
        if sim.patch is not None:
            candidates = np.union1d(candidates, sim.patch.ids)
            if self.separate_leg_skin and sim.dry and wet is None and (sim.grip is not None or sim.moving):
                # On a separately rendered, deformable skin the visible folds
                # can extend beyond the loop-gauge patch. A shoulder may enter
                # another skin triangle while clearing the original bind plane.
                displacement = sim.pos - sim.rest - sim.seating - sim.opening
                moved = np.einsum('ij,ij->i', displacement, displacement) > (.000025**2)
                candidates = np.union1d(candidates, np.flatnonzero(moved))
        if sim.cracks and not self.wet_contact and wet is None:
            # A permanent opening can carry an original vertex beyond the
            # 15 mm bind-plane search, including outside the current finger
            # patch. Query its actual posed skin even after the hand lets go.
            offset=sim.pos[:count]-sim.rest[:count]
            paid=np.flatnonzero(np.einsum('ij,ij->i',offset,offset)>.000025**2)
            candidates=np.union1d(candidates,paid)
        grabbed = set(sim.grip) if sim.grip is not None else set()
        if samples is not None:
            original=candidates[candidates<count]
            same=(travel[original]==0.)
            stationary=original[same]
            rest=samples.projected[self.source_ids[stationary]]
            reusable=rest[:,0]==0. if grabbed else rest[:,0]>=0.
            for i,d in zip(stationary[reusable],rest[reusable]):
                if d[0]>0.:
                    sim.pos[i]=d[1:]
                    self.projected[int(i)]=d[0]
            candidates=np.setdiff1d(candidates,stationary[reusable],assume_unique=True)
        # Collision queries use the same native nearest faces. Collect the
        # independent dry corrections before applying their tangential drag.
        dry_corrections = []
        result = None
        if index is not None and len(candidates):
            result = index.nearest(sim.pos[candidates], check_faces=True, clear_at=CLEARANCE)
        if self.wet_contact and len(candidates):
            candidates = self.wet_candidates(sim.pos, candidates, wet)
        if result is not None:
            winner, values = result
            keep = winner != -2
            candidates, winner, values = candidates[keep], winner[keep], values[keep]
        for row_id, i in enumerate(candidates):
            p = Vector(sim.pos[i])
            regular = i >= count or not self.bind_conflict[i]
            clear_at = .00035 if wet is None and regular else None
            reliable = result is not None and winner[row_id]>=0
            if self.wet_contact:
                q,n=Vector(self.wet_feet[i]),Vector(self.wet_normals[i])
                face,d=self.wet_faces[i],self.wet_distance[i]
            elif reliable:
                row = values[row_id]
                q,n,face,d = Vector(row[:3]), Vector(row[3:6]), int(winner[row_id]), float(row[6])
            else:
                q,n,face,d = self.nearest_contact(self.tree,p,clear_at)
            gap = (p-q).dot(n)
            film = wet.skin_depth(q) if wet is not None else 0.
            floor = CLEARANCE + film
            if gap >= max(.00035, floor) and regular:
                continue
            other = ((self.wet_opposite[i] if self.wet_opposite[i]>=0 else None) if self.wet_contact else
                     None if reliable else opposing_face(p,self.tree,q,n,d))
            if other is not None:
                self.conflicts[int(i)] = [int(self.poly[face]), int(self.poly[other])]
                continue
            if gap < floor:
                depth = floor-gap
                if wet is None:
                    dry_corrections.append((int(i), depth, np.asarray(n), p, gap))
                    continue
                candidate = sim.pos[i]+depth*np.asarray(n)
                if sim.grip is not None and i not in grabbed:
                    tangent = candidate-previous[i]
                    tangent -= np.dot(tangent,n)*np.asarray(n)
                    slip = np.linalg.norm(tangent)
                    drag = wet.mechanics(int(i), film, depth, slip, dt)
                    candidate -= drag*tangent
                try:
                    sim.pos[i]=motion_fit.checked_projection(p,candidate,self.tree,self.triangles)
                except RuntimeError:
                    self.unresolved[int(i)] = 1000*gap
                    continue
                self.projected[int(i)] = depth
        if dry_corrections:
            self.apply_dry_contact(sim, previous, grabbed, dry_corrections, index)
        changed_interior = moved is not None and self.constrain_dragged_triangles(sim, moved)
        if sim.patch is not None and (self.projected or changed_interior):
            sim.tie_fronts(sim.pos)
            sim.patch.update_stitches(sim.pos)
        if same_dry_rest and not self.projected:
            self._settled_contact=sim.pos.copy()
        return not self.wet_contact or not np.array_equal(previous, sim.pos)

    def wet_skin_index(self):
        if self.wet_index is None:
            self.wet_index=ContactIndex.create(self.skin_vertices,self.skin_ids)
        elif self.wet_index.geometry is not self.skin_vertices:
            self.wet_index.update(self.skin_vertices)
        return self.wet_index

    def wet_rays(self,origins,directions,limit):
        index=self.wet_skin_index()
        if index is not None:
            return index.first_hits(origins,directions,limit)
        return np.array([self.tree.ray_cast(Vector(p),Vector(d),limit)[3]
                         for p,d in zip(origins,directions)],dtype=float)

    def seat_wet_grip(self,sim):
        torn=bool(sim.cracks) and self.wet_contact
        if sim.grip is None and not torn:return
        from .knit_patch import surface_normals
        from .seam_contact import seat_held_faces
        if (getattr(self,'contact_face_layout',None) is not sim.faces
                or getattr(self,'contact_axis_pose',None)!=self.pose_key):
            self.contact_face_layout=sim.faces
            self.contact_axis_pose=self.pose_key
            self.contact_axes=surface_normals(sim.rest,sim.faces)
        index=self.wet_skin_index()
        def clearance(points,normals):
            hits=self.wet_rays(points+.014*normals,-normals,.025)
            return np.where(np.isfinite(hits),hits-.014,.025)
        banks=(np.concatenate([sim.closed_banks(c) for c in sim.cracks]) if sim.cracks else np.empty((0,2),dtype='i'))
        if sim.grip is not None:
            seat_held_faces(sim,self.contact_axes,clearance,sim.faces,banks,
                            face_needs=index.wet_face_lifts if index is not None else None)
        if torn:
            # Adhesion acts on both sides. Recheck the real fabric after that
            # displacement, including the transient frames following release.
            from . import seam_overlap
            neutral=sim.relaxed_tears()
            ids,tri=sim.tear_contact_band()
            posed=sim.pos[ids]
            if seam_overlap.uncross(neutral[ids],posed,tri,released_from=(sim.rest[ids],neutral[ids])):
                sim.pos[ids]=posed
            sim.pos[banks[:,1]]=sim.pos[banks[:,0]]

    def seat_wet_seam(self,sheet,previous,wet,dt):
        """Apply the current film thickness and its tangential drag to a fixed, opened sheet."""
        if sheet.grip is None or sheet.affected is None:return
        ids=self.wet_candidates(sheet.pos,sheet.affected,wet)
        grabbed=set(sheet.grip)
        for i in ids:
            if self.wet_opposite[i]>=0:continue
            q=self.wet_feet[i];n=self.wet_normals[i];p=sheet.pos[i]
            film=wet.skin_depth(q)
            depth=CLEARANCE+film-float((p-q)@n)
            if depth<=0.:continue
            candidate=p+depth*n
            if i<len(previous) and i not in grabbed:
                tangent=candidate-previous[i];tangent-=float(tangent@n)*n
                candidate-=wet.mechanics(int(i),film,depth,np.linalg.norm(tangent),dt)*tangent
            try:sheet.pos[i]=motion_fit.checked_projection(Vector(p),candidate,self.tree,self.triangles)
            except RuntimeError:self.unresolved[int(i)]=1000*(CLEARANCE+film-depth)

    def wet_candidates(self, positions, candidates, wet):
        # Contact forces depend on current water depth, but nearest faces and
        # their opposing-face checks depend only on the exact two geometries.
        old_count = len(self.wet_cached_positions) if self.wet_contact_tree is self.tree else 0
        if not old_count or len(positions) != old_count:
            # Subdivision appends vertices. A skin query at an unchanged world
            # position is still valid even when that garment vertex changes faces.
            keep=min(old_count,len(positions))
            for attr,shape,dtype in (('wet_cached_positions',positions.shape,float),
                                     ('wet_feet',positions.shape,float),('wet_normals',positions.shape,float),
                                     ('wet_gaps',(len(positions),),float),('wet_distance',(len(positions),),float),
                                     ('wet_faces',(len(positions),),np.int32),('wet_opposite',(len(positions),),np.int32)):
                buffer=np.empty(shape,dtype=dtype)
                if keep:buffer[:keep]=getattr(self,attr)[:keep]
                setattr(self,attr,buffer)
            self.wet_cached_positions[keep:]=np.nan
            self.wet_contact_tree=self.tree
        changed = candidates[(positions[candidates] != self.wet_cached_positions[candidates]).any(axis=1)]
        self.wet_cached_positions[changed] = positions[changed]
        if len(changed):
            index=self.wet_skin_index()
            if index is not None:
                faces,records=index.nearest(positions[changed],check_faces=True)
                unique=faces>=0
                ids=changed[unique]
                self.wet_feet[ids]=records[unique,:3]
                self.wet_normals[ids]=records[unique,3:6]
                self.wet_distance[ids]=records[unique,6]
                self.wet_gaps[ids]=records[unique,7]
                self.wet_faces[ids]=faces[unique]
                self.wet_opposite[ids]=-1
                changed=changed[~unique]
        for i in changed:
            p = Vector(positions[i])
            q, n, face, distance = self.tree.find_nearest(p)
            other = opposing_face(p, self.tree, q, n, distance)
            self.wet_opposite[i] = -1 if other is None else other
            self.wet_feet[i],self.wet_normals[i]=q,n
            self.wet_faces[i],self.wet_distance[i]=face,distance
            self.wet_gaps[i] = (p-q).dot(n)
        gaps = self.wet_gaps[candidates]
        possible = gaps < CLEARANCE
        if wet is not None and wet.contact_ceiling > 0.:
            deeper = (gaps >= CLEARANCE) & (gaps < CLEARANCE + wet.contact_ceiling + 1e-9)
            floor = CLEARANCE + wet.skin_depths(self.wet_feet[candidates[deeper]])
            # Leave near-equality to the original scalar contact decision.
            possible[deeper] = gaps[deeper] <= floor + 1e-9
        return candidates[possible | (self.wet_opposite[candidates]>=0)]

    def constrain_dragged_triangles(self, sim, moved):
        # A convex skin can cross a cloth triangle even after all three
        # corners have cleared. Reuse the fitted garment's face contact pass
        # only on the part moved by the hand or still returning after a pull.
        from .sock_length import _keep_faces_outside
        faces = sim.faces[moved[sim.faces].any(axis=1)]
        if not len(faces):
            return False
        ids, local_faces = np.unique(faces, return_inverse=True)
        original = sim.pos[ids]
        corrected = original.copy()
        _keep_faces_outside(corrected, local_faces.reshape(-1, 3), self.tree, self.triangles, sample_gap=0.)
        sim.pos[ids] = corrected
        return not np.array_equal(corrected, original)

    def constrain_display_rim(self, solved, displayed, ids):
        """Keep the solved vertex where display smoothing would cross the skin."""
        rejected = self.display_penetrations(displayed[ids], self.tree, 0.)
        displayed[ids[rejected]] = solved[ids[rejected]]

    def display_penetrations(self, points, tree, minimum_gap):
        rejected = np.zeros(len(points), bool)
        candidates = np.arange(len(points))
        index = self.contact_index
        if index is not None and tree is self.base_tree:
            index.update(self.base_skin_vertices)
            faces, _ = index.nearest(points, clear_at=CLEARANCE)
            candidates = candidates[faces != -2]
        for i in candidates:
            point = Vector(points[i])
            hit, normal, _, _ = self.nearest_contact(tree, point, CLEARANCE)
            rejected[i] = (point-hit).dot(normal) < minimum_gap
        return rejected

    def constrain_frozen_rim(self, solved, displayed, faces):
        """Reject extra smoothing on a frozen face whose interior crosses the skin."""
        probes = np.array([[1/3, 1/3, 1/3], [.5, .5, 0.], [0., .5, .5], [.5, 0., .5]])
        while len(faces):
            changed = np.any(displayed != solved, axis=1)
            faces = faces[changed[faces].any(axis=1)]
            if not len(faces):
                break
            samples = (probes @ displayed[faces]).reshape(-1, 3)
            crossed = self.display_penetrations(samples, self.base_tree, -CLEARANCE)
            affected = faces[crossed.reshape(-1, len(probes)).any(axis=1)]
            if not len(affected):
                break
            displayed[affected] = solved[affected]

    def apply_dry_contact(self, sim, previous, grabbed, corrections, index):
        ids, depths, axes, points, gaps = zip(*corrections)
        normals = np.stack(axes)
        depth = np.asarray(depths)
        ids = np.asarray(ids)
        proposed = sim.pos[ids] + depth.astype(normals.dtype)[:, None]*normals
        if grabbed:
            slip = ~np.isin(ids, tuple(grabbed))
            tangent = proposed[slip] - previous[ids[slip]]
            axes = normals[slip]
            tangent -= np.einsum('ij,ij->i', tangent, axes)[:, None]*axes
            travel = np.maximum(np.linalg.norm(tangent, axis=1), 1e-12)
            drag = np.minimum(.35, .28*depth[slip]/travel)
            proposed[slip] -= drag[:, None]*tangent
        certified = np.zeros(len(ids), bool)
        if index is not None:
            faces, _ = index.nearest(proposed, clear_at=CLEARANCE*.45)
            certified = faces==-2
            sim.pos[ids[certified]] = proposed[certified]
        applied = certified.copy()
        for row in np.flatnonzero(~certified):
            i, point, gap, probe = ids[row], points[row], gaps[row], proposed[row]
            try:
                sim.pos[i] = motion_fit.checked_projection(point, probe, self.tree, self.triangles)
            except RuntimeError:
                self.unresolved[int(i)] = 1000*gap
                continue
            applied[row] = True
        self.projected.update(zip(ids[applied].tolist(), depth[applied].tolist()))

    @staticmethod
    def nearest_contact(tree, point, clear_at):
        if isinstance(tree, SkinSurface):
            return tree.find_nearest(point,clear_at=clear_at)
        return tree.find_nearest(point)

    def readings(self, sim, full=False):
        frame, subframe = self.pose_key
        data = dict(object=self.source.name, skin=self.skin.name, frame=frame, subframe=subframe,
                    step=self.step_number, elapsed_s=self.elapsed, contact_conflicts=self.conflicts,
                    unresolved_contact_mm=self.unresolved,
                    projected_vertices=len(self.projected),
                    max_skin_deformation_mm=self.skin_deformation_mm,
                    deformed_skin_vertices=self.skin_deformed_vertices,
                    estimated_max_skin_indent_mm=min(.35,250*max(self.grip_compression,
                                                                 max(self.projected.values(),default=0.))))
        if full:
            data['whole_sock'] = sample(sim.pos,self.tree,self.poly,True,details=True)
        if self.marker is not None:
            ids, weights, face, skin_weights = self.marker
            sock = weights @ sim.pos[ids]
            skin = skin_weights @ self.triangles[face]
            base_skin = skin_weights @ self.base_triangles[face]
            basis = frames(self.base_triangles[face:face+1])[0]
            rel = (sock-skin) @ basis - self.pick_relation @ self.pick_basis
            uv = sim.uv[ids].copy()
            uv[:,0] -= np.round(uv[:,0]-uv[0,0])
            length = np.linalg.norm(np.roll(sim.pos[ids],-1,axis=0)-sim.pos[ids],axis=1)
            posed_length = np.linalg.norm(np.roll(sim.rest[ids],-1,axis=0)-sim.rest[ids],axis=1)
            compression = sum(w*self.projected.get(int(i),0.) for i,w in zip(ids,weights))
            data['marker'] = dict(pick_face=self.pick_face, vertices=ids.tolist(), barycentric=weights.tolist(), material_uv=(weights@uv).tolist(),
                                  sock_xyz=sock.tolist(), skin_xyz=skin.tolist(), skin_polygon=int(self.poly[face]),
                                  skin_triangle=self.skin_ids[face].tolist(), skin_barycentric=skin_weights.tolist(),
                                  tangential_mm=(1000*rel[:2]).tolist(), marker_gap_mm=1000*float(rel[2]+self.pick_gap),
                                  edge_ratio=(length/self.marker_edges).tolist(),
                                  pose_relative_edge_ratio=(length/np.maximum(posed_length,1e-12)).tolist(),
                                  held=sim.grip is not None,
                                  nearest_skin_gap_mm=self.contact_at(sock)['gap_mm'],
                                  grip_error_mm=1000*float(np.linalg.norm(sock-sim.target)) if sim.grip is not None else None,
                                  estimated_skin_indent_mm=min(.35,250*max(self.grip_compression,float(compression))),
                                  measured_skin_marker_deformation_mm=1000*float(np.linalg.norm(skin-base_skin)))
        return data

    def contact_at(self, world_xyz):
        point = Vector(world_xyz)
        hit,normal,triangle,distance = self.tree.find_nearest(point)
        other = opposing_face(point,self.tree,hit,normal,distance)
        return dict(step=self.step_number, frame=self.pose_key[0], subframe=self.pose_key[1],
                    skin=self.skin.name, skin_xyz=list(hit), normal=list(normal),
                    gap_mm=1000*(point-hit).dot(normal), polygon=int(self.poly[triangle]),
                    opposing_polygon=None if other is None else int(self.poly[other]))
