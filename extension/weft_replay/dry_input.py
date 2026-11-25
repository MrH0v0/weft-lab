"""Integrate fixed-pose dry contact independently of Blender's viewport thread."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from threading import Lock
import time
import bpy
import numpy as np

from .fracture import linked_runs

_pool=None
# RNA objects, drawing caches and the material pixel buffers belong to the UI.
_DISPLAY={'sections','published_faces','published_triangles','bank_edge_cache','loop_pixels','loop_image',
          'run_pixels','run_image','run_stamp','released_run_atlas','published_loop_pixels','preview_count','knit_shader','knit_domain','factory_opening','ui_cracks','birth_job','predicting_birth','birth_seated','loop_transfer','run_transfer','changing_bank_rows'}
_MUTABLE=('pos','opening','seating','weight','target','yarn_ledger','failed_cells','loose_yarn')
_CRACK_MUTABLE=('cells','yarn_feed','edge_load','charged_loops')


def copy_physics(dst,src,*,take=False):
    for name,value in vars(src).items():
        if name not in _DISPLAY and not name.endswith(('_obj','_points','_radii')):
            setattr(dst,name,value)
    for name in _MUTABLE:
        value=getattr(src,name,None)
        if value is not None:setattr(dst,name,value if take else value.copy())
    current={c['id']:c for c in getattr(dst,'ui_cracks',())}
    cracks=[]
    for c in src.cracks:
        target=current.get(c['id'],{});target.update(c)
        for key in _CRACK_MUTABLE:
            if key in c:target[key]=copy(c[key])
            else:target.pop(key,None)
        cracks.append(target)
    dst.cracks=cracks
    by_id={c['id']:c for c in cracks}
    for key in ('nearby_crack','active_crack'):
        val=getattr(src,key,None)
        setattr(dst,key,None if val is None else by_id[val['id']])
    dst.run_links=linked_runs(cracks);dst.bridge_supports=src.bridge_supports.copy()
    dst.fracture_events=src.fracture_events.copy()
    if src.warning is not None:dst.warning=src.warning.copy()
    dst.patch=copy(src.patch)
    dst.patch.loops=copy(src.patch.loops)
    dst.patch.loops.ledger=dst.yarn_ledger;dst.patch.loops.failed=dst.failed_cells


def solve(sim,events,*,drain=False):
    started=time.perf_counter()
    changed=False;dirty=False;optic=False;driven=False
    revision=0
    for i,(target,dt,serial) in enumerate(events):
        revision=max(revision,serial)
        sim.input_revision=serial
        previous_holes=len(sim.cracks)
        sim.changed=False;sim.warning_dirty=False;sim.held_optics_changed=False
        if getattr(sim,'neutral_job',None) is not None:
            from .neutral_worker import collect
            collect(sim)
        if target is None:
            sim.hold(dt)
        else:
            sim.step(dt,sim.limit_target(target));sim.worker_pointer=target;driven=True
        if getattr(sim,'birth_seated',False):
            sim.birth_seated=False
        elif target is not None or sim.changed:sim.seat_held_faces()
        if not drain and not sim.cracks and sim.warning is not None:
            from .warning_birth import prepare
            prepare(sim)
        changed|=sim.changed;dirty|=sim.warning_dirty;optic|=sim.held_optics_changed
        # Present the birth of a hole by itself. Several adjacent interlocks
        # can drop within one display interval. Integrate each of their real
        # positions and ledger entries before the next mesh submission.
        if sim.changed and not drain and (len(sim.cracks)>previous_holes or time.perf_counter()-started>=1/90):
            break
    return changed,dirty,optic,driven,revision,events[i+1:]


class DryInput:
    def __init__(self,tool,now):
        global _pool
        if _pool is None:_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Weft dry hand')
        self.tool=tool
        front=tool.sim;self.back=copy(front)
        copy_physics(self.back,front)
        # The worker only uses presence and dimensions of the optical fields.
        # It can prepare array data, but never reads or writes an RNA property.
        self.back.sections=None
        if hasattr(front,'run_image'):self.back.run_image=True
        if hasattr(front,'loop_image'):self.back.loop_image=True
        self.back.ui_cracks=[]
        front.ui_cracks=front.cracks
        self.working=self.back
        self.back=copy(self.working)
        self.incoming=deque();self.receipts=deque();self.lock=Lock()
        self.busy=False;self.job=None
        self.queue=deque()
        self.last=tool.last
        self.dead=False;self.polling=False;self.drain_state=None
        self.poll=self.poll_ready
        self.last_target=None
        self.revision=self.solved_revision=self.presented=0
        self.back.input_revision=0
        self.commit_clock=0.;self.last_event_clock=0.
        self.last_optics=0.;self.pending_geometry=False;self.waiting_for_draw=False
        self.merge_interval=1/40;self.pointer_written=False;self.defer_until=0.;self.poll_deferred=False
        self.preview_base=front.grip_base().copy()

    def integrate(self):
        # Blender may be drawing the previous result. The next pointer can
        # already be integrated against the committed physics on this thread.
        while True:
            with self.lock:
                if not self.incoming:
                    self.busy=False
                    return
                events=list(self.incoming);self.incoming.clear()
                response=self.response
            sim=self.working
            sim.response=response
            while events:
                result=solve(sim,events,drain=self.dead)
                changed,dirty,optic,driven,revision,events=result
                if not (changed or dirty or optic or driven):continue
                snapshot=copy(sim);snapshot.ui_cracks=[]
                copy_physics(snapshot,sim)
                sim.run_job=None
                with self.lock:
                    # An ordinary solved pose is contained in the subsequent rupture.
                    # Never leave that pose ahead of a new, fully separated bank.
                    combining=self.receipts and not self.receipts[-1][0][0]
                    if combining:
                        prior,prev=self.receipts.pop()
                        dirty|=prior[1];optic|=prior[2];driven|=prior[3]
                        if getattr(snapshot,'run_job',None) is None:snapshot.run_job=getattr(prev,'run_job',None)
                    self.receipts.append(((changed,dirty,optic,driven,revision),snapshot))

    def submit(self,*,inline=False):
        run_here=False
        if self.queue:
            with self.lock:
                self.response=self.tool.sim.response
                for sample in self.queue:
                    if sample[0] is None and self.incoming and self.incoming[-1][0] is None:
                        self.incoming[-1]=(None,self.incoming[-1][1]+sample[1],sample[2])
                    else:self.incoming.append(sample)
                self.queue.clear()
                if not self.busy:
                    if self.job is not None and self.job.done():self.job.result()
                    self.busy=True
                    if inline:run_here=True
                    else:self.job=_pool.submit(self.integrate)
        if run_here:self.integrate()
        if self.dead:return
        if self.poll_deferred:
            if bpy.app.timers.is_registered(self.poll):bpy.app.timers.unregister(self.poll)
            self.polling=self.poll_deferred=False
        if not self.polling:
            self.polling=True
            bpy.app.timers.register(self.poll,first_interval=.001)

    def publish(self,wait=False,*,next_pointer=False):
        with self.lock:
            if not self.receipts:return False
            item=self.receipts[0]
            (changed,dirty,optic,driven,revision),self.back=item
        front=self.tool.sim
        gentle=not (self.back.cracks or self.back.warning is not None or changed or dirty)
        # A completed intact step can be submitted together with the next
        # pointer in this refresh interval. A true rupture publishes at once.
        if self.waiting_for_draw and not (changed or self.dead or next_pointer):
            self.defer_until=min(self.last_event_clock+1/30,time.perf_counter()+.004)
            if self.defer_until>time.perf_counter():return False
        if (driven and gentle and not (self.dead or next_pointer) and
                revision>=self.revision and time.perf_counter()-self.last_event_clock<self.merge_interval):
            self.defer_until=self.last_event_clock+self.merge_interval
            return False
        self.defer_until=0.
        with self.lock:
            if not self.receipts or self.receipts[0] is not item:return False
            self.receipts.popleft()
        if self.dead and self.receipts:
            # A later snapshot contains the entire physical history. Carry the
            # earlier topology and optical invalidations to the final submission.
            flags,state=self.receipts.popleft()
            self.receipts.appendleft(((changed or flags[0],dirty or flags[1],optic or flags[2],
                                       driven or flags[3],max(revision,flags[4])),state))
            return True
        parts=getattr(front,'sections',None)
        if parts is not None and getattr(self.back,'warned_lanes',None) and not next_pointer:
            parts.prepare_warning(self.back)
        if not (changed or dirty or driven) and not self.dead:
            # An idle yarn step advances no vertices. Copy just the new optical
            # sample when the display is due; release always settles the ledger.
            since=time.perf_counter()
            if optic and since-self.last_optics>=1/60 and (not gentle or since-self.last_event_clock>=self.merge_interval):
                from .live_view import update_loop_image
                front.patch.loops=copy(self.back.patch.loops)
                front.patch.loops.ledger=front.yarn_ledger
                front.patch.loops.failed=front.failed_cells
                update_loop_image(front);self.tool.region.tag_redraw()
                self.last_optics=time.perf_counter()
            return True
        if gentle and driven and revision<self.revision and not (next_pointer or self.dead):
            # The hand has already presented a newer position. Integrate this
            # sample on the private solver but leave the newer visible mesh in
            # place until its own result, or the next true fracture, is ready.
            self.solved_revision=max(self.solved_revision,revision)
            return True
        # A stationary warning advances the optical cue and the yarn supply;
        # it did not integrate any vertex. Preserve a newer hand position
        # which already reached the actual mesh while that hold was running.
        hand_positions=front.pos if not (driven or changed or self.dead) else None
        copy_physics(front,self.back,take=True);front.ui_cracks=front.cracks
        if hand_positions is not None:front.pos=hand_positions
        front.changed=changed;front.warning_dirty=dirty
        if front.cracks:
            neutral=front.open_rest
            if neutral is not None and neutral[0] is front.faces:self.preview_base=front.bary@neutral[1][front.grip]
        else:self.preview_base=front.grip_base()
        front.run_job=getattr(self.back,'run_job',None);self.back.run_job=None
        if changed or driven:
            if next_pointer and not (changed or dirty):
                self.pending_geometry=True
            else:
                if ((next_pointer or revision<self.revision) and
                        not self.dead and self.last_target is not None):
                    # Keep real recoil from the cut. Transfer only the hand
                    # travel since the actual pointer behind this older state.
                    if revision<self.revision:
                        origin=getattr(self.back,'worker_pointer',None) if changed else None
                        self.elastic_step(self.last_target,write=False,since=origin)
                    self.pointer_written=next_pointer
                self.tool.worker_seated=True
                try:self.tool.publish_positions()
                finally:self.tool.worker_seated=False
        elif dirty or optic:
            from .live_view import write_positions,update_loop_image,publish_warning_yarn
            if dirty:write_positions(self.tool.obj.data,front)
            update_loop_image(front)
            if dirty:publish_warning_yarn(self.tool.obj,front)
            self.tool.region.tag_redraw();front.warning_dirty=False
        self.commit_clock=time.perf_counter()
        self.presented=len(front.fracture_events)
        self.solved_revision=max(self.solved_revision,revision)
        return True

    def poll_ready(self):
        self.poll_deferred=False
        try:
            if self.dead:return None
            committed=self.publish()
            if not committed:
                front=self.tool.sim
                if getattr(front,'warned_lanes',None) and not self.waiting_for_draw:
                    sections=getattr(front,'sections',None)
                    if sections is not None:sections.prepare_warning(front)
            if self.job is not None and self.job.done():self.job.result()
            with self.lock:pending=self.busy or bool(self.receipts)
            if pending:
                if not committed and self.defer_until:
                    self.poll_deferred=True
                    return max(.0015,self.defer_until-time.perf_counter())
                return .0015
        finally:
            with self.lock:
                if self.dead or not (self.busy or self.receipts):self.polling=False
        return None

    def elastic_step(self,raw,*,write=True,since=None):
        # The short elastic substep moves the actual posed triangle mesh
        # and its attached fibres while the next nonlinear contact step runs.
        # Use the latest accepted positions, so a new grip carries the open rim.
        s=self.tool.sim
        if s.grip is None:return
        slip,opening=s.slip_work,s.raw_opening
        pointer=getattr(s,'pointer_travel',None)
        origin=s.bary@s.pos[s.grip] if since is None else s.limit_target(since,base=self.preview_base)
        target=s.limit_target(raw,base=self.preview_base)
        s.slip_work,s.raw_opening=slip,opening
        if pointer is not None:s.pointer_travel=pointer
        delta=target-origin
        length=np.linalg.norm(delta)
        if length<1e-7 and not self.pending_geometry:return
        # A high-velocity jump is integrated in bounded physical substeps.
        # The converged solution retains the full mouse displacement.
        delta*=min(1.,.004/max(length,1e-12))
        ids=s.drag_ids
        weights=np.maximum(s.weight[ids],.2*s.drag_taper[ids]*np.exp(-s.distance_sq[ids]/.00035))
        s.pos[ids]+=weights[:,None]*delta
        from .live_view import write_positions,publish_yarn_ends,publish_warning_yarn
        if getattr(s,'static_skin',None) is not None and np.dot(delta,s.normal)<0.:
            axes=s.contact_normals if s.contact_normals is not None else None
            if axes is not None:
                free=ids[~np.isin(ids,s.grip)]
                s.pos[free]=s.project_skin(s.pos[free],axes[free])
        s.tie_fronts(s.pos)
        if not s.cracks:s.seat_held_faces()
        if not write:return
        # Intact face contact can also lift the neighbouring cuff vertices.
        # For an existing seam the sparse move includes the tied banks.
        dirty=None
        if not self.pending_geometry and s.warning is None:
            if s.cracks and all('pitch' in c for c in s.cracks):
                # An elastic move changes the grab patch and vertices tied
                # along its seams. Include all seams, also on the second hole.
                material=getattr(self,'drag_seams',None)
                if material is None or material[0] is not s.faces or material[1] is not ids:
                    banks=[crack['rim_ids'].ravel() for crack in s.cracks]
                    ends=[joint[0].ravel() for crack in s.cracks for joint in crack['seam']]
                    material=self.drag_seams=(s.faces,ids,np.unique(np.concatenate([ids,*banks,*ends])))
                dirty=material[2]
        if self.pending_geometry:
            from .live_view import update_loop_image
            update_loop_image(s);self.pending_geometry=False
        write_positions(self.tool.obj.data,s,dirty=dirty)
        self.waiting_for_draw=True
        if s.cracks:publish_yarn_ends(self.tool.obj,s)
        if s.warning is not None:publish_warning_yarn(self.tool.obj,s)
        self.tool.region.tag_redraw()

    def tick(self,raw,now):
        dt=max(0.,now-self.last);self.last=now
        self.pointer_written=False
        if raw is not None:
            self.revision+=1
            if self.last_event_clock:
                period=now-self.last_event_clock
                self.merge_interval=min(1/30,max(1/60,.5*self.merge_interval+.65*period))
            self.last_target=raw;self.last_event_clock=now
        if raw is None and self.queue and self.queue[-1][0] is None:
            self.queue[-1]=(None,self.queue[-1][1]+dt,self.revision)
        else:self.queue.append((raw,dt,self.revision))
        front=self.tool.sim;warning=front.warning
        inline=False
        if raw is not None and not front.cracks and warning is not None and 'slack_m' in warning:
            lanes=getattr(front,'warned_lanes',None)
            displacement=raw-self.preview_base-warning['stroke']
            if lanes and displacement@warning['direction']>=warning['slack_m']:
                ready=next(iter(lanes[1].values()),None)
                if ready is not None and ready.done() and not ready.cancelled():
                    packet=ready.result()[2]
                    sections=getattr(front,'sections',None)
                    # Nonconforming seams and unstaged draw buffers can require
                    # a full rebuild. Leave those on the regular worker path.
                    inline=(packet is not None and sections is not None and sections.staged is not None
                            and sections.staged[0] is packet)
        # An idle solver and a prepared material lane can finish on this very
        # input dispatch, before Blender schedules a draw of the previous pose.
        self.submit(inline=inline)
        self.publish(next_pointer=raw is not None)
        if raw is not None and not self.pointer_written:
            self.elastic_step(raw)

    def finish(self):
        self.dead=True
        from .warning_birth import cancel
        if bpy.app.timers.is_registered(self.poll):bpy.app.timers.unregister(self.poll)
        self.polling=False
        self.submit()
        if self.job is not None:self.job.result()
        cancel(self.working)
        while self.receipts:self.publish(True)
        self.back=self.working
        front=self.tool.sim
        # Idle steps may update the yarn supply without moving vertices.
        # Saving, releasing and switching modes need the final conserved supply.
        pending_atlas=getattr(front,'run_job',None)
        copy_physics(front,self.back)
        front.run_job=getattr(self.back,'run_job',None) or pending_atlas
        front.__dict__.pop('ui_cracks',None)


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True,cancel_futures=True);_pool=None
