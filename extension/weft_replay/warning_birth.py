"""Prepare a stationary hand's first contact solution up to the stitch's due time."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import numpy as np

_pool=None


def calculate(trial,remaining):
    trial.slide_held_yarn(remaining)
    ledger=trial.yarn_ledger.copy()
    trial.warning['age']=trial.warning['duration']
    trial.advance_fracture(trial.target-trial.grip_base(),False,0.,0.)
    if len(trial.cracks)!=1:return None
    trial.seat_held_faces()
    return ledger,trial


def cancel(sim):
    pending=getattr(sim,'birth_job',None)
    if pending is not None:
        pending[-1].cancel()
        sim.birth_job=None


def prepare(sim):
    global _pool
    warning=sim.warning
    remaining=warning['duration']-warning['age']
    lanes=getattr(sim,'warned_lanes',None)
    if not (0.<remaining<=.10 and lanes and all(j.done() and not j.cancelled() for j in lanes[1].values())):return
    pending=getattr(sim,'birth_job',None)
    if pending is not None and pending[0]==sim.input_revision and pending[1]==sim.response:return
    cancel(sim)
    if _pool is None:_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Weft first contact')
    from .dry_input import copy_physics
    trial=copy(sim)
    trial.ui_cracks=[]
    copy_physics(trial,sim)
    trial.predicting_birth=True
    trial.birth_job=None
    trial.warned_lanes=lanes[0],dict(lanes[1])
    sim.birth_job=(sim.input_revision,sim.response,sim.faces,sim.target.copy(),_pool.submit(calculate,trial,remaining))


def accept(sim,dt):
    serial,response,faces,target,job=sim.birth_job
    if (serial!=sim.input_revision or response!=sim.response or faces is not sim.faces or
            not job.done() or job.cancelled() or not np.array_equal(target,sim.target)):return False
    value=job.result()
    if value is None or not np.array_equal(value[0],sim.yarn_ledger):return False
    trial=value[1]
    # Wall-clock counters and the input identity belong to the actual due step.
    # All contact and topology arrays below are a completed, exclusively owned
    # solution for the very same grip, face layout and complete yarn ledger.
    clocks={k:getattr(sim,k) for k in ('elapsed','step_number','held_seconds','yarn_pending','held_optics_changed','input_revision')}
    warning_ms=1000*(sim.warning['age']+dt)
    sim.__dict__.update(vars(trial))
    sim.__dict__.update(clocks)
    sim.birth_job=None;sim.predicting_birth=False;sim.birth_seated=True
    event=sim.fracture_events[-1]
    event.update(step=sim.step_number,t_s=sim.elapsed,warning_ms=warning_ms)
    return True


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True,cancel_futures=True)
        _pool=None
