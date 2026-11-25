"""Relax newly released yarn on an owned snapshot of a fixed, indexed dry sheet."""
from concurrent.futures import ThreadPoolExecutor

_pool=None


def snapshot(sim):
    # The collision index owns the fixed skin. Workers use its read-only queries
    # and numeric mesh arrays. No scene, RNA mesh or Blender operator is accessed.
    state=object.__new__(type(sim))
    for key in ('rest','faces','contact_normals','project_skin','seat_opening','clearance',
                'strain_cache','tear_support_faces','neutral_history','material_frame','contact_material_band'):
        setattr(state,key,getattr(sim,key,None))
    state.pos=state.rest
    state.opening=sim.opening.copy();state.seating=sim.seating.copy()
    state.monitor_contact=True;state.open_rest=None;state.grip=None
    state.cracks=[dict(c,cells=list(c['cells']),fronts=c['fronts'].copy()) for c in sim.cracks]
    return state


def solve(state):
    state.relaxed_tears()
    return state


def start(state):
    global _pool
    if _pool is None:_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Weft released loops')
    return _pool.submit(solve,state)


def schedule(sim,previous):
    if previous is None or previous[0] is not sim.faces or not getattr(sim,'async_neutral',False):
        return
    # The committed sheet remains available to the hand. New interlocks have
    # already opened in the live topology and use the actual paid-out displacement.
    state=snapshot(sim)
    sim.open_rest=previous
    if getattr(sim,'neutral_job',None) is None:sim.neutral_job=start(state)
    else:sim.neutral_next=state


def collect(sim,*,settle=False):
    job=getattr(sim,'neutral_job',None)
    while job is not None and (settle or job.done()):
        sim.neutral_job=None
        latest=getattr(sim,'neutral_next',None)
        sim.neutral_next=None
        try:
            state=job.result()
        except Exception:
            # Use the live committed arrays and the synchronous solver on failure.
            sim.async_neutral=False;sim.open_rest=None
            sim.relaxed_tears()
            return
        if state.faces is sim.faces:
            sim.neutral_history=state.neutral_history
            if latest is None:
                sim.open_rest=state.open_rest
                sim.strain_cache=state.strain_cache
        if latest is None:return
        latest.neutral_history=state.neutral_history
        sim.neutral_job=job=start(latest)


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True,cancel_futures=True);_pool=None
