"""Prepare one paid cut on private arrays while the viewport follows the hand."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace
from .seam_surface import TornSheet

_pool=None


class CutState(SimpleNamespace):
    duplicate=TornSheet.duplicate
    refresh_layout=TornSheet.refresh_layout


def prepare(state,plan,amount):
    if not plan.grow(state,amount):return None
    state.refresh_layout()
    return state,plan


def advance(sheet,amount,*,settle=False):
    global _pool
    cut=sheet.plan
    future=getattr(sheet,'pending',None)
    committed=False
    if future is not None:
        if not settle and not future.done():return False
        sheet.pending=None
        try:result=future.result()
        except Exception as error:
            # Retry on the main loop with its current, fully committed mesh.
            sheet.deferred=False
            sheet.worker_error=repr(error)
            return cut.grow(sheet,amount)
        if result:
            state,built=result
            for key in ('rest','faces','knit','support_normals','normals','layout','toe','cuff'):
                setattr(sheet,key,getattr(state,key))
            if hasattr(state,'bank_weave'):sheet.bank_weave=state.bank_weave
            if hasattr(state,'material_reference'):sheet.material_reference=state.material_reference
            cut.copies=built.copies;cut.reached=built.reached;cut.done=built.done
            cut.previous_displacement=built.previous_displacement
            cut.checked_rest=built.checked_rest
            committed=True
            if not settle:return True
    if cut.done:return committed
    next_station=cut.reached+1 if cut.reached or cut.fans[0] is not None else 2
    if cut.stations[next_station]>amount:return committed
    if settle:
        if cut.grow(sheet,amount):
            sheet.refresh_layout()
            return True
        return committed
    if _pool is None:_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Weft dry seam')
    state=CutState(rest=sheet.rest.copy(),pos=sheet.rest.copy(),faces=sheet.faces.copy(),knit=sheet.knit,
                   support_normals=sheet.support_normals,normals=sheet.normals,toe=sheet.toe,cuff=sheet.cuff.copy(),
                   seat=getattr(sheet,'seat',None))
    if getattr(sheet,'bank_weave',None) is not None:state.bank_weave=sheet.bank_weave
    if hasattr(sheet,'material_reference'):state.material_reference=sheet.material_reference
    if hasattr(sheet,'clearance'):state.clearance=sheet.clearance
    built=copy(cut);built.copies=cut.copies.copy()
    # The UI can accept another drag or release during this finite solve.
    # A normal release collects earned work. A reset discards this private result.
    quantum=max(cut.stations[next_station],cut.stations[cut.reached]+.009)
    sheet.pending=_pool.submit(prepare,state,built,min(amount,quantum))
    return False


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=True,cancel_futures=True);_pool=None
