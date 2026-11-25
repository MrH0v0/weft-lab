"""Freeze live geometry at save; remove native callbacks before IDs disappear."""
import json

import bpy
import numpy as np
from bpy.app.handlers import persistent

from .controls import VERSION


def snapshot(tool):
    monitor = getattr(tool, 'monitor', None)
    frame, subframe = tool.committed_pose()
    data = dict(schema=2 if getattr(tool,'continued',False) else 1, version=VERSION, mode='wet' if tool.wet_mode else 'dry',
                source=tool.obj['weft_live_source'].name, pose_frame=frame, pose_subframe=subframe,
                physics_step=tool.wet_step if tool.wet_mode else tool.sim.step_number,
                held_before_finish=tool.sim.grip is not None,
                fracture=tool.sim.fracture_readings(),
                wet=tool.wet.reading() if tool.wet else None,
                monitor=monitor.row if monitor else None,
                persistence=('Frozen geometry, materials and readings' if tool.wet_mode else
                             'Dry openings can be picked again in the saved pose'))
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    text=tool.obj.get('weft_snapshot')
    if text is None:
        text=bpy.data.texts.new(tool.obj.name+' • frozen readings.json')
        tool.obj['weft_snapshot']=text
    text.clear()
    text.write(json.dumps(data,ensure_ascii=False,indent=2,default=encode,allow_nan=False)+'\n')
    tool.obj['weft_release_version']=VERSION
    tool.obj['weft_live_frame']=frame
    tool.obj['weft_live_tears']=len(tool.sim.cracks)
    tool.obj['weft_snapshot_mode']=data['mode']


def finish_all():
    from .live_view import ACTIVE
    for tool in list(ACTIVE.values()):
        area=getattr(tool.area,'area',tool.area)
        with bpy.context.temp_override(window=tool.window,area=area,region=tool.region):
            tool.finish(bpy.context)


@persistent
def before_change(*_):
    finish_all()


@persistent
def after_load(*_):
    # No Python solver survives a .blend load. Saved live surfaces are snapshots.
    from .live_view import ACTIVE
    from .monitor_ui import MONITOR_OWNER, SOCK_LAYER_NAME, remove_sock_layer
    from .yarn_chart import local_bank_highlight
    ACTIVE.clear()
    for scene in bpy.data.scenes:
        if 'weft_check_running' in scene:
            del scene['weft_check_running']
    for obj in tuple(bpy.data.objects):
        legacy = obj.name.startswith(SOCK_LAYER_NAME) and obj.get('weft_wet_owner') is not None
        if obj.type == 'MESH' and (MONITOR_OWNER in obj or legacy):
            remove_sock_layer(obj)
        elif obj.type == 'MESH' and obj.get('weft_snapshot_mode') == 'dry' and obj.get('weft_live_source'):
            for material in obj.data.materials:
                local_bank_highlight(material)


def register():
    for collection in (bpy.app.handlers.save_pre,bpy.app.handlers.load_pre,
                       bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        collection.append(before_change)
    bpy.app.handlers.load_post.append(after_load)


def unregister():
    finish_all()
    from .seam_worker import shutdown
    shutdown()
    from . import bank_finish
    bank_finish.shutdown()
    from . import dry_input
    dry_input.shutdown()
    from . import neutral_worker
    neutral_worker.shutdown()
    from . import warning_birth
    warning_birth.shutdown()
    from . import warned_lane
    warned_lane.shutdown()
    from . import run_finish
    run_finish.shutdown()
    for collection in (bpy.app.handlers.save_pre,bpy.app.handlers.load_pre,
                       bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if before_change in collection: collection.remove(before_change)
    if after_load in bpy.app.handlers.load_post: bpy.app.handlers.load_post.remove(after_load)
