"""Directional yarn remaining in a run, in the same stitch chart as the tear."""
import bpy
import numpy as np

from .fracture import seam_bend
from .sheer_optics import FINISH, REPEAT_LABEL, math_socket


def fine_ground(material):
    """Retain the mean coverage and tint of the four stockings at leg distance."""
    tree=material.node_tree
    optics=tree.nodes.get(FINISH)
    if optics is None or tree.nodes.get("Weft fine ground") is not None:
        return
    socket=optics.inputs["Yarn coverage"]
    if not socket.is_linked:
        return
    detail=math_socket(tree,"MULTIPLY_ADD",socket.links[0].from_socket,.46,.315*(1.-.46))
    detail.node.name="Weft fine ground"
    tree.links.new(detail,socket)
    repeat=next((n for n in tree.nodes if n.label==REPEAT_LABEL),None)
    if repeat is not None:
        repeat.inputs["Scale"].default_value=2.45
    shader=tree.nodes.get("Principled BSDF")
    if shader.inputs["Normal"].is_linked:
        bump=shader.inputs["Normal"].links[0].from_node
        if bump.type=="BUMP":
            bump.inputs["Strength"].default_value=.13


def bind_run(material,sim):
    fine_ground(material)
    tree=material.node_tree
    shader=tree.nodes.get("Principled BSDF")
    if shader is None or not shader.inputs["Alpha"].is_linked:
        return
    # The atlas resolves the shoulders of a dropped course; the shader
    # resolves the narrower filaments that still cross its opening.
    scale=3
    h,w=sim.failed_cells.shape
    pixels=np.zeros((h*scale,w*scale,4),dtype=np.float32)
    pixels[:,:,3]=1.
    atlas=bpy.data.images.new("Weft • dropped stitches",w*scale,h*scale,alpha=True,is_data=True)
    atlas.alpha_mode="CHANNEL_PACKED"
    atlas["weft_first_course"]=int(sim.first_row)
    atlas.pixels.foreach_set(pixels.ravel())
    atlas.update()
    sim.run_pixels,sim.run_image,sim.run_stamp=pixels,atlas,None
    nodes,link=tree.nodes,tree.links.new
    knit=nodes.new("ShaderNodeUVMap")
    knit.uv_map="Knit"
    split=nodes.new("ShaderNodeSeparateXYZ")
    link(knit.outputs["UV"],split.inputs[0])
    x,y=split.outputs[:2]
    def op(kind,*args):
        return math_socket(tree,kind,*args)
    address=nodes.new("ShaderNodeCombineXYZ")
    link(op("DIVIDE",x,w),address.inputs[0])
    link(op("DIVIDE",op("SUBTRACT",y,sim.first_row),h),address.inputs[1])
    lookup=nodes.new("ShaderNodeTexImage")
    lookup.label="Surviving wale and course yarn"
    lookup.image=atlas
    link(address.outputs[0],lookup.inputs[0])
    channels=nodes.new("ShaderNodeSeparateColor")
    link(lookup.outputs["Color"],channels.inputs[0])

    def remaining(across,along):
        # Fine lengthwise strands continue into the opaque material. Surviving
        # transverse courses bow slightly where the columns have slipped.
        column=op("MULTIPLY_ADD",op("SINE",op("MULTIPLY",along,.19)),.055,across)
        warp=op("MULTIPLY_ADD",op("COSINE",op("MULTIPLY",column,2.*np.pi)),.5,.5)
        bundle=op("MULTIPLY_ADD",op("COSINE",op("MULTIPLY",column,2.*np.pi/3.)),.5,.5)
        rib=op("MULTIPLY_ADD",op("SINE",op("MULTIPLY",across,.83)),.19,along)
        rib=op("MULTIPLY_ADD",op("SINE",op("MULTIPLY",along,.28)),.27,rib)
        rib=op("MULTIPLY_ADD",op("COSINE",op("MULTIPLY",rib,4.35)),.5,.5)
        filaments=op("MULTIPLY_ADD",op("POWER",warp,10.),.31,op("MULTIPLY_ADD",op("POWER",bundle,12.),.24,.012))
        rib_stock=op("MULTIPLY_ADD",lookup.outputs["Alpha"],-.27,.32)
        return op("MULTIPLY_ADD",op("POWER",rib,7.),rib_stock,filaments)

    a,b,selvedge=channels.outputs[:3]
    missing=op("MAXIMUM",op("MULTIPLY",a,op("SUBTRACT",1.,remaining(x,y))),
               op("MULTIPLY",b,op("SUBTRACT",1.,remaining(y,x))))
    intact=shader.inputs["Alpha"].links[0].from_socket
    # Yarn removal reduces optical depth. Scaling alpha directly makes a
    # lightly run stocking look like a bright, solid stripe on pale skin.
    transmittance=op("MAXIMUM",.0001,op("SUBTRACT",1.,intact))
    stock=op("MULTIPLY_ADD",lookup.outputs["Alpha"],1.85,-.39)
    stock=op("MINIMUM",1.,op("MAXIMUM",.015,stock))
    remaining_stock=op("MULTIPLY",stock,op("SUBTRACT",1.,missing))
    # Rolling several loops together increases the optical path through the
    # nylon. It therefore retains the actual dye and responds to every palette.
    depth=op("MULTIPLY_ADD",selvedge,1.13,remaining_stock)
    coverage=op("SUBTRACT",1.,op("POWER",transmittance,depth))
    link(coverage,shader.inputs["Alpha"])


def paint_ladder(pixels, first_row, axis, tip, side, length, half_width, strength):
    """Resolve alternating dropped and retained courses beyond a physical seam tip."""
    if length < 2. or strength <= 0.:
        return
    h, w = pixels.shape[:2]
    scale = w // 144
    # A course that runs around the leg meets its opposite front after half
    # a turn. Lengthwise runs instead end at the real chart boundary.
    if axis == 1:
        length = min(length, 70.)
    reach = np.array([half_width + 3., length * .5 + 2.])
    if axis == 1:
        reach = reach[::-1]
    center = np.array(tip, dtype=float)
    center[1-axis] += side * length * .5
    cols = np.arange(int(np.floor((center[0]-reach[0])*scale)),
                     int(np.ceil((center[0]+reach[0])*scale)))[:w]
    rows = np.arange(max(0, int(np.floor((center[1]-reach[1]-first_row)*scale))),
                     min(h, int(np.ceil((center[1]+reach[1]-first_row)*scale))))
    if not len(cols) or not len(rows):
        return
    x = (cols[None, :] + .5) / scale
    y = (rows[:, None] + .5) / scale + first_row
    cross, course = (x, y) if axis == 0 else (y, x)
    u = np.clip(side * (course-tip[1-axis]) / length, 0., 1.)
    # The knit remains dense at the freed lip. The widest eyes are farther
    # along the run, where loop length has passed into neighbouring courses.
    envelope = np.sin(np.pi*u)**.61
    lane = tip[axis]
    wave = 2.*np.pi/144.
    centerline = lane + envelope*(.46*np.sin(wave*4*course + .31*np.sin(wave*lane))
                                  + .20*np.sin(wave*9*course))
    along_lane = cross-centerline
    width = half_width*(.22+.78*envelope)
    across = np.clip(along_lane/np.maximum(width,.35), -1.7, 1.7)
    # Tie the rungs to absolute stitch rows. Two runs entering the same
    # bridge agree on which courses still carry yarn, including at the UV wrap.
    station = (course + .54*np.sin(wave*3*course) + .26*np.sin(wave*7*course)
               + (.24+.65*np.sin(wave*3*course + wave*lane))*across
               + (.32+.22*np.cos(wave*6*course))*across**2) / 4.
    index = np.floor(station+.5)
    phase = index*(2.*np.pi/36.)
    retained = .5+.5*np.cos(phase*5 + .8*np.sin(phase*3) + .40*np.sin(wave*lane))
    eye_width = width*(.81+.22*np.sin(phase*7)+.13*np.cos(phase*11))
    # One lip can stay pinned by an adjacent wale; the group narrows and
    # opens off centre before the slipped courses regain their full breadth.
    bias = width*(.12*np.sin(phase*3)+.13*(1.-retained))
    position = (along_lane-bias)/(eye_width+.03)
    edge = np.clip(1.-position**2, 0., 1.)
    height = (.73+.17*np.sin(phase*5)+.11*np.cos(phase*11))*edge**.46
    # Use a finite optical shoulder so small slots narrow into wisps. Each
    # clearer eye has a darker surviving course and a few filament crossings.
    dy = abs(station-index)*4.
    vertical = np.clip((height+.22-dy)/.50, 0., 1.)
    lateral = np.clip((1.-abs(position))*eye_width/.48, 0., 1.)
    opening = vertical*lateral
    opening *= opening*(3.-2.*opening)
    haze = np.clip((width+.46-abs(along_lane))/(.82+.24*width), 0., 1.)
    exposure = strength*envelope**.78*(.60+.32*retained)
    cut = strength*.14*envelope*haze + exposure*.84*opening
    idx = np.ix_(rows, cols % w)
    band = pixels[:, :, axis]
    band[idx] = np.maximum(band[idx], cut)


def shoulder_eyes(cross,along,crack,run):
    """Fraction of local yarn remaining beside the more heavily loaded front."""
    sign=np.sign(crack['cross_course'])
    hand=crack['draw_side']
    phase=crack['curve_phase']
    n=len(crack['cells'])
    growth=np.clip((n-3.)/6.,0.,1.)*(.54+.46*run)
    # In a longitudinal run the pulled yarn feeds the physical slot.
    # The shoulder columns keep their stock beside its narrow neck.
    growth*=1.-crack.get('wale_run',0.)
    payout=max(abs(crack['cross_course']),1.)
    depth=np.clip((sign*crack['fronts'][int(sign>0)]-1.)/3.8,0.,1.)
    # Both loop heads stay at fixed material coordinates once their
    # supporting wale is released. Their stock drains as the front advances.
    remaining=np.ones(np.broadcast_shapes(cross.shape,along.shape))
    gathered=np.zeros_like(remaining)
    for size,station,bank in ((1.,payout+6.45+.37*np.sin(phase),.68*hand),
                              (.61,payout+.76,-hand*(2.62+.31*np.cos(phase)))):
        strength=growth*depth
        if size==1. and crack.get('cap_round',0.):
            # Once the actual advancing tip takes up this loop, its former
            # shoulder eye drains into the cut. It no longer sits on the lip.
            clear=np.clip((station-sign*crack['fronts'][int(sign>0)]-.9)/2.6,0.,1.)
            strength*=1.-crack['cap_round']*(1.-clear*clear*(3.-2.*clear))
        if size<1.:
            strength*=np.clip(.64+.35*crack['course_hold']+.17*np.sin(phase),0.,1.)
        if strength<.09:continue
        u=cross-bank
        v=sign*along-station
        # The two shoulders of an eye do not release at the same course.
        # Leave a pointed corner at the end where the adjacent wale still grips.
        height=size*(1.32+.23*crack['course_hold'])*(.32+.68*strength)
        width=size*(2.60+.34*np.cos(phase))*(.35+.65*strength)
        lateral=u/max(width,.12)
        vertical=(v-(.42*hand*u+.18*u*u/max(width,.12)))/max(height,.12)
        radius=np.sqrt((lateral/(1.+.14*np.tanh(vertical)))**2+vertical**2)
        edge=np.clip((1.12-radius)/.32,0.,1.)
        edge*=edge*(3.-2.*edge)
        opened=min(.995,1.16*strength)*edge
        remaining=np.minimum(remaining,1.-opened)
        gathered=np.maximum(gathered,strength*(.27+.13*np.sin(phase+.7*u))*
                            np.exp(-((radius-1.13)/.25)**2))
    return remaining,gathered


def paint_span(pixels,first_row,crack,run,warning=False):
    """Draw the opened column and taper into neighbouring intact interlocks."""
    height,width=pixels.shape[:2]
    scale=width//144
    across_axis=crack['load_axis']
    uv=crack['uv']*np.array([144.,1.])
    lo,hi=crack['fronts']
    n=max(1.,hi-lo)
    phase=crack['curve_phase']
    if warning:
        tail=np.array([3.,3.])
        width_cells=1.22
        strength=crack['exposure']
    else:
        slip=min(n,24.)*run
        tails=np.array([abs(lo),abs(hi)])+.25*n
        tail=3.+(10.+3.2*slip)*(.55+.85*tails/max(tails.max(),1.))*run
        width_cells=min(4.75,1.1+.32*n)*(.65+.44*run)
        strength=min(.96,.61+.056*n)*(.60+.40*run)
    extent=[width_cells+3.,max(-lo+tail[0],hi+tail[1])+2.]
    radius=np.array(extent if across_axis==0 else extent[::-1])
    c=np.arange(np.floor((uv[0]-radius[0])*scale),np.ceil((uv[0]+radius[0])*scale),dtype=int)
    c=c[:width]
    r=np.arange(max(0,int(np.floor((uv[1]-radius[1]-first_row)*scale))),
                min(height,int(np.ceil((uv[1]+radius[1]-first_row)*scale))),dtype=int)
    if not len(r) or not len(c):
        return
    dx=(c[None,:]+.5)/scale-uv[0]
    dy=(r[:,None]+.5)/scale+first_row-uv[1]
    cross,along=(dx,dy) if across_axis==0 else (dy,dx)
    tip=np.maximum(lo-along,along-hi)
    path=np.maximum(0.,1.-np.maximum(0.,tip)/np.where(along<lo,tail[0],tail[1]))
    bend=seam_bend(along,phase)+.12*np.sin(.25*along+phase)*path
    span=width_cells*path**.73*(1.+.085*np.sin(.31*along+phase))
    distance=np.abs(cross-bend)
    feather=.72+.34*span
    core=np.clip((span+.24-distance)/feather,0.,1.)
    core=core*core*(3.-2.*core)
    loss=strength*core*path**.79
    pile=.72*np.exp(-((distance-span-.25)/.36)**2)*path
    if not warning:
        # The fan stretches out of the tips. At a fully open lateral bank
        # the yarn has gathered against the rim instead of bleaching a halo.
        interior=np.clip(-tip/min(3.5,max(1.1,.16*n)),0.,1.)
        interior=interior*interior*(3.-2.*interior)
        if 'cross_course' in crack:
            interior=interior*(1.-.75*crack['course_hold']*np.exp(-((along-crack['cross_course'])/2.)**2))
        loss*=.36*(1.-.88*interior)
        pile=np.maximum(pile,.74*interior*np.exp(-(distance/.74)**2))
    idx=np.ix_(r,c%width)
    if not warning and 'rim_bias' in crack:
        # A narrow rolled lip opens into a wider, more translucent set of
        # stretched wales. Its width and stock differ from bank to bank.
        bank=np.sign(cross-bend)
        fleck=.5+.5*np.sin(1.8*along+phase+.9*bank)
        weight=np.clip((1.1-tip)/1.65,0.,1.)
        rim_width=.55+.35*fleck+.20*(bank*crack['draw_side']>0.)
        lip=weight*(.53+.30*fleck)*np.exp(-(distance/rim_width)**2)
        fan=weight*(.15+.15*fleck)*np.exp(-((distance-rim_width-1.0)/1.05)**2)
        loss=np.maximum(loss,fan)
        pile=np.maximum(.18*pile,lip)
        stock,rolled=shoulder_eyes(cross,along,crack,run)
        pixels[:,:,3][idx]=np.minimum(pixels[:,:,3][idx],stock)
        pile=np.maximum(pile,rolled)
    for channel,field in ((across_axis,loss),(2,pile)):
        band=pixels[:,:,channel]
        band[idx]=np.maximum(band[idx],field)
    if warning:
        return

    # Prefer the end that has already paid out more interlocks. The apparent
    # length and breadth follow the number of broken stitches and run setting.
    leading = 1 if hi+lo >= -.22*n else -1
    for side, front in ((-1, lo), (1, hi)):
        tip_uv = uv.copy()
        tip_uv[1-across_axis] += front
        tip_uv[across_axis] += seam_bend(front, phase)
        long_side = side == leading
        length = (26.+6.4*min(n,18.)) * run * (1.32 if long_side else .53)
        breadth = (1.36+.145*min(n,13.)) * (.62+.64*run) * (1. if long_side else .73)
        exposure = min(1., (.42+.075*n)*(.55+.69*run)) * (1. if long_side else .72)
        paint_ladder(pixels,first_row,across_axis,tip_uv,side,length,breadth,exposure)


_paint_pool=None


def paint_released(shape,first,cracks,run):
    pixels=np.zeros(shape,dtype=np.float32);pixels[:,:,3]=1.
    for crack in cracks:paint_span(pixels,first,crack,run)
    return pixels


def prepare_run(sim,events):
    global _paint_pool
    previous=getattr(sim,'run_job',None)
    if previous is not None:previous[1].cancel()
    if _paint_pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _paint_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Weft knit atlas')
    # paint_span reads only numeric chart and rim data. A subsequent crack
    # can replace its fronts and width while the image is being prepared.
    cracks=[dict(c,fronts=c['fronts'].copy(),cells=tuple(c['cells'])) for c in sim.cracks]
    run=sim.response.run
    job=_paint_pool.submit(paint_released,sim.run_pixels.shape,sim.first_row,cracks,run)
    sim.run_job=((events,run,None,-1),job)


def shutdown():
    global _paint_pool
    if _paint_pool is not None:
        _paint_pool.shutdown(wait=True,cancel_futures=True);_paint_pool=None


def update_run(sim):
    if not hasattr(sim,"run_image"):
        return
    warning=sim.warning
    age=min(6,int(7*max(warning['age']/warning['duration'],warning.get('extension',0.)))) if warning is not None else -1
    seed=None if warning is None else (int(warning['axis']),tuple(sim.patch.loops.cells[warning['stitches'][0]]))
    stamp=(len(sim.fracture_events),sim.response.run,seed,age)
    if stamp==sim.run_stamp:
        return
    sim.run_stamp=stamp
    pending=getattr(sim,'run_job',None)
    sim.run_job=None
    settled=stamp[:2]
    cached=getattr(sim,'released_run_atlas',None)
    if cached is not None and cached[0]==settled:
        pixels=cached[1]
        if pending is not None:pending[1].cancel()
    elif pending is not None and pending[0][:2]==settled:
        try:pixels=pending[1].result()
        except Exception:pixels=paint_released(sim.run_pixels.shape,sim.first_row,sim.cracks,sim.response.run)
    else:
        if pending is not None:pending[1].cancel()
        pixels=paint_released(sim.run_pixels.shape,sim.first_row,sim.cracks,sim.response.run)
    # A warning moves over an existing, unchanged paid-out yarn map. Paint
    # its temporary stitches onto a copy and keep the original map exact.
    sim.released_run_atlas=settled,pixels
    if warning is not None:
        pixels=pixels.copy()
    sim.run_pixels=pixels
    if warning is not None:
        _,cells,reveal=sim.loaded_stitches()
        axis=warning['axis']
        active=cells[reveal>.12]
        if len(active):
            origin=(cells[0]+.5)/[144.,1.]
            delta=active[:,1-axis]-cells[0,1-axis]
            if axis==1:
                delta=(delta+72)%144-72
            cue=dict(uv=origin,fronts=(delta.min()-.5,delta.max()+.5),load_axis=axis,
                     curve_phase=float(origin@[84.4,1.13]),exposure=.28+.11*age)
            paint_span(pixels,sim.first_row,cue,sim.response.run,True)
    from .atlas_transfer import publish
    publish(sim,'run',pixels)
    if not sim.dry or getattr(sim,'pose',None) is not None:sim.run_image.update()
