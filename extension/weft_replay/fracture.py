"""Local knitted fracture in material coordinates (course fraction, wale rows)."""
from dataclasses import dataclass

import numpy as np


PITCH = np.array([144, 1])


@dataclass(frozen=True)
class FabricResponse:
    recovery: float = 1.0
    break_extension: float = .80
    run: float = .65
    pretension: float = .04
    fit: float = .75
    wrinkles: bool = True


def chart_delta(uv, origin):
    delta = np.asarray(uv) - origin
    delta[..., 0] -= np.round(delta[..., 0])
    return delta * PITCH


def strain(loops):
    # Neighbour support suppresses skinny-chart outliers. Positive feed leaves
    # some extension in loop shape instead of loading that yarn axially.
    nominal = np.maximum(0., loops.stretch - 1.) - .8 * loops.feed[:, None]
    nearby = np.where(loops.intact[loops.adjacent, None], nominal[loops.adjacent], nominal[:, None])
    return .52 * nominal + .48 * nearby.mean(1)


def seam_bend(along, phase):
    """Displace the split within the loaded stitch lane in material space."""
    return (.15*(np.sin(.48*along+phase)-np.sin(phase))
            +.052*(np.sin(1.17*along-.63*phase)-np.sin(-.63*phase)))


def shared_wales(crack, previous, run):
    """How much of this column was unloaded by a neighbouring lengthwise tear."""
    coupling = 0.
    for old in previous:
        if old is crack or old['load_axis'] != 0 or len(old['cells']) < 3:
            continue
        across, along = chart_delta(crack['uv'], old['uv'])
        gap = max(old['fronts'][0] - along, along - old['fronts'][1], 0.)
        reach = (15. + 2.65 * min(15, len(old['cells']))) * (.74 + .88 * run)
        # A column can shed load into the next few wales. Beyond the
        # paid-out run the original interlocks continue to carry it.
        lateral = max(0., 1. - (across / (2.4 + .13 * gap))**2)
        coupling = max(coupling, lateral * np.clip((reach - gap) / (.48 * reach), 0., 1.))
    return float(coupling)


def aperture(crack, width):
    lo, hi = crack['fronts']
    x = crack['axis']
    if 'pitch' in crack:
        # Open the lips over the actual length of released yarn. The wider
        # longitudinal support off the seam spreads the strain into intact cloth.
        half=.5*(hi-lo)
        t=(x-.5*(lo+hi))/(half+.55*np.minimum(crack['cross_cells'],half))
        phase=crack['curve_phase']
        t=np.clip(t,-1.,1.)
        n=len(crack['cells'])
        drift=(hi+lo)/max(hi-lo,1.)
        shape=np.cos(.5*np.pi*t)
        rounding=crack.get('cap_round',0.)
        if rounding:
            # A free rounded cap has a gentler tangent at the joined end.
            # Broaden the first few open rows, including their attached cloth.
            shape=np.maximum(shape,0.)**(1.-.50*rounding*t**4)
        # Preserve the opened loop shoulders. A short tear has a round belly;
        # a longer one retains a notch at the end that has paid out more yarn.
        if 'cross_course' in crack:
            row=crack['cross_course']
            beyond=min(hi-row,row-lo)
            engage=np.clip((beyond-.75)/2.75,0.,1.)
            # A retained bundle holds two short stretches of lip together.
            # Its material row stays fixed while new loops drop on either side.
            span=max(1.25,.135*n)
            transverse=crack['cross_cells']
            waist=np.exp(-((x-row)/(span+.24*transverse))**2)
            side=np.sign(row)
            smaller=np.clip(.5+side*(x-row)/3.2,0.,1.)
            smaller*=smaller*(3.-2.*smaller)
            # The smaller eye has less free yarn than its neighbour. Keep
            # room between the banks for the transverse yarn that separates them.
            pin=crack['course_hold']*engage
            if 'rim_bias' in crack:
                # A transverse bundle holds the adjacent eye, while most of
                # the released bank can still slip around the main opening.
                room=crack['tube_room']
                shape*=1.-(.40-.29*room)*pin*smaller
                shape*=1.-(.58-.40*room)*pin*waist
            else:
                shape*=1.-.40*pin*smaller
                shape*=1.-.58*pin*waist
        bank=1.+crack['bank_side']*(.13*np.sin(phase+.25*x)+.09*t)
        if 'rim_bias' in crack:
            # Unequal loop length gives the loaded bank a broad belly and
            # leaves a tighter shoulder at the advancing end of the split.
            end=np.sign(crack['cross_course'])
            room=crack['tube_room']
            shape*=1.+room*(.095-.035*end*t+.030*np.sin(phase+.60*x)*np.exp(-.5*(t/.72)**2))
            bank+=.15*room*crack['draw_side']*crack['bank_side']*(1.-t*t)
        shoulder=1.+(.27*drift+.13*np.sin(phase))*t+.04*np.sin(.84*x+phase)
        aspect=crack.get('aperture_aspect',1.)
        run=crack.get('wale_run',0.)
        if run:
            # Each side of the held course has its own available length.
            # Once both fronts have passed it, the near eye can round off
            # without borrowing the free yarn in the longer lobe.
            row=crack['cross_course']
            length=hi-row if row>0 else row-lo
            other=hi-lo-length
            held=run*np.clip((min(length,other)-1.)/3.,0.,1.)
            if held:
                near=(x-row)*np.sign(row)>0.
                span=np.where(near,length,other)
                extension=.55*np.minimum(crack['cross_cells'],span)
                u=np.clip(abs(x-row)/(span+extension),0.,1.)
                small=min(.72,.35+.37*np.sqrt(length/max(other,1.)))
                neck=.75*small
                belly=np.where(near,small,1.)
                f=np.minimum(1.,u/.43)
                body=neck+(belly-neck)*f*f*(3.-2.*f)
                cap=np.maximum(0.,(u-.43)/.57)
                body*=np.sqrt(np.maximum(0.,1.-cap*cap))
                shape+=held*(body-shape)
            aspect+=run*(min(aspect,.38)-aspect)
        width=min(width,aspect*half*crack['pitch'])
        return width*shape*bank*shoulder
    # Close at the two intact fronts. Unequal propagation moves the widest
    # section off center and the released loops leave shallow scallops.
    shape = np.maximum(0., 4 * (x-lo) * (hi-x) / (hi-lo)**2)
    if crack.get('curve_phase') is not None:
        t = np.clip((2*x-lo-hi)/(hi-lo), -1., 1.)
        phase = crack['curve_phase']
        cells = chart_delta((np.asarray(crack['cells'])+.5)/PITCH, crack['uv'])
        centres = cells[:,1-crack['load_axis']]
        order = centres.argsort()
        load = np.asarray(crack['edge_load'])
        shoulder = np.interp(x, centres[order], load[order])
        bias = .20*np.sin(phase) + .14*np.clip((lo+hi)/(hi-lo), -1., 1.)
        profile = np.clip(.78 + .28*shoulder + bias*t + .077*np.sin(3.0*x+phase), .53, 1.45)
        bank = 1. + crack['bank_side']*(.16*np.sin(phase+.27*x) + .10*t)
        return width * shape**(.63+.09*np.cos(phase)) * profile * bank
    if len(crack['cells'])>1:
        along=np.clip((2*x-lo-hi)/(hi-lo),-1,1)
        phase=crack['uv']@[144*.81,.37]
        shape*=1.+.15*(hi+lo)/(hi-lo)*along+.046*np.sin(2.65*x+phase)
    return width * shape**.58


def round_free_caps(crack, rest, offset):
    """Spread the tension of a free cap along its rim and into the attached wales."""
    select=rim_slice(crack)
    rails=crack['rim_ids'][select]
    station=crack['rim_axis'][select]
    if len(station)<5:return
    local=np.searchsorted(crack['ids'],rails)
    banks=rest[rails]+offset[local]
    rim_line=np.vstack((banks[:,0],banks[-2:0:-1,1]))
    cycle=np.vstack((rim_line,rim_line[0]))
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(cycle,axis=0),axis=1))]
    if arc[-1]<.026:return
    # Equal lengths of yarn carry the same tension, including where triangles
    # are concentrated tightly around the tip. Both tips are single points.
    reach=min(.0085,.098*arc[-1])
    samples=np.linspace(-2.,2.,15)
    weight=np.exp(-.5*samples**2);weight/=weight.sum()
    at=(arc[:-1,None]+reach*samples)%arc[-1]
    target=np.column_stack([np.interp(at,arc,cycle[:,axis])@weight for axis in range(3)])
    shift=target-rim_line
    shift*=np.minimum(1.,.003/np.maximum(np.linalg.norm(shift,axis=1),1.e-10))[:,None]
    n=len(station)
    sides=(shift[:n],np.vstack((shift[0],shift[:n-1:-1],shift[n-1])))
    x=crack['axis'];a,b=crack['fronts'];outside=np.maximum(0.,np.maximum(a-x,x-b))
    cap=np.clip((abs((2.*x-a-b)/(b-a))-.40)/.40,0.,1.)
    fade=cap*cap*(3.-2.*cap)*np.exp(-.5*((outside/3.)**2+(crack['cross_cells']/3.2)**2))
    for sign,values in zip((-1,1),sides):
        keep=(crack['bank_side']*sign>0)&(fade>.001)
        delta=np.column_stack([np.interp(x[keep],station,values[:,axis]) for axis in range(3)])
        offset[keep]+=(crack['cap_round']*fade[keep])[:,None]*delta


def released_shape(crack, rest=None):
    opening=aperture(crack,crack['width'])
    offset=opening[:,None]*crack['direction']
    if 'pitch' in crack:
        lo,hi=crack['fronts']
        half=.5*(hi-lo)
        chart=crack['axis']
        t=np.clip((chart-.5*(hi+lo))/half,-1.,1.)
        # Redistribute material length around the cap of the opening. A purely
        # transverse displacement leaves angular tips even with a rounded width profile.
        travel=.97*half*crack['pitch']*(np.sin(.5*np.pi*t)-t)
        travel*=np.clip(crack['width']/(.73*half*crack['pitch']),0.,1.)
        travel*=1.-.64*crack.get('wale_run',0.)
        offset+=travel[:,None]*crack['cap_tangents']
        if 'rim_bias' in crack:
            # Shear both lips in the same direction at each end, so even
            # the two closed tips tilt together without tearing a new face.
            # Spread the return into the surrounding material beyond the tip.
            raw=(chart-.5*(lo+hi))/half
            fade=np.exp(-(np.maximum(abs(raw)-1.,0.)*half/(3.5+.60*half))**2)
            release=min(1.,max(0.,(hi-lo-2.)/5.))*min(1.,crack['width']/(half*crack['pitch']))
            slip=release*half*crack['pitch']*np.sin(.5*np.pi*t)*fade
            tilt=crack['rim_bias']*(1.-.78*crack.get('cap_round',0.))*(1.-.83*crack.get('wale_run',0.))
            offset+=(tilt*slip)[:,None]*crack['bias_tangents']
            # Rotate the open banks with their displaced tips. A shear alone
            # leaves one flattened shoulder and one sharply pointed corner.
            rotation=.14*release*tilt
            bank_slide=-rotation*crack['bank_side']*opening
            offset+=(.10*crack['tube_room']*slip+bank_slide)[:,None]*crack['cap_tangents']
        if 'cross_course' in crack:
            row=crack['cross_course']
            side=np.sign(row)
            lobe=np.clip(.5+side*(chart-row)/3.,0.,1.)
            end=hi if side>0 else lo
            # Yarn drawn past a retained course shifts the smaller opening
            # towards the loaded bank. It stays attached at both closed tips.
            window=np.clip(side*(end-chart)/3.,0.,1.)
            payout=.45*crack['width']
            if 'rim_bias' in crack:
                # The drawn shoulder shares the same finite length of free
                # yarn as the belly. A raw target width can exceed it several-fold.
                available=.36*min(crack['width'],crack['aperture_aspect']*half*crack['pitch'])
                payout+=crack['tube_room']*(available-payout)
            payout*=1.-.72*crack.get('wale_run',0.)
            bias=payout*crack['draw_side']*lobe*window*crack['course_hold']
            offset+=bias[:,None]*crack['course_tangents']
        if rest is not None and crack.get('cap_round',0.)>.25 and not crack.get('wale_run',0.):
            round_free_caps(crack,rest,offset)
    return offset


def rim_slice(crack):
    open_node = (crack['rim_axis'] > crack['fronts'][0]) & (crack['rim_axis'] < crack['fronts'][1])
    open_edge = open_node[:-1] | open_node[1:]
    strips = np.flatnonzero(open_edge)
    return slice(strips[0],strips[-1]+2) if len(strips) else slice(0,0)


def rim(crack, points):
    return points[crack['rim_ids'][rim_slice(crack)]]


def tube_radius(points,chart,row):
    """Fit the worn tube across nearby courses using its original, unrefined vertices."""
    along=chart[:,1]-row
    band=abs(along)<9.
    station=along[band]
    angle=2.*np.pi*chart[band,0]
    w=np.exp(-.5*(station/3.)**2)
    design=np.column_stack((np.ones(len(w)),np.cos(angle),np.sin(angle),station))
    if len(w)<4:return 0.
    coeff,_,rank,_=np.linalg.lstsq(w[:,None]*design,w[:,None]*points[band],rcond=None)
    return float(np.linalg.svd(coeff[1:3],compute_uv=False)[-1]) if rank==4 else 0.




def open_area(banks):
    if len(banks) < 2:
        return 0.
    a, b = banks[:-1].transpose(1, 0, 2)
    c, d = banks[1:].transpose(1, 0, 2)
    return float(.5 * (np.linalg.norm(np.cross(b-a, c-a), axis=1).sum()
                       + np.linalg.norm(np.cross(b-c, d-c), axis=1).sum()))


def linked_runs(cracks):
    """Find neighbouring tips along a common wale or course."""
    links = []
    for i, first in enumerate(cracks):
        axis = first['load_axis']
        for second in cracks[i+1:]:
            if axis != second['load_axis']:
                continue
            across, along = chart_delta(second['uv'], first['uv'])[[axis,1-axis]]
            if abs(across) > 3.25 or abs(along) < 1.:
                continue
            side = 1 if along > 0. else -1
            far, near = (1,0) if side > 0 else (0,1)
            left = first['fronts'][far]
            right = along+second['fronts'][near]
            gap = side*(right-left)
            strength = (1.-abs(across)/3.5)*(1.-max(gap,0.)/19.)
            if -3. <= gap <= 15. and strength > .08:
                links.append((first, second, side, across, left, right, strength))
    return links


def loose_stock(stock, first_row, cracks, links, run):
    """Retained loose yarn beyond the split and between adjacent fronts."""
    stock.fill(0.)
    if run <= 0.:
        return

    def deposit(uv, axis, transverse, along, loss):
        cell = np.floor(uv*PITCH).astype(int)
        x, y = (cell + (np.stack((transverse,along), axis=1)[:,[axis,1-axis]])).T
        y = y - first_row
        use = (y >= 0) & (y < stock.shape[0])
        np.maximum.at(stock, (y[use], x[use] % 144), loss[use])

    for crack in cracks:
        axis = crack['load_axis']
        n = len(crack['cells'])
        length = min(17, 4+2*n)
        t = np.arange(1,length+1)
        falloff = (1.-(t/(length+1))**1.15)**1.7
        for direction, front in zip((-1,1),crack['span']):
            extent = front+direction*t
            for shoulder, gain in ((0,1.),(-1,.34),(1,.34)):
                cue = np.where((t+shoulder)%3 == 0, .48, 1.)
                deposit(crack['uv'],axis,np.full_like(t,shoulder),extent,
                        (.23+.17*run)*min(1.,.35+.15*n)*gain*falloff*cue)
    for a, _, _, across, left, right, strength in links:
        n = int(abs(right-left))
        if n < 1:
            continue
        t = np.arange(1,n+1)/(n+1)
        extent = np.rint(left+t*(right-left)).astype(int)
        transverse = np.rint(t*across).astype(int)
        for side, weight in ((0,1.),(-1,.33),(1,.33)):
            rung = np.where(extent%3 == 0, .57, 1.)
            deposit(a['uv'],a['load_axis'],transverse+side,extent,
                    weight*rung*(.24+.18*run)*np.sqrt(strength))
