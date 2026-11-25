"""One continuous stocking course in stitch coordinates, from left foot to right foot."""
import numpy as np

KNOTS=np.array([(0,-.22),(.365,.045),(.245,.65),(.365,1.045),(.5,1.25),
                (.635,1.045),(.755,.65),(.635,.045),(1,-.22)])
HANDLES=np.array([(.35,-.22),(.385,.312),(.365,.88),(.365,1.25),(.635,1.25),
                  (.635,.88),(.615,.312),(.65,-.22)])
DEPTH=np.array([.70,.15,.70,.95,.65,.95,.70,.15,.70])

def centerline():
    t=np.linspace(0,1,18,endpoint=False)[None,:,None]
    curves=(1-t)**2*KNOTS[:-1,None]+2*t*(1-t)*HANDLES[:,None]+t*t*KNOTS[1:,None]
    return np.vstack((curves.reshape(-1,2),KNOTS[-1]))


def stitch_crossing(axis):
    """A continuous centreline segment crossing the interior of the seed cell."""
    course = centerline()
    matches = []
    for row_offset in (-1, 0):
        path = course + [0, row_offset]
        sides = path[:,axis]-.5
        for edge in np.flatnonzero((sides[:-1]<0)!=(sides[1:]<0)):
            a,b = path[edge:edge+2]
            impact = a+(b-a)*(-sides[edge]/(sides[edge+1]-sides[edge]))
            if .06<impact[1-axis]<.94:
                matches.append((abs(impact[1-axis]-.5), path, edge, impact))
    _,path,edge,impact = min(matches,key=lambda item:item[0])
    before = path[max(0,edge-31):edge+1]
    after = path[edge+1:edge+33]
    return before[abs(before[:,axis]-.5)>1e-7],impact,after[abs(after[:,axis]-.5)>1e-7]


def uncrimp(segment, load):
    """Straighten the loaded span while its held endpoints stay on the knit."""
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(segment,axis=0),axis=1))]
    chord = segment[0]+(lengths/lengths[-1])[:,None]*(segment[-1]-segment[0])
    tension = np.clip((load-.16)/.59,0,1)
    return segment+.93*tension**2*(3.-2.*tension)*(chord-segment)


def retract_tips(left, right, age, normal):
    """Follow the recoil of two released yarn ends from their loaded shoulders."""
    delta = right[0]-left[-1]
    reach = np.exp(-age/.014)*min(.0024,.47*np.linalg.norm(delta))
    if reach<.000008:
        return [left,right]
    direction = delta/np.linalg.norm(delta)
    bend = np.cross(normal,direction)
    t = np.linspace(1/6,1,6)[:,None]
    tail = reach*(t*direction+.28*t**2*(1.-np.exp(-age/.012))*bend)
    return [np.vstack((left,left[-1]+tail)),np.vstack((right[0]-tail[::-1],right))]

def yarn_length(gauges,angle):
    # Convert horizontal and vertical gauge chords to full cell pitches.
    pitch=gauges/[.7,.8]
    # Integrate over the same quadratic centreline used by the rendered knit.
    d=np.diff(centerline(),axis=0)
    dx=pitch[:,0,None]*d[:,0]
    dy=pitch[:,1,None]*d[:,1]
    return np.sqrt(np.maximum(dx*dx+dy*dy+2*dx*dy*angle[:,None],0)).sum(1)


def loose_end(root, across, along, normal, length, grain):
    """An exposed loop shoulder left attached at one tip after disengagement."""
    t=np.linspace(0,1,12)[:,None]
    return root+length*(.88*np.sin(t*np.pi*(.66+.19*grain))*across
                        +(.43+.30*grain)*t*t*along+(.18+.13*grain)*np.sin(t*np.pi*.88)*normal)


def rolled_edge(rails, outward, axis, phase):
    """Twisted rim yarn and lighter strands that surface between loop shoulders."""
    delta=rails[1]-rails[0]
    gap=np.linalg.norm(delta,axis=1)
    across=delta/np.maximum(gap[:,None],1e-12)
    taper=np.clip(gap/.0018,0,1)
    for bank,sign in enumerate((1,-1)):
        center=rails[bank]
        wave=np.sin(4.8*axis+phase+2.2*bank)
        stock=np.sin(1.85*axis-.45*phase+2.4*bank)
        lift=.00012*taper*(.85+.30*wave)
        drift=np.minimum(.00017,.09*gap)*(.65+.26*stock)
        outer=center+lift[:,None]*outward+sign*drift[:,None]*across
        inner=center+(.38*lift)[:,None]*outward-sign*(.00016*taper*(1+.35*wave))[:,None]*across
        yield outer,.075+taper*(1.10+.34*wave+.21*stock)
        yield inner,.025+taper*.52*np.maximum(0,.46+.65*stock)


def retained_courses(rails,rows,crack,phase):
    """Short groups of transverse yarn stay attached beside an advancing tip."""
    if len(crack['cells'])<4 or len(rows)<3:
        return
    courses=np.arange(np.ceil(rows[0]),np.floor(rows[-1])+1)
    blocks=(courses.astype(int)+int(abs(phase)))%9
    courses=courses[(blocks==0)|(blocks==1)|(blocks==5)]
    waist=crack.get('cross_course')
    if waist is not None and crack['course_hold']>.54 and rows[0]<waist-1.5<waist+1.5<rows[-1]:
        span=min(waist-rows[0],rows[-1]-waist)
        bars=np.arange(-10,11)
        bars=bars[((bars+10)*8%21)/21.<np.clip((span-.7)/5.,.20,1.)]
        courses=np.unique(np.r_[courses,waist+bars*.121])
    skew=.22*np.sin(phase+.93*courses)+.19*np.sin(phase*.47)
    shaped='rim_bias' in crack
    if shaped:
        skew+=crack['draw_side']*(.42+.32*np.sin(courses*.61+phase))
    left=np.array([np.interp(courses-skew,rows,v) for v in rails[0].T]).T
    right=np.array([np.interp(courses+skew,rows,v) for v in rails[1].T]).T
    lost=crack.setdefault('severed_courses',set())
    gap=np.linalg.norm(right-left,axis=1)
    bundle=np.zeros(len(courses),bool)
    if waist is not None and crack['course_hold']>.54:bundle=abs(courses-waist)<1.23
    used=np.ones(len(courses),bool)
    if shaped:
        if waist is not None:used &= ~(bundle & (abs(courses-waist)>.22) & (abs(courses-waist-.69)>.12))
        feed=np.where(bundle,.0055+min(.009,.00065*len(crack['cells'])),.003)
    else:
        feed=np.where(bundle,.012+min(.029,.00135*len(crack['cells'])),.0046)
    lengths=feed+.005*(.5+.5*np.sin(phase+1.9*courses))
    lost.update(courses[used & (gap>lengths)])
    used &= (gap>=.0009) & ~np.isin(courses,tuple(lost))
    if not used.any():return
    t=np.linspace(0.,1.,7)
    slack=np.minimum(.0014,.22*(lengths[used]-gap[used]))
    bow=slack[:,None,None]*np.sin(np.pi*t)[None,:,None]*(crack['along']+.27*crack['normal'])
    points=left[used,None]+(right[used]-left[used])[:,None]*t[None,:,None]+bow
    stock=np.where(bundle[used],.27+.07*np.sin(12.5*courses[used]+phase),.22)
    yield from zip(points,stock[:,None]+.22*np.cos(np.pi*t)**2)
