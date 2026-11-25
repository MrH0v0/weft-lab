"""Surviving longitudinal yarn between two open parts of the same stocking."""
import numpy as np

BAND='Weft drawn course stock'
COORD='Weft drawn course coordinates'
SPAN='Weft surviving wale span'
STAMP='weft_drawn_course_events'
JOINS='weft_rounded_split_sources'

def smooth(x):
    t=np.clip(x,0.,1.)
    return t*t*(3.-2.*t)

def wrap(x):
    return x-144.*np.round(x/144.)

def samples(sim):
    """Physical free edges and the released material cells which own them."""
    out=[]
    if hasattr(sim,'corner_uv'):
        for c in sim.cracks:
            owners=[]
            for k in c['members']:
                src=sim.sources[k]
                owners.append(dict(id=k,open_arc=bool(c.get('notch')),paid=len(src['cells']),material_uv=tuple(src['material_uv']),
                                   run_axis=src['run_axis']))
            out.append((np.asarray(c['rim']).copy(),owners))
    else:
        from .fracture import rim_slice
        for c in sim.cracks:
            lips=c['rim_ids'][rim_slice(c)]
            if len(lips)>2:
                out.append((np.r_[lips[:,0],lips[::-1,1]],[dict(paid=len(c['cells']),material_uv=tuple(c['uv']),run_axis=('wale' if c['load_axis']==0 else 'course'))]))
    return out

def lobes(chart,knit,edges):
    result=[]
    for ring,owners in edges:
        if not owners or len(ring)<5:continue
        coords=chart[ring].copy()
        coords[:,0]=coords[0,0]+wrap(coords[:,0]-coords[0,0])
        arc=np.linalg.norm(np.roll(coords,-1,axis=0)-coords,axis=1)
        if owners[0].get('open_arc'):arc[-1]=0.
        mass=.5*(arc+np.roll(arc,1))
        for src in owners:
            if src['paid']<3 or src['run_axis']!='wale':continue
            seed=np.asarray(src['material_uv'])*[144.,1.]
            use=np.ones(len(ring),bool)
            if len(owners)>1:
                # Paid cells also include the connecting cut. The lobes are
                # anchored at the two original apertures, far from its middle.
                use=(abs(wrap(knit[ring,0]-seed[0]))<3.2)&(abs(knit[ring,1]-seed[1])<19.)
            if use.sum()<5:continue
            p=coords[use];w=mass[use]
            def quantile(x,at):
                order=np.argsort(x);values=x[order];weight=w[order]
                return np.interp(at*np.sum(weight),np.cumsum(weight)-.5*weight,values)
            low,high=quantile(p[:,1],np.array([.045,.955]))
            left,right=quantile(p[:,0],np.array([.035,.965]))
            if high-low<1. or right-left<1.:continue
            half=.5*(right-left)
            y=p[:,1];length=high-low
            x=[]
            for t in (.15,.85):
                weight=w*np.exp(-.5*((y-(low+t*length))/max(.28*length,.5))**2)
                x.append(float(np.average(p[:,0],weights=weight)))
            result.append((low,high,x[0],x[1],half,src['paid']))
    return sorted(result,key=lambda h:.5*(h[0]+h[1]))

def crossings(chart,edges,start,end):
    """Fractions at which a centreline enters or leaves a real opening."""
    origin=np.asarray(start);line=np.asarray(end)-origin
    hits=[]
    for ring,owners in edges:
        a=chart[ring]-origin;b=np.roll(a,-1,axis=0)
        if owners[0].get('open_arc'):a=a[:-1];b=b[:-1]
        a[:,0]=wrap(a[:,0]);b[:,0]=a[:,0]+wrap(b[:,0]-a[:,0])
        side=b-a;det=line[0]*side[:,1]-line[1]*side[:,0]
        valid=abs(det)>1.e-8
        along=np.divide(a[:,0]*side[:,1]-a[:,1]*side[:,0],det,out=np.zeros_like(det),where=valid)
        across=np.divide(a[:,0]*line[1]-a[:,1]*line[0],det,out=np.zeros_like(det),where=valid)
        hits.extend(along[valid&(across>=0.)&(across<1.)&(along>0.)&(along<1.)])
    return np.array(hits)


def connections(shapes,run,chart,edges):
    candidates=[]
    for i,low in enumerate(shapes):
        for j,high in enumerate(shapes[i+1:],i+1):
            gap=high[0]-low[1];dy=.5*(high[0]+high[1]-low[0]-low[1])
            dx=abs(wrap(high[2]-low[3]))
            if dy<3. or gap<-.32*min(low[1]-low[0],high[1]-high[0]):continue
            reach=(15.+2.65*min(15.,min(low[5],high[5])))*(.74+.88*run)
            if gap>reach or dx>1.6+.65*(low[4]+high[4])+.15*max(0.,gap):continue
            hits=crossings(chart,edges,(low[3],low[1]),(low[3]+wrap(high[2]-low[3]),high[0]))
            if np.any((hits>.14)&(hits<.86)):continue
            cost=max(1.,gap)+2.6*dx
            candidates.append((cost,i,j))
    above=set();below=set();pairs=[]
    for _,i,j in sorted(candidates):
        if i not in above and j not in below:
            pairs.append((i,j));above.add(i);below.add(j)
    return pairs

def fabric(chart,knit,edges,run):
    """Blend only adjacent facing holes linked by paid-out columns in Knit."""
    layers=np.zeros((len(chart),9),'f')
    shapes=lobes(chart,knit,edges)
    if len(edges)<2 or len(shapes)<2 or run<=0.:return layers
    pairs=connections(shapes,run,chart,edges)
    u,v=chart.T
    # Texture lookup uses the face-corner Knit plus this small fitted
    # displacement. It has a consistent phase across shared vertices.
    layers[:,3]=wrap(chart[:,0]-knit[:,0])
    layers[:,4]=chart[:,1]-knit[:,1]
    owner=np.full(len(chart),np.inf)
    def lay(center,radius,end,amplitude):
        x=wrap(u-center);a=abs(x)/np.maximum(radius,.42)
        cost=a+24.*(1.-end)
        best=cost<owner;owner[best]=cost[best]
        layers[best,6]=x[best];layers[best,7]=radius[best];layers[best,8]=end[best]
        cover=end*smooth((1.42-a)/.56)
        clear=smooth((1.10-a)/.48)
        hem=np.exp(-((a-1.02)/.14)**2)
        take=cover>layers[:,0]
        layers[take,0]=cover[take]
        layers[take,1]=(amplitude*clear)[take]
        layers[take,2]=hem[take]
    # Once neighbouring fronts meet, their former ladder rungs run out
    # into the new field. Isolated fronts outside the cluster keep theirs.
    active={i for pair in pairs for i in pair}
    for i,one in enumerate(shapes):
        if any(i!=j and abs(wrap(one[2]-two[2]))<18. and
               abs(.5*(one[0]+one[1]-two[0]-two[1]))<54. for j,two in enumerate(shapes)):
            active.add(i)
    if not active:return np.zeros_like(layers)
    for i in active:
        lo,hi,xlo,xhi,half,n=shapes[i]
        center=xlo+wrap(xhi-xlo)*np.clip((v-lo)/(hi-lo),0.,1.)
        tip=np.maximum(lo-v,v-hi)
        reach=(28.+6.4*min(n,18))*run*1.35
        outside=smooth((tip+.5)/3.)
        band=smooth((max(half+1.8,6.2)-abs(wrap(u-center)))/2.1)
        layers[:,5]=np.maximum(layers[:,5],band*outside*smooth((reach+4.-tip)/8.))
    joined_up={i for i,j in pairs};joined_down={j for i,j in pairs}
    for i,j in pairs:
        lower,upper=shapes[i],shapes[j]
        al,ah,_,ax,aw,_=lower;bl,bh,bx,_,bw,_=upper
        # Continue the same wales a little way around each rounded shoulder.
        begin=ah-.40*(ah-al);end=bl+.40*(bh-bl)
        if end<=begin:continue
        t=np.clip((v-begin)/(end-begin),0.,1.);s=smooth(t)
        center=ax+wrap(bx-ax)*s
        # Only the wales common to both released lobes traverse the span.
        # A broad aperture fans into that bundle over a few actual courses.
        # Interpolating the full aperture width made a conspicuous triangle.
        middle=min(aw,bw)+.16*(np.sqrt(aw*bw)-min(aw,bw))
        def shoulder(half,d):
            width=min(half,1.08*middle+.18)
            reach=2.3+.34*min(half,7.)
            return (width-middle)*np.exp(-.5*(np.maximum(0.,d)/reach)**2)
        width=middle+shoulder(aw,v-ah)+shoulder(bw,bl-v)+.28
        front=smooth((v-begin)/max(.18*(ah-al),.7))*smooth((end-v)/max(.18*(bh-bl),.7))
        lay(center,width,front,np.full(len(v),.50+.105*run))
    for i in active:
        lo,hi,xlo,xhi,half,n=shapes[i]
        for side,bound,x in ((-1,lo,xlo),(1,hi,xhi)):
            if (side<0 and i in joined_down) or (side>0 and i in joined_up):continue
            start=bound-side*.36*(hi-lo)
            length=(10.+1.32*min(n,16))*(.70+.70*run)
            hits=crossings(chart,edges,(x,bound),(x,bound+side*length))
            encountered=hits[hits*length>2.2]
            if len(encountered):length=min(length,encountered.min()*length+abs(bound-start))
            dist=side*(v-start);t=np.clip(dist/length,0.,1.)
            edge=smooth(dist/max(.16*(hi-lo),.7))
            radius=(.94*half+.55)*(1.-t)**.82+.28
            weight=edge*smooth((1.-t)/.22)
            lay(x,radius,weight,(.52+.08*run)*(1.-smooth(t))**.58)
    return layers


def mapped_fabric(points,knit,edges,run,stock):
    from .yarn_chart import stock_offset
    fields=np.zeros((len(points),9),'f')
    if len(edges)<2 or run<=0.:return fields
    # The damage is local to the released wales. Skip the back of the leg
    # and the foot when finding its chart on the posed undamaged stocking.
    seed=np.concatenate([knit[ring] for ring,_ in edges])
    near=(knit[:,1]>=seed[:,1].min()-160.)&(knit[:,1]<=seed[:,1].max()+160.)
    eligible=np.zeros(len(points),bool)
    for ring,_ in edges:
        center=knit[ring[0],0];u=wrap(knit[ring,0]-center)
        across=wrap(knit[:,0]-center)
        eligible|=(across>=u.min()-22.)&(across<=u.max()+22.)
    ids=np.flatnonzero(near&eligible)
    local=[(np.searchsorted(ids,ring),owners) for ring,owners in edges]
    shift,valid=stock_offset(points[ids],knit[ids],stock)
    chart=knit[ids].copy();chart[valid]+=shift[valid]
    fields[ids]=fabric(chart,knit[ids],local,run)
    return fields

def limit_span(material):
    """Resolve the width of the surviving yarn per shading sample on stretched triangles."""
    from .sheer_optics import math_socket
    tree=material.node_tree
    if tree.nodes.get(SPAN):return
    shader=tree.nodes.get('Principled BSDF')
    if shader is None or not shader.inputs['Alpha'].is_linked:return
    mix=shader.inputs['Alpha'].links[0].from_node
    if mix.type!='MIX_RGB' or not mix.inputs[0].is_linked:return
    old=mix.inputs[0].links[0].from_socket
    attr=tree.nodes.new('ShaderNodeAttribute');attr.name=SPAN;attr.attribute_name=SPAN
    parts=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(attr.outputs['Vector'],parts.inputs[0])
    x,width,ends=parts.outputs[:3]
    def op(kind,*inputs):return math_socket(tree,kind,*inputs)
    distance=op('DIVIDE',op('ABSOLUTE',x),op('MAXIMUM',width,.25))
    inside=op('MINIMUM',1.,op('MAXIMUM',0.,op('MULTIPLY_ADD',distance,-2.9,3.25)))
    inside=op('MULTIPLY',op('MULTIPLY',inside,inside),op('MULTIPLY_ADD',inside,-2.,3.))
    tree.links.new(op('MINIMUM',old,op('MULTIPLY',inside,ends)),mix.inputs[0])


def bind(material,scale):
    from .sheer_optics import FINISH,math_socket
    tree=material.node_tree
    if tree.nodes.get(BAND):
        limit_span(material);return
    shader=tree.nodes.get('Principled BSDF');ground=tree.nodes.get(FINISH)
    if shader is None or ground is None or not shader.inputs['Alpha'].is_linked:return
    old=shader.inputs['Alpha'].links[0].from_socket
    def field(name):
        attr=tree.nodes.new('ShaderNodeAttribute');attr.name=name;attr.attribute_name=name
        split=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(attr.outputs['Vector'],split.inputs[0])
        return split.outputs[:3]
    mask,loss,roll=field(BAND);du,dv,spent=field(COORD)
    knit=tree.nodes.new('ShaderNodeUVMap');knit.uv_map='Knit'
    split=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(knit.outputs['UV'],split.inputs[0])
    gauge=tree.nodes.new('ShaderNodeValue');gauge.name='Weft drawn yarn repeat';gauge.outputs[0].default_value=1./max(.01,scale)
    def op(kind,*args):return math_socket(tree,kind,*args)
    def wave(a,h):
        return op('POWER',op('MULTIPLY_ADD',op('COSINE',a),.5,.5),h)
    # The two families use the dye, optical thickness and normal of the
    # actual garment. Both remain attached to the material under a moving hand.
    x=op('MULTIPLY',op('ADD',split.outputs[0],du),gauge.outputs[0])
    y=op('MULTIPLY',op('ADD',split.outputs[1],dv),gauge.outputs[0])
    along=op('MULTIPLY_ADD',op('SINE',op('MULTIPLY',y,.26)),.065,x)
    fine=wave(op('MULTIPLY',along,12.56637),5.)
    ply=wave(op('MULTIPLY',along,4.712389),9.)
    across=wave(op('MULTIPLY_ADD',y,5.18,op('MULTIPLY',op('SINE',x),.68)),7.)
    stock=op('MULTIPLY_ADD',fine,.37,op('MULTIPLY_ADD',ply,.18,op('MULTIPLY_ADD',across,.25,.065)))
    depth=op('ADD',op('SUBTRACT',1.,op('MULTIPLY',loss,op('SUBTRACT',1.,stock))),op('MULTIPLY',roll,.31))
    run=next((n for n in tree.nodes if n.label=='Surviving wale and course yarn'),None)
    holes=1.
    if run:
        holes=op('MINIMUM',1.,op('MAXIMUM',.015,op('MULTIPLY_ADD',run.outputs['Alpha'],1.85,-.39)))
        depth=op('MULTIPLY',depth,holes)
    remaining=op('MAXIMUM',.0001,op('SUBTRACT',1.,ground.outputs['Coverage']))
    target=op('SUBTRACT',1.,op('POWER',remaining,depth))
    mix=tree.nodes.new('ShaderNodeMixRGB');tree.links.new(mask,mix.inputs[0])
    settled=tree.nodes.new('ShaderNodeMixRGB');tree.links.new(spent,settled.inputs[0])
    tree.links.new(old,settled.inputs[1]);tree.links.new(op('SUBTRACT',1.,op('POWER',remaining,holes)),settled.inputs[2])
    tree.links.new(settled.outputs[0],mix.inputs[1]);tree.links.new(target,mix.inputs[2]);tree.links.new(mix.outputs[0],shader.inputs['Alpha'])
    limit_span(material)
