"""The relaxed yarn repeat on a torn neutral surface; Knit remains the stitch ledger."""
import numpy as np
from mathutils.bvhtree import BVHTree
from .binding import coordinates
from .contact_index import ContactIndex
from .open_seam import chart_offset
from . import drawn_courses

FIELD='Weft relaxed courses'
STAMP='weft_bank_finish_events'
FIELDS=(FIELD,'Weft seated direction','Weft seated position','Weft neutral shading normal',
        drawn_courses.BAND,drawn_courses.COORD,drawn_courses.SPAN)


def reference(source,dg):
    ev=source.evaluated_get(dg);mesh=ev.data
    layer=mesh.uv_layers.get('Knit')
    if not len(mesh.polygons):return
    mesh.calc_loop_triangles();n=len(mesh.loop_triangles)
    faces=np.empty((n,3),'i');mesh.loop_triangles.foreach_get('vertices',faces.ravel())
    mat=np.asarray(ev.matrix_world);xyz=coordinates(mesh)@mat[:3,:3].T+mat[:3,3]
    if layer:
        corners=np.empty_like(faces);mesh.loop_triangles.foreach_get('loops',corners.ravel())
        uv=np.empty((len(mesh.loops),2),'f');layer.uv.foreach_get('vector',uv.ravel())
        corner_uv=uv[corners].astype(float)
    else:
        from .sock_length import CHART
        from .fitted_chart import fitted_chart
        from .live_membrane import knit_chart
        from .vroid_foot import foot_chart,is_vroid_foot
        same=len(source.data.vertices)==len(xyz)
        material=coordinates(source.data) if same else xyz
        authored=source.data.attributes.get(CHART)
        if same and authored is not None:
            values=np.empty((len(material),3),'f');authored.data.foreach_get('vector',values.ravel())
            chart=values[:,:2].astype(float)
        else:
            chart=fitted_chart(source,material) if source.get('weft_candidate_ankle') is not None else knit_chart(material)
        if same and is_vroid_foot(source,material):chart=foot_chart(material,chart)
        corner_uv=chart[faces]*[144.,1.]
    return xyz,faces,corner_uv,ContactIndex.create(xyz,faces)


def neutral_normals(points,faces):
    # Blender smooths a shared fan by corner angle. Equal vertex weights
    # would let the densest triangles on a torn bank steer its normal.
    triangle=points[faces]
    edges=np.roll(triangle,-1,axis=1)-triangle
    edges/=np.maximum(np.linalg.norm(edges,axis=2,keepdims=True),1.e-16)
    face=np.cross(edges[:,0],edges[:,1])
    face/=np.maximum(np.linalg.norm(face,axis=1,keepdims=True),1.e-16)
    angles=np.arccos(np.clip(-(edges*np.roll(edges,1,axis=1)).sum(2),-1.,1.))
    normal=np.zeros_like(points)
    corners=faces.ravel()
    for axis in range(3):
        normal[:,axis]=np.bincount(corners,(angles*face[:,axis,None]).ravel(),minlength=len(points))
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1.e-16)
    return normal


def stock_offset(points,knit,stock_mesh,*,shading=False):
    xyz,faces,uv,index=stock_mesh
    if index is None:face=np.full(len(points),-1,'i');p=np.zeros((len(points),3))
    else:
        face,close=index.nearest(points);p=close[:,:3].copy()
    if (face<0).any():
        tree=BVHTree.FromPolygons(xyz.tolist(),faces.tolist(),all_triangles=True)
        for i in np.flatnonzero(face<0):
            q,_,f,_=tree.find_nearest(points[i])
            if f is not None:face[i]=f;p[i]=q
    tri=xyz[faces[face]];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];d=p-tri[:,0]
    axis=np.cross(a,b);area=(axis*axis).sum(1)
    u=(np.cross(d,b)*axis).sum(1)/np.maximum(area,1.e-24)
    v=(np.cross(a,d)*axis).sum(1)/np.maximum(area,1.e-24)
    stock=uv[face]
    target=stock[:,0]+u[:,None]*chart_offset(stock[:,1],stock[:,0])+v[:,None]*chart_offset(stock[:,2],stock[:,0])
    shift=chart_offset(target,knit);dist=np.linalg.norm(points-p,axis=1)
    valid=(face>=0)&(area>1.e-24)&(dist<.020)
    if shading:
        stock_normal=neutral_normals(xyz,faces)[faces[face]]
        normal=(1.-u-v)[:,None]*stock_normal[:,0]+u[:,None]*stock_normal[:,1]+v[:,None]*stock_normal[:,2]
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1.e-16)
        return shift,valid,normal
    return shift,valid


def fit(rest,knit,stock_mesh,region,display_faces):
    ids,weight,feather,support=region
    take=weight>.0001
    ids=ids[take];weight=weight[take];feather=feather[take];support=support[take]
    shift,valid=stock_offset(rest[ids],knit[ids],stock_mesh)
    data=np.zeros((len(knit),3*len(FIELDS)),'f')
    data[ids[valid],:2]=weight[valid,None]*shift[valid]
    data[ids[valid],2]=feather[valid]
    data[ids[valid],3:6]=support[valid]
    data[ids[valid],6:9]=rest[ids[valid]]
    data[:,9:12]=neutral_normals(rest,display_faces)
    return data



def read(mesh):
    attr=mesh.attributes.get(FIELD)
    if attr is None:return None
    result=np.zeros((len(mesh.vertices),3*len(FIELDS)),'f')
    for i,name in enumerate(FIELDS):
        attr=mesh.attributes.get(name)
        if attr is not None:
            values=np.empty((len(mesh.vertices),3),'f')
            attr.data.foreach_get('vector',values.ravel());result[:,3*i:3*i+3]=values
    return result


def check_domain(sim,mesh):
    values=getattr(sim,'bank_weave',None)
    if values is None:return
    if len(values)==len(sim.rest) and getattr(sim,'bank_rest_owner',None) is sim.rest:return
    # A full material-state replacement has no parent map into the previous
    # weave. Keep the ledger and rebuild the appearance from this sheet on release.
    sim.bank_weave=None;sim.bank_rest_owner=None;sim.bank_style_count=-1
    sim.drawn_style_count=-1;sim.rounded_joins=set()
    parts=getattr(sim,'sections',None)
    current=[mesh]
    if parts is not None:
        current.extend(part.obj.data for part in parts.sections.values())
    for item in current:
        for name in FIELDS:
            attr=item.attributes.get(name)
            if attr is not None:item.attributes.remove(attr)
        if STAMP in item:del item[STAMP]
        for name in (drawn_courses.STAMP,drawn_courses.JOINS):
            if name in item:del item[name]


def write(mesh,values,ids=None):
    if values is None:return
    count=len(mesh.vertices)
    if ids is not None:
        # New vertices have the ordinary knit until their region is evaluated.
        local=np.zeros((count,values.shape[1]),'f')
        known=ids<len(values);local[known]=values[ids[known]]
        values=local
    elif count!=len(values):
        local=np.zeros((count,values.shape[1]),'f');n=min(count,len(values));local[:n]=values[:n];values=local
    for i,name in enumerate(FIELDS):
        attr=mesh.attributes.get(name) or mesh.attributes.new(name,'FLOAT_VECTOR','POINT')
        attr.data.foreach_set('vector',values[:,3*i:3*i+3].ravel())


def bind(material):
    tree=material.node_tree
    if tree.nodes.get(FIELD):
        update_held_finish(tree)
        local_bank_highlight(material)
        return
    shader=tree.nodes.get('Principled BSDF')
    optics=tree.nodes.get('Weft • Sheer nylon optics')
    attr=tree.nodes.new('ShaderNodeAttribute');attr.name=FIELD;attr.attribute_name=FIELD
    split=tree.nodes.new('ShaderNodeSeparateXYZ');tree.links.new(attr.outputs['Vector'],split.inputs[0])
    shift=tree.nodes.new('ShaderNodeCombineXYZ')
    for j in (0,1):tree.links.new(split.outputs[j],shift.inputs[j])
    knit=tree.nodes.new('ShaderNodeUVMap');knit.uv_map='Knit'
    visual=tree.nodes.new('ShaderNodeVectorMath');visual.operation='ADD'
    tree.links.new(knit.outputs['UV'],visual.inputs[0]);tree.links.new(shift.outputs[0],visual.inputs[1])
    for node in list(tree.nodes):
        if node.type=='TEX_COORD':
            for wire in list(node.outputs['UV'].links):
                if wire.to_node.type=='SEPXYZ':tree.links.new(visual.outputs[0],wire.to_socket)
        elif node.label=='Open pore coverage and approximate top relief':
            tree.links.new(visual.outputs[0],node.inputs['Vector'])
        elif node.label=='Wider palette · Knit':
            for wire in list(node.outputs['UV'].links):tree.links.new(visual.outputs[0],wire.to_socket)
    if shader is None or optics is None or not shader.inputs['Alpha'].is_linked:return
    # A few tight rows remain at the cut. Farther out the same nylon resumes
    # its regular pore coverage, with a gradual return to the existing run atlas.
    alpha=shader.inputs['Alpha'];old=alpha.links[0].from_socket
    mix=tree.nodes.new('ShaderNodeMixRGB');mix.blend_type='MIX'
    tree.links.new(split.outputs['Z'],mix.inputs[0])
    tree.links.new(old,mix.inputs[1]);tree.links.new(optics.outputs['Coverage'],mix.inputs[2])
    tree.links.new(mix.outputs[0],alpha)

    # The seated weight is also used to locate the dry cloth lifted by the hand.
    normal_input=shader.inputs['Normal']
    if not normal_input.is_linked:return
    live=normal_input.links[0].from_socket
    coords=tree.nodes.new('ShaderNodeTexCoord')
    def field(name):
        socket=tree.nodes.new('ShaderNodeAttribute');socket.attribute_name=name
        return socket.outputs['Vector']
    def vector(kind,a,b=None):
        node=tree.nodes.new('ShaderNodeVectorMath');node.operation=kind
        tree.links.new(a,node.inputs[0])
        if b is not None:tree.links.new(b,node.inputs[1])
        return node.outputs['Value' if kind in ('LENGTH','DISTANCE') else 'Vector']
    from .sheer_optics import math_socket
    support=field(FIELDS[1])
    travel=vector('DISTANCE',coords.outputs['Object'],field(FIELDS[2]))
    fade=math_socket(tree,'MINIMUM',1.,math_socket(tree,'MAXIMUM',0.,math_socket(tree,'MULTIPLY_ADD',travel,-190.,1.24)))
    weight=math_socket(tree,'MULTIPLY',vector('LENGTH',support),fade)
    mix=tree.nodes.new('ShaderNodeMix');mix.name='Weft seated highlight';mix.data_type='VECTOR'
    tree.links.new(weight,mix.inputs[0])
    tree.links.new(live,mix.inputs[4])
    normal=tree.nodes.new('ShaderNodeVectorTransform')
    normal.vector_type='NORMAL';normal.convert_from='OBJECT';normal.convert_to='WORLD'
    tree.links.new(vector('NORMALIZE',support),normal.inputs[0]);tree.links.new(normal.outputs[0],mix.inputs[5])
    tree.links.new(vector('NORMALIZE',mix.outputs[1]),normal_input)
    update_held_finish(tree)
    local_bank_highlight(material)


def update_held_finish(tree):
    mix=tree.nodes.get('Weft seated highlight')
    if mix is None or tree.nodes.get('Weft carried yarn normal'):return
    shader=tree.nodes.get('Principled BSDF')
    if shader is None:return
    link=tree.links.new
    def vector(operation,a,b=None):
        node=tree.nodes.new('ShaderNodeVectorMath');node.operation=operation
        link(a,node.inputs[0])
        if b is not None:link(b,node.inputs['Scale'] if operation=='SCALE' else node.inputs[1])
        return node.outputs['Value' if operation in ('DOT_PRODUCT','LENGTH') else 'Vector']
    from .sheer_optics import math_socket
    def op(kind,*values):return math_socket(tree,kind,*values)
    direction=next(n for n in tree.nodes if n.type=='ATTRIBUTE' and n.attribute_name==FIELDS[1]).outputs['Vector']
    covered=vector('LENGTH',direction)
    seated_weight=mix.inputs[0].links[0].from_socket
    movement=op('MAXIMUM',0.,op('SUBTRACT',covered,seated_weight))
    # Rotate the seated yarn by the actual change of surface orientation.
    # Translation by itself does not roll a patch of nylon onto a different
    # specular lobe. Use the live normals on unprocessed cloth.
    zero=tree.nodes.new('ShaderNodeAttribute');zero.name='Weft carried yarn normal';zero.attribute_name=FIELDS[3]
    basis=tree.nodes.new('ShaderNodeVectorTransform');basis.vector_type='NORMAL'
    basis.convert_from='OBJECT';basis.convert_to='WORLD'
    link(zero.outputs['Vector'],basis.inputs[0]);n0=vector('NORMALIZE',basis.outputs[0])
    geom=tree.nodes.new('ShaderNodeNewGeometry');n1=geom.outputs['Normal']
    base=mix.inputs[5].links[0].from_socket
    k=vector('CROSS_PRODUCT',n0,n1);first=vector('CROSS_PRODUCT',k,base)
    cos=vector('DOT_PRODUCT',n0,n1)
    second=vector('SCALE',vector('CROSS_PRODUCT',k,first),
                  op('DIVIDE',1.,op('MAXIMUM',.005,op('ADD',1.,cos))))
    carried=vector('NORMALIZE',vector('ADD',base,vector('ADD',first,second)))
    # Keep the installed resting shader, including its original shoulder
    # blend, completely intact at zero travel. The raw geometry normal in
    # Blender can differ from a computed reference even on a seated fan.
    final=shader.inputs['Normal']
    seated=final.links[0].from_socket
    raised=tree.nodes.new('ShaderNodeMix');raised.data_type='VECTOR'
    link(movement,raised.inputs[0]);link(seated,raised.inputs[4]);link(carried,raised.inputs[5])
    link(vector('NORMALIZE',raised.outputs[1]),final)
    # Dry filaments scatter the highlight over the raised bank. Retain
    # reflection on the yarn and avoid an unbroken white film over its pores.
    for name,fraction in (('Specular IOR Level',.68),('Anisotropic',.55),('Sheen Weight',.62)):
        socket=shader.inputs.get(name)
        if socket is None:continue
        value=socket.links[0].from_socket if socket.is_linked else socket.default_value
        link(op('MULTIPLY',value,op('MULTIPLY_ADD',movement,-fraction,1.)),socket)
    rough=shader.inputs['Roughness']
    value=rough.links[0].from_socket if rough.is_linked else rough.default_value
    link(op('MINIMUM',.90,op('MULTIPLY_ADD',movement,.15,value)),rough)


def local_bank_highlight(material):
    """Use the supported frame only where a dry bank actually bends."""
    if not material or not material.is_editable or material.get('weft_wet_shader'):
        return
    tree = material.node_tree
    name = 'Weft local bank normal'
    if tree is None or tree.nodes.get(name) or not tree.nodes.get('Weft carried yarn normal'):
        return
    seated = tree.nodes.get('Weft seated highlight')
    shader = tree.nodes.get('Principled BSDF')
    if seated is None or shader is None or not seated.inputs[4].is_linked:
        return
    final = shader.inputs['Normal']
    if not final.is_linked:
        return
    yarn = seated.inputs[4].links[0].from_socket
    supported = final.links[0].from_socket
    link = tree.links.new

    def vector(kind, a, b=None):
        node = tree.nodes.new('ShaderNodeVectorMath')
        node.operation = kind
        link(a, node.inputs[0])
        if b is not None:
            link(b, node.inputs[1])
        scalar = kind in ('LENGTH','DOT_PRODUCT','DISTANCE')
        return node.outputs['Value' if scalar else 'Vector']

    def attribute(field):
        return next(n.outputs['Vector'] for n in tree.nodes
                    if n.type == 'ATTRIBUTE' and n.attribute_name == field)

    from .sheer_optics import math_socket
    def calc(op, *args):
        return math_socket(tree, op, *args)

    def bend(a, b):
        dot = vector('DOT_PRODUCT', a, b)
        return calc('MINIMUM', 1., calc('MAXIMUM', 0., calc('MULTIPLY_ADD', dot, -7., 6.90)))

    # Older multi-hole materials already end in a yarn/support mix. Reuse
    # its two branches and replace the global "all banks joined" override.
    mix = None
    last = supported.node
    if tree.nodes.get('Weft unjoined banks') and last.type == 'VECT_MATH' and last.operation == 'NORMALIZE' and last.inputs[0].is_linked:
        old = last.inputs[0].links[0].from_node
        if old.type == 'MIX' and old.inputs[4].is_linked and old.inputs[4].links[0].from_socket == yarn:
            mix = old
    if mix is None:
        mix = tree.nodes.new('ShaderNodeMix')
        mix.data_type = 'VECTOR'
        link(yarn, mix.inputs[4])
        link(supported, mix.inputs[5])
        link(vector('NORMALIZE', mix.outputs[1]), final)
    mix.name = name

    support = attribute(FIELDS[1])
    neutral = vector('NORMALIZE', attribute(FIELDS[3]))
    rest_bend = bend(vector('NORMALIZE', support), neutral)
    to_world = tree.nodes.new('ShaderNodeVectorTransform')
    to_world.vector_type = 'NORMAL'
    to_world.convert_from = 'OBJECT'
    to_world.convert_to = 'WORLD'
    link(neutral, to_world.inputs[0])
    live_normal = tree.nodes.new('ShaderNodeNewGeometry').outputs['Normal']
    live_bend = bend(vector('NORMALIZE', to_world.outputs[0]), live_normal)
    coords = tree.nodes.new('ShaderNodeTexCoord').outputs['Object']
    travel = vector('DISTANCE', coords, attribute(FIELDS[2]))
    raised = calc('MINIMUM', 1., calc('MAXIMUM', 0., calc('MULTIPLY_ADD', travel, 240., -.12)))
    use = calc('MAXIMUM', rest_bend, calc('MULTIPLY', raised, live_bend))
    coverage = calc('MINIMUM', 1., calc('MULTIPLY', vector('LENGTH', support), 12.))
    link(calc('MULTIPLY', coverage, use), mix.inputs[0])
