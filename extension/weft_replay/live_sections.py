"""Draw the dry membrane in longitudinal sections without losing its smooth boundary fans."""
import bpy
import numpy as np
from .dry_fabric import TOE_FINISH_ATTRIBUTE



def layout_groups(rest,base,banks,anchor):
    """Use material-space bands so static parts keep Blender's draw buffers."""
    axis=int(np.argmax(np.ptp(rest,axis=0)))
    extent=rest[:,axis][base]
    keys=np.floor((extent.mean(1)-rest[:,axis].min())/.055).astype('i')
    weld=np.arange(len(rest),dtype='i')
    for pair in banks:weld[pair[:,1]]=pair[:,0]
    joined=weld[base];groups={}
    low=extent.min(1);high=extent.max(1)
    for k in np.unique(keys):
        own=np.flatnonzero(keys==k)
        candidates=np.flatnonzero((keys!=k)&(low<=high[own].max())&(high>=low[own].min()))
        marked=np.zeros(len(rest),bool);marked[joined[own]]=True
        halo=candidates[marked[joined[candidates]].any(1)]
        rows=np.r_[own,halo].astype('i')
        # The owned welded vertices are marked already. Add the halo and all
        # unwelded corners. Dense material indices give the same sorted union
        # without sorting tens of thousands of repeated corners per band.
        marked[joined[halo]]=True
        marked[base[rows]]=True
        ids=np.flatnonzero(marked).astype('i')
        remap=np.full(len(rest),-1,dtype='i');remap[ids]=np.arange(len(ids),dtype='i')
        groups[int(k)]=(len(own),rows,ids,remap)
    return groups


def bank_face_rows(sim):
    # The render weld maps only the second vertex of a closed pair onto
    # the first. No other material corner can change while this layout lasts.
    switch=np.zeros(len(sim.rest),bool)
    for crack in sim.cracks:switch[crack['banks'][:,1]]=True
    return np.flatnonzero(switch[sim.faces].any(axis=1))


class Section:
    def __init__(self, owner, material, invisible, index):
        mesh=bpy.data.meshes.new('Weft • active courses')
        mesh.materials.append(material);mesh.materials.append(invisible)
        self.obj=bpy.data.objects.new(mesh.name,mesh)
        self.index=index
        self.obj['weft_courses']=index
        owner.users_collection[0].objects.link(self.obj)
        self.obj.hide_select=True
        self.world_faces=self.positions=None

    def rebuild(self,sim,faces,coords,toe):
        mesh=self.obj.data
        tri=self.remap[faces]
        if self.world_faces is not None and self.world_faces.shape==faces.shape:
            from .live_view import update_bank_edges
            if update_bank_edges(mesh,self,tri):
                self.world_faces=faces;self.published_triangles=tri
                self.update_positions(coords);mesh.update()
                return
        self.world_faces=faces
        self.faces=self.published_faces=object()
        self.published_triangles=tri;self.bank_edge_cache=None
        self.uv=sim.uv[self.ids]
        self.positions=np.ascontiguousarray(coords[self.ids])
        mesh.clear_geometry()
        mesh.vertices.add(len(self.ids))
        mesh.attributes['position'].data.foreach_set('vector',self.positions.ravel())
        mesh.loops.add(tri.size)
        mesh.attributes['.corner_vert'].data.foreach_set('value',tri.ravel())
        mesh.polygons.add(len(tri))
        mesh.polygons.foreach_set('loop_start',np.arange(0,tri.size,3,dtype='i'))
        mesh.polygons.foreach_set('use_smooth',np.ones(len(tri),bool))
        material_ids=np.zeros(len(tri),dtype='i');material_ids[self.owned:]=1
        mesh.polygons.foreach_set('material_index',material_ids)
        from .contact_index import mesh_edges
        from .live_view import seed_edge_cache
        seed_edge_cache(self,mesh_edges(mesh,tri,len(self.ids)))
        uv=sim.uv[faces].copy()
        uv[:,:,0]-=np.round(uv[:,:,0]-uv[:,:1,0]);uv[:,:,0]*=144.
        layer=mesh.uv_layers.get('Knit') or mesh.uv_layers.new(name='Knit')
        layer.uv.foreach_set('vector',uv.astype('f').ravel())
        if toe is not None:
            field=mesh.attributes.get(TOE_FINISH_ATTRIBUTE) or mesh.attributes.new(TOE_FINISH_ATTRIBUTE,'FLOAT','POINT')
            field.data.foreach_set('value',toe[self.ids])
        mesh.update()

    def update_positions(self,coords):
        local=coords[self.ids]
        if np.array_equal(local,self.positions):return
        self.positions=local
        self.obj.data.attributes['position'].data.foreach_set('vector',local.ravel())


class Sections:
    def __init__(self,owner,sim):
        self.owner=owner;self.sections={};self.source=None;self.drawn=None;self.show_parts=True
        self.spares={};self.lanes=None;self.staged=None;self.retired=[]
        self.invisible=bpy.data.materials.new('Weft • shading neighbours')
        self.invisible.surface_render_method='BLENDED'
        self.invisible.use_transparency_overlap=True
        self.invisible.use_nodes=True
        tree=self.invisible.node_tree
        shader=tree.nodes.new('ShaderNodeBsdfTransparent')
        tree.links.new(shader.outputs[0],tree.nodes.get('Material Output').inputs['Surface'])
        self.layout(sim,np.ascontiguousarray(sim.display_positions(),dtype='f'))
        owner.hide_set(True);owner.hide_render=True

    def layout(self,sim,coords):
        base=sim.faces
        tri=sim.display_faces()
        prepared_rows=None
        if self.staged is not None and self.staged[0][0] is not base:
            self.restore_staged()
        new_layout=self.source is not base
        if new_layout:
            for part in self.sections.values():part.bank_rows=None
            self.source=base
            banks=[c['banks'] for c in sim.cracks]
            prepared=getattr(sim,'draw_packet',None)
            if prepared is not None and prepared[0] is base:
                groups=prepared[1]
                if np.array_equal(tri,prepared[2]):prepared_rows=prepared[13]
            else:groups=layout_groups(sim.rest,base,banks,getattr(sim,'anchor',None))
            for k,part in self.sections.items():
                if k not in groups:
                    part.obj.hide_set(True);part.obj.hide_render=True
            for k,(owned,rows,ids,remap) in groups.items():
                part=self.sections.get(k)
                if part is None:
                    part=self.sections[k]=Section(self.owner,self.owner.data.materials[0],self.invisible,k)
                if part.obj.hide_get()==self.show_parts:part.obj.hide_set(not self.show_parts)
                if part.obj.hide_render:part.obj.hide_render=False
                if part.positions is not None and (owned!=part.owned or not np.array_equal(ids,part.ids)):
                    part.world_faces=None
                part.owned,part.rows,part.ids,part.remap=owned,rows,ids,remap
                if prepared is not None and prepared[0] is base and k in prepared[8]:
                    staged=self.staged is not None and k in self.staged[3]
                    spare=part.obj.data if staged else self.take(prepared,k)
                    # The material plan may finish between the last warning frame and failure.
                    # Reuse its exact corners, UVs and edge incidences even without a warm mesh.
                    matches=prepared_rows is not None or np.array_equal(prepared[2][rows],tri[rows])
                    local_tri,local_uv,local_edges,edge_cache=prepared[8][k]
                    cold_positions=None
                    if spare is None and matches:
                        toe=prepared[7];cold_positions=coords[ids]
                        spare=self.new_mesh((self.owner.data.materials[0],self.invisible),cold_positions,
                                            local_tri,local_uv,local_edges,None if toe is None else toe[ids],owned)
                    if spare is not None and matches:
                        if staged:
                            old=self.staged[3].pop(k)
                        else:
                            old=part.obj.data;part.obj.data=spare
                        if not old.users:self.retired.append(old)
                        part.world_faces=prepared_rows[k][0] if prepared_rows is not None else prepared[2][rows]
                        part.faces=part.published_faces=object();part.published_triangles=local_tri
                        part.uv=sim.uv[ids];part.bank_edge_cache=edge_cache
                        part.positions=cold_positions
                    elif staged:self.restore_staged()
                    elif spare is not None:bpy.data.meshes.remove(spare)
            self.active=tuple(self.sections[k] for k in groups)
            self.toe=None
            if hasattr(sim,'toe_finish'):
                direction,start,end=sim.toe_finish[:3],sim.toe_finish[3],sim.toe_finish[4]
                ramp=np.clip((sim.rest@direction-start)/(end-start),0,1)
                self.toe=(ramp*ramp*(3-2*ramp)).astype('f')
        for part in self.active:
            subset=None if new_layout else getattr(part,'bank_rows',None)
            if subset is not None and part.world_faces is not None and np.array_equal(
                    tri[part.rows[subset]],part.world_faces[subset]):
                part.update_positions(coords)
                continue
            faces=prepared_rows[part.index][0] if prepared_rows is not None else tri[part.rows]
            if part.world_faces is None or not np.array_equal(faces,part.world_faces):
                part.rebuild(sim,faces,coords,self.toe)
            else:part.update_positions(coords)
        if new_layout and getattr(sim,'bank_weave',None) is not None:
            from .yarn_chart import write
            for part in self.active:write(part.obj.data,sim.bank_weave,part.ids)
        if new_layout and sim.cracks:
            if prepared is not None and prepared[0] is base:
                sim.changing_bank_rows=(base,prepared[12])
                for part in self.active:part.bank_rows=prepared[13][part.index][1]
            else:
                rows=bank_face_rows(sim);sim.changing_bank_rows=(base,rows)
                changing=np.zeros(len(tri),bool);changing[rows]=True
                for part in self.active:part.bank_rows=np.flatnonzero(changing[part.rows])
        self.drawn=tri
        if self.staged is not None:self.restore_staged()

    def update(self,sim,coords,dirty=None):
        if self.staged is not None and self.source is sim.faces:
            if sim.warning is not None and self.staged[0][0] is not sim.faces:
                self.update_staged(coords)
                return
            self.restore_staged()
        if self.drawn is not sim.display_faces():self.layout(sim,coords)
        elif dirty is not None:
            cache=getattr(self,'active_grip',None)
            if cache is None or cache[0] is not self.active or cache[1] is not dirty:
                changing=tuple(part for part in self.active if np.any(part.remap[dirty]>=0))
                cache=self.active_grip=(self.active,dirty,changing)
            for part in cache[2]:part.update_positions(coords)
        else:
            for part in self.active:part.update_positions(coords)

    def prepare_warning(self,sim):
        """Upload each affected course band while its two banks still coincide."""
        lanes=getattr(sim,'warned_lanes',None)
        if not lanes or self.source is not sim.faces:return
        peak=next(iter(lanes[1].values()),None)
        if self.lanes is not peak:
            self.drop_spares();self.lanes=peak
        for future in list(lanes[1].values())[:1]:
            if future.cancelled() or not future.done():return
            plan,_,packet=future.result()
            if packet is None or packet[4] is None:continue
            base,groups,shown,uv,arrays,cache,rest,toe,tiles=packet[:9]
            key=id(base);ready=self.spares.setdefault(key,{})
            if self.staged is not None:return
            if -1 not in ready:
                ready[-1]=self.new_mesh(self.owner.data.materials,rest,shown,uv,arrays,toe,None)
            for k,(tri,part_uv,edges,part_cache) in tiles.items():
                if k in ready:continue
                owned,rows,ids,_=groups[k]
                ready[k]=self.new_mesh((self.owner.data.materials[0],self.invisible),rest[ids],tri,part_uv,
                                       edges,None if toe is None else toe[ids],owned)
            # Feed the actual material preview its new triangulation while the
            # interlocks are still closed. Both banks use the same interpolated
            # position until the solver releases that stitch. Picking and saving
            # continue to use the committed canonical cloth.
            if sim.warning is not None and self.show_parts and tiles and all(k in self.sections for k in tiles):
                originals={}
                for k in tiles:
                    part=self.sections[k]
                    originals[k]=part.obj.data
                    part.obj.data=ready.pop(k)
                if getattr(sim,'bank_weave',None) is not None:
                    from .yarn_chart import write
                    data=sim.bank_weave
                    parents,weights,first=plan[:3]
                    add=np.einsum('ij,ijk->ik',weights,data[parents]).astype('f')
                    extended=np.concatenate((data,add,add[first[1:-1]-len(data)]))
                    for k in originals:write(self.sections[k].obj.data,extended,packet[1][k][2])
                self.staged=(packet,plan,sim.faces,originals)
                self.update_staged(sim.display_positions().astype('f',copy=False))

    def update_staged(self,coords):
        packet,plan,_,originals=self.staged
        parents,weights,first=plan[:3]
        extra=np.einsum('ni,nij->nj',weights,coords[parents]).astype('f')
        extended=np.concatenate((coords,extra,extra[first[1:-1]-len(coords)]))
        for key,part in self.sections.items():
            if key in originals:
                part.obj.data.attributes['position'].data.foreach_set('vector',extended[packet[1][key][2]].ravel())
            elif part in self.active:
                part.update_positions(coords)

    def restore_staged(self):
        if self.staged is None:return
        _,_,_,originals=self.staged
        for key,old in originals.items():
            part=self.sections[key];preview=part.obj.data
            part.obj.data=old
            if not preview.users:bpy.data.meshes.remove(preview)
            part.positions=None
        self.staged=None

    def new_mesh(self,materials,coords,tri,uv,edges,toe,owned):
        from .contact_index import assign_edges
        mesh=bpy.data.meshes.new('Weft • prepared knit')
        for mat in materials:mesh.materials.append(mat)
        mesh.vertices.add(len(coords));mesh.vertices.foreach_set('co',coords.astype('f').ravel())
        mesh.loops.add(tri.size);mesh.attributes['.corner_vert'].data.foreach_set('value',tri.ravel())
        mesh.polygons.add(len(tri));mesh.polygons.foreach_set('loop_start',np.arange(0,tri.size,3,dtype='i'))
        if owned is not None:
            material_ids=np.zeros(len(tri),dtype='i');material_ids[owned:]=1
            mesh.polygons.foreach_set('material_index',material_ids)
        mesh.polygons.foreach_set('use_smooth',np.ones(len(tri),bool))
        assign_edges(mesh,edges)
        mesh.uv_layers.new(name='Knit').uv.foreach_set('vector',uv.ravel())
        if toe is not None:mesh.attributes.new(TOE_FINISH_ATTRIBUTE,'FLOAT','POINT').data.foreach_set('value',toe)
        mesh.update()
        return mesh

    def take(self,packet,key):
        return self.spares.get(id(packet[0]),{}).pop(key,None)

    def drop_spares(self):
        self.restore_staged()
        for pieces in self.spares.values():
            for mesh in pieces.values():
                if not mesh.users:bpy.data.meshes.remove(mesh)
        self.spares.clear();self.lanes=None

    def shading(self,mode):
        show=mode in {'MATERIAL','RENDERED'}
        if show==self.show_parts:return
        self.show_parts=show;self.owner.hide_set(show)
        for part in self.active:part.obj.hide_set(not show)

    def free_retired(self):
        unused=[mesh for mesh in self.retired if not mesh.users]
        if unused:bpy.data.batch_remove(ids=unused)
        self.retired.clear()

    def close(self):
        self.drop_spares()
        self.free_retired()
        self.owner.hide_set(False);self.owner.hide_render=False
        for part in self.sections.values():
            mesh=part.obj.data
            bpy.data.objects.remove(part.obj,do_unlink=True);bpy.data.meshes.remove(mesh)
        bpy.data.materials.remove(self.invisible)
