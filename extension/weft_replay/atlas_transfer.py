"""Transfer native, filtered gauge samples into the resident material image."""
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

class AtlasTransfer:
    def __init__(self,image):
        self.image=image
        w,h=image.size
        self.staging=bpy.data.images.new('Weft • gauge transfer',w,h,alpha=True,float_buffer=image.is_float,is_data=True)
        self.staging.alpha_mode='CHANNEL_PACKED'
        self.frames=[];self.destination=None;self.dirty=False
        info=gpu.types.GPUShaderCreateInfo()
        info.vertex_in(0,'VEC2','position')
        info.push_constant('INT','level')
        info.sampler(0,'FLOAT_2D','source')
        info.fragment_out(0,'VEC4','result')
        info.vertex_source('void main() { gl_Position=vec4(position,0,1); }')
        info.fragment_source('void main() { result=texelFetch(source,ivec2(gl_FragCoord.xy),level); }')
        self.shader=gpu.shader.create_from_info(info)
        self.batch=batch_for_shader(self.shader,'TRIS',{'position':((-1,-1),(3,-1),(-1,3))})
        self.sizes=[(max(1,w>>k),max(1,h>>k)) for k in range(max(w,h).bit_length())]

    def write(self,pixels):
        self.staging.pixels.foreach_set(pixels.ravel())
        source=gpu.texture.from_image(self.staging)
        if not self.frames:
            self.destination=gpu.texture.from_image(self.image)
            self.frames=[gpu.types.GPUFrameBuffer(color_slots=({'texture':self.destination,'mip':k},))
                         for k in range(len(self.sizes))]
        blend=gpu.state.blend_get();depth=gpu.state.depth_test_get();mask=gpu.state.depth_mask_get()
        try:
            gpu.state.blend_set('NONE');gpu.state.depth_test_set('NONE');gpu.state.depth_mask_set(False)
            for k,(frame,(w,h)) in enumerate(zip(self.frames,self.sizes)):
                with frame.bind():
                    frame.viewport_set(0,0,w,h)
                    self.shader.bind();self.shader.uniform_sampler('source',source)
                    self.shader.uniform_int('level',k)
                    self.batch.draw(self.shader)
        finally:
            gpu.state.blend_set(blend);gpu.state.depth_test_set(depth);gpu.state.depth_mask_set(mask)
        self.dirty=True

    def flush(self,pixels):
        if self.dirty:
            self.frames.clear();self.destination=None
            self.image.pixels.foreach_set(pixels.ravel())
            self.dirty=False

    def close(self,pixels):
        self.flush(pixels);bpy.data.images.remove(self.staging)


def publish(sim,name,pixels):
    image=getattr(sim,name+'_image')
    key=name+'_transfer'
    transfer=getattr(sim,key,None)
    active=(getattr(sim,'sections',None) is not None and sim.grip is not None
            and not bpy.app.background and gpu.platform.backend_type_get()=='VULKAN')
    if active:
        if transfer is None:
            transfer=AtlasTransfer(image)
            setattr(sim,key,transfer)
        transfer.write(pixels)
    else:
        if transfer is not None:transfer.flush(pixels)
        image.pixels.foreach_set(pixels.ravel())


def flush(sim,close=False):
    for name in ('loop','run'):
        key=name+'_transfer';transfer=getattr(sim,key,None)
        if transfer is None:continue
        pixels=getattr(sim,name+'_pixels')
        if close:
            transfer.close(pixels);setattr(sim,key,None)
        else:transfer.flush(pixels)
