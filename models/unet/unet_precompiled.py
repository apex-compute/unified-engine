"""Fixed-shape, single-upload, single-HALT U-Net deployment.

Reuse the proven packed-map CONV/MAXPOOL and AXI-safe DMA emitters. Transpose
phases scatter straight into the skip-concat destination. No live host tensors
or layer dispatch are involved between the single kick and final readback.
"""
import hashlib
import math
import sys
import time
from pathlib import Path

import torch
from unet_common import ROOT, udc

sys.path.insert(0, str(ROOT/'models/yolov5s'))
import yolov5_precompiled as shared

FORMAT = 'andromeda.unet.whole-graph-v2'


def graph(layers, height, width):
    if min(height, width) < 16 or height % 16 or width % 16:
        raise ValueError('whole-graph resolution must be positive multiples of 16')
    shapes = {'input': (3, height, width)}
    ops = []
    def conv(name, source):
        shape = (layers[name]['codes'].shape[0], *shapes[source][1:])
        shapes[name] = tuple(shape)
        ops.append(dict(name=name, op='conv', inputs=[source], output=name,
                        stride=1, pad=layers[name]['pad'], dilation=1))
        return name
    def double(prefix, source):
        return conv(prefix+'.3', conv(prefix+'.0', source))
    x = double('inc', 'input')
    skips = [x]
    for level in range(1, 5):
        name = f'down{level}.pool'
        c,h,w = shapes[x]
        shapes[name] = (c,h//2,w//2)
        ops.append(dict(name=name, op='maxpool', inputs=[x], output=name,
                        kernel=2, stride=2, pad=0))
        x = double(f'down{level}.maxpool_conv.1', name)
        if level < 4:
            skips.append(x)
    for level in range(1, 5):
        phases = [conv(f'up{level}.phase{p}', x) for p in range(4)]
        skip = skips.pop()
        name = f'up{level}.merge'
        c,h,w = shapes[skip]
        shapes[name] = (2*c,h,w)
        ops.append(dict(name=name, op='merge', inputs=[skip]+phases, output=name))
        x = double(f'up{level}.conv', name)
    conv('outc', x)
    return shapes, ops


def memory_plan(shapes, ops):
    layouts = {'input': shared._layout_for_conv(shapes['input'])}
    layouts['input'].address = shared.INPUT_BASE
    last = {name: i for i,op in enumerate(ops) for name in op['inputs']}
    free = []
    cursor = shared.TENSOR_BASE
    for i,op in enumerate(ops):
        name = op['output']
        layout = shared._layout_for_conv(shapes[name])
        size = shared._align_up(layout.size_bytes, 128)
        candidates = [(span[1],j) for j,span in enumerate(free) if span[1] >= size]
        if candidates:
            _,j = min(candidates)
            address, available = free.pop(j)
            if available > size:
                free.append((address+size, available-size))
        else:
            address = cursor
            cursor += size
        layout.address = address
        layouts[name] = layout
        for source in op['inputs']:
            if source != 'input' and last[source] == i:
                old = layouts[source]
                free.append((old.address, shared._align_up(old.size_bytes,128)))
        free.sort()
        merged = []
        for address,length in free:
            if merged and sum(merged[-1]) == address:
                merged[-1] = (merged[-1][0], merged[-1][1]+length)
            else:
                merged.append((address,length))
        free = merged
    scratch = shared._align_up(cursor,128)
    if scratch + (512 << 10) > shared.TENSOR_LIMIT:
        raise RuntimeError('U-Net live tensor arena exceeds 512 MiB')
    if layouts['input'].size_bytes > shared.INPUT_LIMIT-shared.INPUT_BASE:
        raise RuntimeError('U-Net input exceeds input arena')
    return layouts, scratch


def emit_conv(engine, plan, zero, scratch, relu):
    mode,a,b = udc._conv_fused_lalu(relu,False,False)
    src,dst = plan['source'],plan['destination']
    ct = (plan['convolution_c']+63)//64
    for chunk in plan['chunks']:
        for start,stop,th,tw,_,win_w in plan['groups']:
            blocks = plan['gather_chunks'] if plan['use_gather'] else plan['kernel_h']*plan['kernel_w']*ct
            engine.accelerator_memory_to_scale_sram(chunk['scale_address'],plan['oc_chunk']*blocks)
            engine.accelerator_memory_to_bias_sram(chunk['bias_address'],plan['oc_chunk'])
            for tile in plan['tiles'][start:stop]:
                shared._stage_conv_window(engine,src,tile,plan['operation']['pad'],zero)
                engine.start_queue_for_conv2d_operation(
                    act_sram_start_addr=0, output_sram_wb_addr=shared._WB_SRAM_ADDRESS,
                    weights_dram_addr=chunk['weight_address'],kernel_w=plan['kernel_w'],
                    kernel_h=plan['kernel_h'],ct=ct,oc_count=plan['oc_chunk'],
                    out_w=tw,out_h=th,w_pad=win_w,stride_s=1,data_type=udc.TYPE.IF8,
                    bias_enable=True,lalu_mode=mode,lalu_a=a,lalu_b=b,
                    dilation=1,gather=plan['use_gather'],c_in=plan['convolution_c'])
                if plan['oc_chunk'] != 32:
                    shared._scatter_conv_tile(engine,dst,tile,chunk['oc0'],plan['oc_chunk'])
                else:
                    # Dense OC32 pixels occupy half a 128-byte SRAM line.
                    # A multi-chunk 64-byte strided write was incorrect on
                    # AXI512 hardware. Spill once, then stage each 64-byte
                    # pixel at SRAM line 0 and use a contiguous write instead.
                    row_bytes = tw*64
                    engine.sram_to_accelerator_memory(shared._WB_SRAM_ADDRESS,scratch,0,
                                                      memcpy_length_bytes=th*row_bytes)
                    oy,ox = tile[:2]
                    pixel_bytes = dst.physical_channels*2
                    for row in range(th):
                        for col in range(tw):
                            engine.accelerator_memory_to_sram(scratch+row*row_bytes+col*64,0,0,
                                                             memcpy_length_bytes=64)
                            engine.sram_to_accelerator_memory(0,
                                dst.address+((oy+row)*dst.width+ox+col)*pixel_bytes+chunk['oc0']*2,
                                0,memcpy_length_bytes=64)


def emit_merge(engine, sources, dst):
    skip,*phases = sources
    pixel = skip.physical_channels*2
    out_pixel = dst.physical_channels*2
    # Skip channels occupy the first half; transpose phases the second half.
    take_pixels = max(1,udc.URAM_NEAR_FULL_SIZE//pixel)
    for start in range(0,skip.height*skip.width,take_pixels):
        n = min(take_pixels,skip.height*skip.width-start)
        engine.accelerator_memory_to_sram(skip.address+start*pixel,0,0,memcpy_length_bytes=n*pixel)
        shared._copy_contiguous_or_strided_write(engine,sram=0,
            destination=dst.address+start*out_pixel,total=n*pixel,chunk=pixel,jump=out_pixel)
    for phase,source in enumerate(phases):
        for y in range(source.height):
            for x in range(0,source.width,take_pixels):
                n = min(take_pixels,source.width-x)
                engine.accelerator_memory_to_sram(source.address+(y*source.width+x)*pixel,
                                                 0,0,memcpy_length_bytes=n*pixel)
                address = dst.address+((2*y+phase//2)*dst.width+2*x+phase%2)*out_pixel+pixel
                shared._copy_contiguous_or_strided_write(engine,sram=0,destination=address,
                    total=n*pixel,chunk=pixel,jump=2*out_pixel)


def compile_hardware(layers, height, width):
    previous = udc.UE_AXI_DATA_WIDTH_BITS
    udc.UE_AXI_DATA_WIDTH_BITS = 256
    try:
        return _compile(layers,height,width)
    finally:
        udc.UE_AXI_DATA_WIDTH_BITS = previous


def _compile(layers,height,width):
    shapes,ops = graph(layers,height,width)
    layouts,scratch = memory_plan(shapes,ops)
    image = shared._ImageBuilder(shared.MODEL_BASE,shared.MODEL_LIMIT)
    zero = image.allocate(torch.zeros(shared._ACT_TEMPLATE_BYTES//2,dtype=torch.bfloat16))
    neg = image.allocate(torch.full((shared._ACT_TEMPLATE_BYTES//2,),float('-inf'),dtype=torch.bfloat16))
    plans = {}
    for op in ops:
        if op['op'] != 'conv':
            continue
        layer = layers[op['name']]
        codes = layer['codes']
        oc,c,kh,kw = codes.shape
        chunks = (kh*kw*c+63)//64
        gather = c <= 255 and chunks <= 4 and max(kh*kw*((c+3)//4),oc*chunks) < oc*kh*kw*((c+63)//64)
        blocks = chunks if gather else kh*kw*((c+63)//64)
        weight = dict(precision='if8',codes_packed=codes.contiguous().view(torch.uint8).flatten(),
                      codes_shape=list(codes.shape),layout='gather' if gather else 'channel',
                      block_scales=-layer['scales'][:,None].expand(oc,blocks).contiguous(),bias=layer['bias'])
        # A deep 1024-channel kernel alone exceeds YOLO's 4 MiB per-layer
        # stream budget. Permit four spatial copies there so the planner is
        # not forced into single-pixel launches; retain the 4 MiB floor.
        budget = max(4 << 20, oc*blocks*64*4)
        plans[op['name']] = shared._prepare_conv_plan(op,weight,layouts[op['inputs'][0]],
            layouts[op['output']],image,allow_half_vector_output=True,
            weight_stream_budget_bytes=budget)
    base = image.align(64)
    engine = shared._WholeGraphEngine(base)
    engine.start_capture()
    entries = []
    for op in ops:
        start = engine.capture_count
        sources = [layouts[name] for name in op['inputs']]
        dst = layouts[op['output']]
        if op['op'] == 'conv':
            emit_conv(engine,plans[op['name']],zero,scratch,layers[op['name']]['relu'])
        elif op['op'] == 'maxpool':
            shared._emit_maxpool(engine,op,sources[0],dst,neg)
        else:
            emit_merge(engine,sources,dst)
        entries.append(dict(name=op['name'],start=start,stop=engine.capture_count))
    engine.generate_instruction_halt()
    engine.stop_capture()
    program = b''.join(inst.get_bytes() for inst in engine.capture_buffer)
    image.write(base,program)
    hardware = dict(resolution=[width,height],model_base=shared.MODEL_BASE,
        model_image=shared._bytes_tensor(image.data),program_address=base,
        program_offset=base-shared.MODEL_BASE,program_size=len(program),
        model_sha256=hashlib.sha256(image.data).hexdigest(),
        program_sha256=hashlib.sha256(program).hexdigest(),
        tensors={name:layout.manifest() for name,layout in layouts.items()},
        scratch=scratch,operations=entries)
    validate(layers,hardware)
    return hardware


def validate(layers, hardware):
    width,height = hardware['resolution']
    shapes,ops = graph(layers,height,width)
    layouts,scratch = memory_plan(shapes,ops)
    if hardware['tensors'] != {name:m.manifest() for name,m in layouts.items()} or hardware['scratch'] != scratch:
        raise RuntimeError('U-Net tensor memory plan mismatch')
    image = hardware['model_image']
    if image.dtype != torch.uint8 or image.ndim != 1 or not 0 < image.numel() <= shared.MODEL_LIMIT-shared.MODEL_BASE:
        raise RuntimeError('invalid U-Net deployment image')
    raw = shared._tensor_bytes(image)
    if hardware['model_base'] != shared.MODEL_BASE or hashlib.sha256(raw).hexdigest() != hardware['model_sha256']:
        raise RuntimeError('U-Net deployment image checksum/base mismatch')
    offset,size = hardware['program_offset'],hardware['program_size']
    if offset < 0 or offset%64 or size <= 0 or size%64 or offset+size != len(raw) or hardware['program_address'] != shared.MODEL_BASE+offset:
        raise RuntimeError('invalid U-Net program bounds')
    program = raw[offset:offset+size]
    if hashlib.sha256(program).hexdigest() != hardware['program_sha256']:
        raise RuntimeError('U-Net program checksum mismatch')
    types = shared._instruction_types(program)
    if types.count(udc.INSTRUCTION_HALT) != 1 or udc.INSTRUCTION_SWI in types:
        raise RuntimeError('U-Net requires exactly one HALT and no SWI')
    halt = types.index(udc.INSTRUCTION_HALT)
    if any(t != udc.INSTRUCTION_NOP for t in types[halt+1:]):
        raise RuntimeError('non-terminal U-Net HALT')
    cursor = 0
    if len(hardware['operations']) != len(ops):
        raise RuntimeError('U-Net operation manifest mismatch')
    for entry,op in zip(hardware['operations'],ops):
        if entry['name'] != op['name'] or entry['start'] != cursor or entry['stop'] <= cursor:
            raise RuntimeError('U-Net operation range mismatch')
        cursor = entry['stop']
    if cursor != halt:
        raise RuntimeError('U-Net operations do not end at HALT')
    shared._scan_queue_configs(program,0,len(program))
    issues = udc.check_isa_jumps(shared.decode_precompiled_program(hardware),hardware['program_address'])
    if issues:
        raise RuntimeError(str(issues))


class WholeGraphBackend:
    def __init__(self,engine,hardware,timeout=300,trace_directory=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        self.ue,self.hw,self.timeout = engine,hardware,timeout
        self.trace_directory = trace_directory
        self.cycles = self.kicks = 0
        self.trace_result = None
        self.trace_export_s = 0.0
        self.model_upload_bytes = hardware['model_image'].numel()
        start = time.perf_counter()
        written = engine.dma_write(engine.h2c_device,hardware['model_base'],hardware['model_image'],self.model_upload_bytes)
        if written != self.model_upload_bytes:
            raise RuntimeError('short U-Net model upload')
        self.model_upload_s = time.perf_counter()-start

    def execute(self,x):
        width,height = self.hw['resolution']
        if tuple(x.shape) != (3,height,width):
            raise ValueError('input resolution is not the compiled profile; rebuild bin')
        ue = self.ue
        packed = udc.conv2d_pack_activation_map(x.to(torch.bfloat16),0)
        nbytes = packed.numel()*2
        if ue.is_queue_busy():
            raise RuntimeError('U-Net engine is busy')
        if ue.dma_write(ue.h2c_device,shared.INPUT_BASE,packed,nbytes) != nbytes:
            raise RuntimeError('short U-Net input upload')
        ue.write_reg32(udc.UE_INT_REG,1)
        ue.start_execute_from_dram(self.hw['program_address'])
        self.kicks += 1
        deadline = time.monotonic()+self.timeout
        while (ue.read_reg32(udc.UE_INT_REG)&3) != udc.INT_CAUSE_HALT or ue.is_queue_busy():
            if time.monotonic() >= deadline:
                raise TimeoutError('U-Net whole graph did not reach HALT')
            time.sleep(0.0001)
        self.cycles = ue.read_latency_cycles()
        if self.trace_directory is not None:
            from read_trace import generate_circular_tail_trace
            start = time.perf_counter()
            path = Path(self.trace_directory).expanduser()/f'unet_{width}x{height}_tail.csv'
            instructions = shared.decode_precompiled_program(self.hw)
            labels = [''] * len(instructions)
            for entry in self.hw['operations']:
                labels[entry['start']:entry['stop']] = [entry['name']]*(entry['stop']-entry['start'])
            self.trace_result = generate_circular_tail_trace(ue,path,
                instructions=instructions,instruction_labels=labels,
                program_dram_addr=self.hw['program_address'])
            self.trace_export_s = time.perf_counter()-start
        out = self.hw['tensors']['outc']
        flat = torch.empty(out['size_bytes']//2,dtype=torch.bfloat16)
        if ue.dma_read(ue.c2h_device,out['address'],flat,out['size_bytes']) != out['size_bytes']:
            raise RuntimeError('short U-Net output read')
        return flat.reshape(height,width,out['physical_channels'])[...,:2].permute(2,0,1).contiguous()
