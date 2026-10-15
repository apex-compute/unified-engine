"""Validate and execute a complete fixed-length BigCodec device image."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch

from bigcodec_common import CHECKPOINT_SHA256, HOP_LENGTH, SAMPLE_RATE
from bigcodec_device import ROOT, shared, udc
from bigcodec_layout import layout_for_hardware
from bigcodec_quantizer import decode_split_tokens

path = ROOT / 'models/dpdfnet8khz'
if str(path) not in sys.path:
    sys.path.insert(0, str(path))
from dpdfnet8khz_engine import StreamingEngine

FORMAT = 'andromeda.bigcodec.whole-utterance-v1'
DEFAULT_BIN = Path(__file__).resolve().parent / 'bigcodec_bin/bigcodec-andromeda.bin'


def image_bytes(value):
    return memoryview(value.numpy()).cast('B')


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_closed_jumps(instructions, base, halt):
    absolute = {udc.JUMP_MODE_ABSOLUTE, udc.JUMP_MODE_JNZ, udc.JUMP_MODE_JZ}
    relative = {udc.JUMP_MODE_RELATIVE, udc.JUMP_MODE_RELA_JNZ, udc.JUMP_MODE_RELA_JZ}
    for index, instruction in enumerate(instructions):
        words = instruction.words
        if (words[0] >> 8) & 15 != udc.INSTRUCTION_JUMP:
            continue
        mode = words[1] & 15
        immediate = ((words[1] >> 22) & 1023) | ((words[2] & ((1 << 22) - 1)) << 10)
        if mode in absolute:
            offset = (immediate << 3) - base
            if offset % 64 or not 0 <= offset <= halt * 32:
                raise ValueError('BigCodec jump leaves its resident program or bypasses HALT')
        elif mode in relative:
            if not 0 <= index + 1 - immediate <= halt:
                raise ValueError('BigCodec relative jump leaves its resident program')
        else:
            raise ValueError('BigCodec static programs cannot use indirect or reserved jumps')


def _validate_tensor_inventory(h, samples, operations):
    """Check static memory ranges and lifetimes without loading model weights."""
    tensors = h.get('tensors')
    if not isinstance(tensors, dict) or 'input' not in tensors:
        raise ValueError('Missing BigCodec tensor inventory')
    final = operations[-1]['output']
    first, last = {'input': -1}, {'input': -1}
    for index, operation in enumerate(operations):
        output, inputs = operation['output'], operation['inputs']
        if output in first or any(name not in first for name in inputs):
            raise ValueError('Invalid tensor production order')
        first[output] = index
        last[output] = index
        for name in inputs:
            last[name] = index
    last[final] = len(operations)
    if set(tensors) != set(first):
        raise ValueError('Tensor inventory differs from the operations')
    scratch = h['scratch_address']
    reserved_end = h['output_address'] + h['output_bytes']
    checked = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise ValueError('Invalid tensor descriptor')
        shape = tensor.get('shape')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(not _integer(n) or n < 1 for n in shape)):
            raise ValueError('Invalid time/channel tensor shape')
        width = (shape[1] + 63) // 64 * 64
        address, size = tensor.get('address'), tensor.get('bytes')
        if (any(not _integer(tensor.get(key)) for key in
                ('address', 'bytes', 'padded_channels', 'first', 'last')) or address % 128
                or tensor.get('padded_channels') != width
                or size != shape[0] * width * 2
                or tensor.get('first') != first[name] or tensor.get('last') != last[name]):
            raise ValueError('Invalid tensor layout or lifetime')
        if name == 'input':
            if shape != [samples, 1] or address != h['input_address'] or size != h['input_bytes']:
                raise ValueError('Input tensor does not match the waveform ABI')
        elif name == final:
            if shape != [samples, 1] or address != h['output_address'] or size != h['tokens_offset']:
                raise ValueError('Output tensor does not match the waveform ABI')
        elif not reserved_end <= address < address + size <= scratch:
            raise ValueError('Intermediate tensor overlaps reserved output or workspace')
        checked.append((address, address + size, first[name], last[name]))
    for index, (start, end, born, dead) in enumerate(checked):
        for other_start, other_end, other_born, other_dead in checked[index + 1:]:
            if (start < other_end and other_start < end
                    and born <= other_dead and other_born <= dead):
                raise ValueError('Live BigCodec tensors overlap')


def validate_artifact(payload):
    if (not isinstance(payload, dict) or payload.get('format') != FORMAT
            or payload.get('model') != 'bigcodec'):
        raise ValueError('Expected a compiled BigCodec whole-utterance bin')
    if payload.get('checkpoint_sha256') != CHECKPOINT_SHA256:
        raise ValueError('BigCodec bin does not identify the pinned official checkpoint')
    h = payload.get('hardware')
    if not isinstance(h, dict) or h.get('axi_data_width_bits') != 256:
        raise ValueError('BigCodec bin must target AXI256')
    integer_fields = (
        'model_base', 'model_limit', 'program_address', 'program_offset', 'program_size',
        'parameter_bytes', 'instructions', 'halt_index', 'axi_data_width_bits',
        'compiled_samples', 'code_frames', 'input_address', 'input_bytes', 'output_address',
        'output_bytes', 'tokens_address', 'tokens_offset', 'tensor_base', 'tensor_limit',
        'tensor_end', 'scratch_address', 'scratch_bytes', 'identity_address', 'zero_address')
    if any(not _integer(h.get(key)) for key in integer_fields):
        raise ValueError('Hardware addresses, counts and bounds must be integers')
    # Exact supported arena tuples preserve legacy bins while allowing the
    # longer-utterance layout inside the board's visible 2 GiB DRAM window.
    layout = layout_for_hardware(h)
    image = h.get('model_image')
    if (not isinstance(image, torch.Tensor) or image.dtype != torch.uint8
            or image.ndim != 1 or image.device.type != 'cpu' or not image.is_contiguous()
            or not 0 < image.numel() <= layout.model_limit - layout.model_base):
        raise ValueError('Invalid model image')
    raw = image_bytes(image)
    if hashlib.sha256(raw).hexdigest() != h.get('model_sha256'):
        raise ValueError('Model image checksum mismatch')
    offset, size = h.get('program_offset'), h.get('program_size')
    if (not _integer(offset) or offset < 0 or offset % 128
            or not _integer(size) or size <= 0 or size % 64
            or offset + size != len(raw)
            or h.get('program_address') != layout.model_base + offset):
        raise ValueError('Invalid resident instruction bounds')
    program = raw[offset:offset + size]
    if hashlib.sha256(program).hexdigest() != h.get('program_sha256'):
        raise ValueError('Instruction checksum mismatch')
    kinds = shared._instruction_types(program)
    if any(op not in {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12} for op in kinds):
        raise ValueError('Reserved BigCodec instruction type')
    halts = [i for i, op in enumerate(kinds) if op == udc.INSTRUCTION_HALT]
    if len(halts) != 1 or udc.INSTRUCTION_SWI in kinds:
        raise ValueError('BigCodec requires one terminal HALT and no SWI')
    halt = halts[0]
    parameters = h.get('parameter_bytes')
    if (h.get('instructions') != len(kinds) or h.get('halt_index') != halt
            or not _integer(parameters) or not 0 < parameters <= offset
            or offset - parameters >= 128 or any(raw[parameters:offset])):
        raise ValueError('Invalid instruction count or parameter extent')
    for name, length in (('zero_address', udc.URAM_NEAR_FULL_SIZE), ('identity_address', 64 * 64 * 2)):
        address = h.get(name)
        if (not _integer(address) or address % 128
                or not layout.model_base <= address < address + length <= layout.model_base + parameters):
            raise ValueError('Invalid resident constant range')
    if any(op != udc.INSTRUCTION_NOP for op in kinds[halt + 1:]):
        raise ValueError('HALT must be terminal')
    cursor = 0
    operations = h.get('operations')
    if not isinstance(operations, list) or not operations:
        raise ValueError('Missing whole-graph operation inventory')
    names = set()
    for op in operations:
        if (not isinstance(op, dict) or op.get('start') != cursor
                or not _integer(op.get('start')) or not _integer(op.get('stop'))
                or op['stop'] <= cursor or not isinstance(op.get('name'), str)
                or not op['name'] or op['name'] in names
                or op.get('op') not in {'conv1d', 'conv_transpose1d', 'activation', 'lstm', 'quantizer', 'add', 'tanh'}
                or not isinstance(op.get('output'), str) or not op['output']
                or not isinstance(op.get('inputs'), list) or not op['inputs']
                or any(not isinstance(name, str) or not name for name in op['inputs'])):
            raise ValueError('Invalid operation instruction coverage')
        names.add(op['name'])
        cursor = op['stop']
    if cursor != halt:
        raise ValueError('Operation ranges do not cover the complete program')
    samples, frames = h.get('compiled_samples'), h.get('code_frames')
    native = payload.get('native_samples')
    if (not _integer(samples) or samples <= 0 or samples % HOP_LENGTH
            or not _integer(frames) or frames != samples // HOP_LENGTH
            or not _integer(native) or native < 1
            or native + HOP_LENGTH - native % HOP_LENGTH != samples
            or payload.get('sample_rate') != SAMPLE_RATE or payload.get('hop_length') != HOP_LENGTH
            or payload.get('full_utterance') is not True
            or payload.get('all_neural_operations_on_device') is not True):
        raise ValueError('Invalid BigCodec sample/token dimensions')
    wave_bytes = samples * 64 * 2
    if (h.get('input_address') != layout.input_base or h.get('input_bytes') != wave_bytes
            or wave_bytes > layout.input_limit - layout.input_base
            or h.get('output_address') != layout.tensor_base
            or h.get('tokens_offset') != wave_bytes
            or h.get('tokens_address') != layout.tensor_base + wave_bytes
            or h.get('output_bytes') != wave_bytes + frames * 64 * 2
            or layout.tensor_base + h['output_bytes'] > layout.tensor_limit
            or h.get('tensor_base') != layout.tensor_base or h.get('tensor_limit') != layout.tensor_limit
            or not _integer(h.get('scratch_address')) or h['scratch_address'] % 128
            or not _integer(h.get('scratch_bytes')) or h['scratch_bytes'] <= 0 or h['scratch_bytes'] % 128
            or not _integer(h.get('tensor_end'))
            or h['tensor_end'] != h['scratch_address'] + h['scratch_bytes']
            or not layout.tensor_base + h['output_bytes'] <= h['scratch_address'] < h['tensor_end'] <= layout.tensor_limit):
        raise ValueError('Invalid audio/token DRAM bounds')
    _validate_tensor_inventory(h, samples, operations)
    shared._scan_queue_configs(program, 0, len(program))
    # Decode this view directly; the shared decoder copies the entire weight
    # image before slicing out its instruction stream.
    decoded = []
    for position in range(0, len(program), udc.INSTRUCTION_SIZE_BYTES):
        instruction = udc.Instructions()
        instruction.words = list(struct.unpack_from('<8I', program, position))
        decoded.append(instruction)
    _validate_closed_jumps(decoded, h['program_address'], halt)
    issues = udc.check_isa_jumps(decoded, h['program_address'],
                                name='BigCodec whole utterance')
    if issues:
        raise ValueError('Invalid BigCodec instruction jumps: ' + '; '.join(issues))
    return h


def load_artifact(path=DEFAULT_BIN):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    validate_artifact(payload)
    return payload


class WholeGraphBackend:
    def __init__(self, engine, payload, *, axi_data_width_bits, timeout_s=300.0):
        self.h = dict(validate_artifact(payload))
        if axi_data_width_bits != 256 or engine.conv_geometry_mode != udc.CONV_GEOMETRY_QUEUE_CONFIG:
            raise ValueError('BigCodec requires the AXI256 queue-CONFIG FPGA')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Timeout must be positive and finite')
        if engine.is_queue_busy():
            raise RuntimeError('FPGA queue is busy')
        self.engine, self.timeout_s = engine, timeout_s
        self.model_upload_writes = self.input_upload_writes = self.program_kicks = 0
        self.halts = self.output_reads = 0
        self.failed = False
        started = time.perf_counter()
        count = engine.dma_write(engine.h2c_device, self.h['model_base'],
                                 self.h['model_image'], self.h['model_image'].numel())
        if count != self.h['model_image'].numel():
            raise RuntimeError('Short BigCodec model upload')
        self.model_upload_writes = 1
        self.model_upload_s = time.perf_counter() - started
        self.last_metrics = {}

    def execute(self, padded_audio):
        if self.failed:
            raise RuntimeError('BigCodec backend stopped after a failed execution')
        source = np.asarray(padded_audio, dtype=np.float32)
        if source.shape != (self.h['compiled_samples'],) or not np.isfinite(source).all():
            raise ValueError('Input must match the bin’s complete padded waveform length and be finite')
        packed = torch.zeros(self.h['compiled_samples'], 64, dtype=torch.bfloat16)
        packed[:, 0] = torch.from_numpy(source)
        if not torch.isfinite(packed).all():
            raise ValueError('Input exceeds finite BF16 range')
        engine = self.engine
        if engine.is_queue_busy():
            raise RuntimeError('FPGA queue is busy')
        started = time.perf_counter()
        try:
            count = engine.dma_write(engine.h2c_device, self.h['input_address'], packed,
                                     self.h['input_bytes'])
            if count != self.h['input_bytes']:
                raise RuntimeError('Short audio upload')
            self.input_upload_writes += 1
            engine.write_reg32(udc.UE_INT_REG, 1)
            engine.start_execute_from_dram(self.h['program_address'])
            self.program_kicks += 1
            deadline = time.monotonic() + self.timeout_s
            while engine.read_reg32(udc.UE_INT_REG) & 3 != udc.INT_CAUSE_HALT:
                if time.monotonic() >= deadline:
                    raise TimeoutError('BigCodec did not reach HALT')
                time.sleep(0.0001)
            while engine.is_queue_busy():
                if time.monotonic() >= deadline:
                    raise TimeoutError('BigCodec queue remained busy after HALT')
                time.sleep(0.0001)
            self.halts += 1
            cycles = engine.read_latency_cycles()
            output = torch.empty(self.h['output_bytes'] // 2, dtype=torch.bfloat16)
            count = engine.dma_read(engine.c2h_device, self.h['output_address'], output,
                                    self.h['output_bytes'])
            if count != self.h['output_bytes']:
                raise RuntimeError('Short audio/token output read')
            self.output_reads += 1
            wave_elements = self.h['tokens_offset'] // 2
            wave_rows = output[:wave_elements].reshape(-1, 64)
            waveform = wave_rows[:, 0].float().numpy()
            tokens = decode_split_tokens(output[wave_elements:].reshape(-1, 64)).numpy()
            if not np.isfinite(waveform).all():
                raise FloatingPointError('FPGA produced nonfinite BigCodec audio')
            self.last_metrics = {'host_execute_s': time.perf_counter() - started,
                                 'fpga_cycles': cycles, 'finite': True,
                                 'model_upload_writes': self.model_upload_writes,
                                 'input_upload_writes': self.input_upload_writes,
                                 'program_kicks': self.program_kicks,
                                 'halts': self.halts, 'output_reads': self.output_reads,
                                 'waveform_padding_finite': bool(torch.isfinite(wave_rows[:, 1:]).all()),
                                 'waveform_padding_nonzero': int((wave_rows[:, 1:] != 0).sum()),
                                 'cpu_neural_ops': 0}
            return waveform, tokens
        except BaseException:
            self.failed = True
            raise
