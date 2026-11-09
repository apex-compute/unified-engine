"""YOLO26n/s offline graph compilation, checked artifacts and CPU reference."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from yolo26_common import TensorSpec, FloatBackend, execute_model, _class_names, variant_paths
from yolov5_common import (fold_conv_bn, gather_if8_is_profitable, quantize_conv_gather_if8,
                           QuantizedConv)
from yolov5_artifact import ArtifactCompileBackend, _pack_codes, _unpack_codes
from yolov5_precompiled import compile_precompiled_hardware, validate_precompiled_hardware, _tensor_bytes
import quant_lib
import user_dma_core as core

VERSION = 1


def quantize_conv(conv, bn):
    if gather_if8_is_profitable(channels=conv.in_channels, out_channels=conv.out_channels,
                                kernel_h=conv.kernel_size[0], kernel_w=conv.kernel_size[1]):
        return quantize_conv_gather_if8(conv, bn)
    weight, bias = fold_conv_bn(conv, bn)
    oc, channels, kh, kw = weight.shape
    ct = (channels + 63) // 64
    padded = torch.zeros(oc, ct * 64, kh, kw)
    padded[:, :channels] = weight
    blocks = padded.view(oc, ct, 64, kh, kw).permute(0, 3, 4, 1, 2).contiguous().view(-1, 64)
    data, scales = quant_lib.quantize_if8(blocks, block_size=64, int_variant=None)
    codes = (torch.frombuffer(bytearray(data), dtype=torch.uint8).view(oc, kh, kw, ct, 64)
             .permute(0, 3, 4, 1, 2).reshape(oc, ct * 64, kh, kw)[:, :channels].contiguous())
    return QuantizedConv(codes=codes,
        block_scales=core.tensor_from_le_bytes(scales, torch.bfloat16).view(oc, kh * kw * ct),
        bias=bias.to(torch.bfloat16) if bias is not None else None,
        data_type=core.TYPE.IF8, gather=False)


class CompileBackend(ArtifactCompileBackend):
    def __init__(self, weights=None, *, optimize=False):
        super().__init__()
        self.weights = {} if weights is None else weights
        self.optimize = optimize

    def conv_tensor(self, name, x, conv, bn, *, activate):
        kh, kw = conv.kernel_size
        if kh != kw or conv.groups != 1 or len(set(conv.stride)) != 1 or len(set(conv.padding)) != 1:
            raise ValueError(f'{name}: unsupported convolution geometry')
        if x.shape[0] != conv.in_channels:
            raise ValueError(f'{name}: input channels differ')
        if name not in self.weights:
            prepared = quantize_conv(conv, bn)
            packed, shape = _pack_codes(prepared.codes, 'if8')
            self.weights[name] = dict(precision='if8', layout='gather' if prepared.gather else 'channels',
                codes_shape=shape, codes_packed=packed, block_scales=prepared.block_scales.contiguous(),
                bias=prepared.bias if prepared.bias is not None else torch.empty(0, dtype=torch.bfloat16))
        h, w = self._output_hw(x, kh, conv.stride[0], conv.padding[0], conv.dilation[0])
        result = TensorSpec(name, (conv.out_channels, h, w))
        kind = ('depthwise' if self.optimize and getattr(conv, '_andromeda_depthwise', False)
                and conv.in_channels > 64 else 'conv')
        self.operations.append(dict(op=kind, name=name, inputs=[x.key], output=name,
            output_shape=list(result.shape), stride=conv.stride[0], pad=conv.padding[0],
            dilation=conv.dilation[0], activate=activate))
        return result

    def select(self, name, x, start, count, channels):
        if start < 0 or count <= 0 or start + count > x.shape[0] or channels < count:
            raise ValueError('Invalid channel selection')
        shape = (channels, *x.shape[1:])
        self.operations.append(dict(op='select', name=name, inputs=[x.key], output=name,
                                    output_shape=list(shape), start=start, count=count))
        return TensorSpec(name, shape)

    def attention(self, name, q, k, v, scale):
        if not q.shape == k.shape == v.shape or q.shape[0] != 64:
            raise ValueError('Attention shape mismatch')
        self.operations.append(dict(op='attention', name=name, inputs=[q.key, k.key, v.key],
                                    output=name, output_shape=list(v.shape), scale=scale))
        return TensorSpec(name, v.shape)


def tensor_digest(value):
    return hashlib.sha256(_tensor_bytes(value.contiguous())).hexdigest()


def descriptor(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=tensor_digest(value))
    if isinstance(value, dict):
        return {key: descriptor(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [descriptor(v) for v in value]
    return value


def bundle_digest(payload):
    meta = {k: v for k, v in payload.items() if k != 'bundle_sha256'}
    return hashlib.sha256(json.dumps(descriptor(meta), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select_profile(payload, resolution=None):
    resolution = resolution or payload['default_resolution']
    if resolution not in payload['profiles']:
        raise ValueError(f'Unsupported resolution {resolution!r}; choose {list(payload["profiles"])}')
    result = dict(payload['profiles'][resolution])
    result['weights'] = payload['weights']
    result['model'] = dict(result['model'], names=payload['names'])
    return result


def compile_artifact(model, output, *, winograd=False, optimize=False):
    variant = getattr(model, '_andromeda_variant', 'n')
    config, _, _ = variant_paths(variant)
    weights = {}
    winograd = bool(winograd or optimize)
    payload = dict(format=f'andromeda.yolo26{variant}.single-bin', artifact_version=2 if optimize else VERSION, source_commit=config['source_commit'],
        checkpoint_sha256=config['checkpoint_sha256'], default_resolution=config['default_resolution'],
        names=_class_names(model), head='one2one', precision=config['precision'],
        winograd=bool(winograd), weights=weights, profiles={})
    if optimize:
        payload['optimized'] = True
    for resolution in config['resolutions']:
        w, h = map(int, resolution.split('x'))
        backend = CompileBackend(weights, optimize=optimize)
        heads = execute_model(model, TensorSpec('input', (3, h, w)), backend)
        selected = dict(model=dict(input_shape=[3, h, w]), operations=backend.operations,
                        head_outputs=[head.key for head in heads], weights=weights)
        with contextlib.redirect_stdout(io.StringIO()):
            hardware = compile_precompiled_hardware(selected, winograd=winograd)
        selected.pop('weights')
        selected['hardware'] = hardware
        payload['profiles'][resolution] = selected
        print(f'{resolution}: {len(backend.operations)} operations, '
              f'{hardware["program_size"]} program bytes, '
              f'{len(hardware.get("winograd_layers", []))} Winograd layers', flush=True)
    payload['bundle_sha256'] = bundle_digest(payload)
    validate_artifact(payload)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def artifact_variant(payload):
    for variant in ('n', 's'):
        if payload.get('format') == f'andromeda.yolo26{variant}.single-bin':
            return variant
    raise RuntimeError('Unsupported YOLO26 artifact format')


def validate_artifact(payload):
    expected = {'format', 'artifact_version', 'source_commit', 'checkpoint_sha256', 'default_resolution',
                'names', 'head', 'precision', 'winograd', 'weights', 'profiles', 'bundle_sha256'}
    if isinstance(payload, dict) and payload.get('artifact_version') == 2:
        expected.add('optimized')
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RuntimeError('Invalid YOLO26 artifact schema')
    config, _, _ = variant_paths(artifact_variant(payload))
    if payload['artifact_version'] not in (VERSION, 2):
        raise RuntimeError('Unsupported YOLO26 artifact version; rebuild with --force')
    if payload['artifact_version'] == 2 and (payload['optimized'] is not True or payload['winograd'] is not True):
        raise RuntimeError('YOLO26 optimized artifacts require all optimization passes')
    for key in ('source_commit', 'checkpoint_sha256', 'default_resolution', 'precision'):
        if payload[key] != config[key]:
            raise RuntimeError(f'YOLO26 artifact {key} differs from the pinned configuration')
    if payload['head'] != 'one2one' or len(payload['names']) != 80:
        raise RuntimeError('Expected the 80-class NMS-free YOLO26 head')
    if set(payload['profiles']) != set(config['resolutions']):
        raise RuntimeError('YOLO26 resolution set differs')
    if bundle_digest(payload) != payload['bundle_sha256']:
        raise RuntimeError('YOLO26 artifact digest does not match')
    if not isinstance(payload['winograd'], bool):
        raise RuntimeError('YOLO26 convolution strategy must be boolean')
    strategy = 'optimized' if payload.get('optimized') else ('winograd' if payload['winograd'] else 'native')
    if payload['bundle_sha256'] != config['bundle_sha256'].get(strategy):
        raise RuntimeError('YOLO26 artifact differs from the canonical compiled model: ' + payload['bundle_sha256'])
    for resolution in config['resolutions']:
        selected = select_profile(payload, resolution)
        w, h = map(int, resolution.split('x'))
        if selected['model']['input_shape'] != [3, h, w]:
            raise RuntimeError('YOLO26 input shape differs')
        shapes = {'input': (3, h, w)}
        for op in selected['operations']:
            if op['output'] in shapes or any(x not in shapes for x in op['inputs']):
                raise RuntimeError('Invalid YOLO26 graph dependency')
            if op['op'] not in ('conv', 'maxpool', 'upsample2x', 'add', 'concat', 'attention', 'select', 'depthwise'):
                raise RuntimeError('Unsupported YOLO26 graph operation')
            if op['op'] in ('conv', 'depthwise') and op['name'] not in payload['weights']:
                raise RuntimeError('Missing YOLO26 convolution weights')
            shapes[op['output']] = tuple(op['output_shape'])
        if [shapes[k] for k in selected['head_outputs']] != [(84, h // s, w // s) for s in (8, 16, 32)]:
            raise RuntimeError('YOLO26 detection-head shapes differ')
        if ('winograd_layers' in selected['hardware']) != payload['winograd']:
            raise RuntimeError('YOLO26 convolution strategy differs')
        validate_precompiled_hardware(selected)


def load_artifact(path):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    validate_artifact(payload)
    return payload


def execute_cpu(selected, image, *, intermediates=None):
    if tuple(image.shape) != tuple(selected['model']['input_shape']):
        raise ValueError('Image shape differs from the selected YOLO26 profile')
    backend = FloatBackend(quantized=True)
    values = {'input': image.to(torch.bfloat16)}
    prepared = {}
    for name, entry in selected['weights'].items():
        prepared[name] = QuantizedConv(codes=_unpack_codes(entry['codes_packed'], entry['codes_shape'], 'if8'),
            block_scales=entry['block_scales'], bias=entry['bias'] if entry['bias'].numel() else None,
            data_type=core.TYPE.IF8, gather=entry['layout'] == 'gather')
    for op in selected['operations']:
        inputs = [values[k] for k in op['inputs']]
        kind, name = op['op'], op['name']
        if kind in ('conv', 'depthwise'):
            value = backend.conv_prepared(name, inputs[0], prepared[name], stride=op['stride'],
                                          pad=op['pad'], dilation=op['dilation'], activate=op['activate'])
        elif kind == 'attention':
            value = backend.attention(name, *inputs, op['scale'])
        elif kind == 'select':
            value = inputs[0][op['start']:op['start'] + op['count']]
            if value.shape[0] < op['output_shape'][0]:
                value = torch.cat((value, value.new_zeros(op['output_shape'][0] - value.shape[0], *value.shape[1:])))
        elif kind == 'concat':
            value = backend.concat(name, inputs)
        elif kind == 'add':
            value = backend.add(name, *inputs)
        elif kind == 'maxpool':
            value = backend.maxpool(name, inputs[0], kernel=op['kernel'], stride=op['stride'], pad=op['pad'])
        elif kind == 'upsample2x':
            value = backend.upsample2x(name, inputs[0])
        else:
            raise ValueError(f'Unsupported operation {kind}')
        values[op['output']] = value
        if intermediates is not None:
            intermediates[name] = value
    return [values[key] for key in selected['head_outputs']]
