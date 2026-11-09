"""Pinned YOLO26n/s graph adapter and NMS-free detection decoding.

The checkpoint is loaded with an explicit weights-only allow-list. No
Ultralytics installation or executable checkpoint classes are required.
"""
from __future__ import annotations

import collections
import contextlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE.parent / 'yolov5s'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolov5_common import TorchBackend, Detection, ensure_checkpoint, sha256_file, _class_names
from yolov5_artifact import TensorSpec

def variant_paths(variant='n'):
    if variant not in ('n', 's'):
        raise ValueError(f'Unsupported YOLO26 variant: {variant!r}')
    directory = HERE.parent / f'yolo26{variant}'
    config = json.loads((directory / f'yolo26{variant}_config.json').read_text())
    bins = directory / f'yolo26{variant}_bin'
    return config, bins / f'yolo26{variant}.pt', bins / f'yolo26{variant}-andromeda.bin'


CONFIG, CHECKPOINT, ARTIFACT = variant_paths()


def add_strategy_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--optimize', dest='strategy', action='store_const', const='optimized',
                       default='optimized', help='Use channel copies, block-depthwise and Winograd (default)')
    group.add_argument('--winograd', dest='strategy', action='store_const', const='winograd',
                       help='Use the Winograd-only comparison path')
    group.add_argument('--native', dest='strategy', action='store_const', const='native',
                       help='Use the native convolution baseline')
    return group


def strategy_artifact_path(variant='n', strategy='optimized'):
    suffixes = {'native': '', 'winograd': '_winograd', 'optimized': '_optimized'}
    if strategy not in suffixes:
        raise ValueError(f'Unsupported convolution strategy: {strategy!r}')
    _, _, artifact = variant_paths(variant)
    return artifact.with_name(artifact.stem + suffixes[strategy] + artifact.suffix)


@contextlib.contextmanager
def checkpoint_classes():
    names = {
        'ultralytics.nn.tasks': ('DetectionModel',),
        'ultralytics.nn.modules.conv': ('Conv', 'DWConv', 'Concat'),
        'ultralytics.nn.modules.block': ('C3k2', 'C3k', 'Bottleneck', 'SPPF',
                                         'C2PSA', 'PSABlock', 'Attention'),
        'ultralytics.nn.modules.head': ('Detect',),
    }
    safe = [collections.OrderedDict, set, torch.nn.Sequential, torch.nn.ModuleList,
            torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.SiLU,
            torch.nn.MaxPool2d, torch.nn.Upsample, torch.nn.Identity]
    # Supplying explicit (class, serialized-name) entries avoids replacing any
    # installed ultralytics modules in sys.modules.
    for module, classes in names.items():
        for name in classes:
            cls = type(name, (torch.nn.Module,), {'__module__': module})
            safe.append((cls, f'{module}.{name}'))
    with torch.serialization.safe_globals(safe):
        yield


def load_checkpoint(path=None, *, variant='n'):
    config, default, _ = variant_paths(variant)
    path = Path(path or default)
    if sha256_file(path) != config['checkpoint_sha256']:
        raise RuntimeError(f'YOLO26{variant} checkpoint SHA-256 differs from the pinned official release')
    with checkpoint_classes():
        payload = torch.load(path, map_location='cpu', weights_only=True)
    model = (payload.get('ema') or payload['model']).float().eval().requires_grad_(False)
    expected = ('Conv', 'Conv', 'C3k2', 'Conv', 'C3k2', 'Conv', 'C3k2', 'Conv',
                'C3k2', 'SPPF', 'C2PSA', 'Upsample', 'Concat', 'C3k2', 'Upsample',
                'Concat', 'C3k2', 'Conv', 'Concat', 'C3k2', 'Conv', 'Concat', 'C3k2', 'Detect')
    if tuple(type(m).__name__ for m in model.model) != expected:
        raise RuntimeError(f'Unexpected YOLO26{variant} topology')
    head = model.model[-1]
    if head.nc != 80 or head.reg_max != 1 or tuple(model.stride.tolist()) != (8, 16, 32):
        raise RuntimeError('Expected the 80-class, DFL-free YOLO26 detection head')
    if not hasattr(head, 'one2one_cv2') or not hasattr(head, 'one2one_cv3'):
        raise RuntimeError('Checkpoint has no NMS-free one-to-one head')
    model._andromeda_variant = variant
    return model


def dense_conv(conv):
    """Exact grouped-convolution lowering to a sparse dense kernel.

    The native walker has no grouped opcode. Zero off-group weights preserve
    the operation; this is a correctness path, not a depthwise speed claim.
    """
    if conv.groups == 1:
        return conv
    out = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                          stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
                          bias=conv.bias is not None)
    with torch.no_grad():
        out.weight.zero_()
        ci, co = conv.in_channels // conv.groups, conv.out_channels // conv.groups
        for group in range(conv.groups):
            out.weight[group * co:(group + 1) * co, group * ci:(group + 1) * ci] = conv.weight[group * co:(group + 1) * co]
        if conv.bias is not None:
            out.bias.copy_(conv.bias)
    out._andromeda_depthwise = conv.groups == conv.in_channels == conv.out_channels
    return out.eval()


def select_conv(channels, start, count, output_channels=None):
    """Exact channel selection, including zero-padding Q/K to 64 lanes."""
    output_channels = output_channels or count
    if start < 0 or count <= 0 or start + count > channels or output_channels < count:
        raise ValueError('Invalid channel selection')
    conv = torch.nn.Conv2d(channels, output_channels, 1, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[torch.arange(count), torch.arange(start, start + count), 0, 0] = 1
    return conv.eval()


def select(backend, name, x, start, count, output_channels=None):
    if isinstance(x, TensorSpec):
        if getattr(backend, 'optimize', False) and count % 32 == 0:
            return backend.select(name, x, start, count, output_channels or count)
        return backend.conv_tensor(name, x, select_conv(x.shape[0], start, count, output_channels),
                                    None, activate=False)
    value = x[start:start + count]
    if output_channels is not None and output_channels > count:
        value = torch.cat((value, value.new_zeros(output_channels - count, *x.shape[1:])))
    return value


def run_module(module, x, backend, name):
    kind = type(module).__name__
    if kind in ('Conv', 'DWConv'):
        if not isinstance(module.act, (torch.nn.SiLU, torch.nn.Identity)):
            raise ValueError(f'{name}: unsupported activation')
        return backend.conv_tensor(name, x, dense_conv(module.conv), module.bn,
                                    activate=isinstance(module.act, torch.nn.SiLU))
    if isinstance(module, torch.nn.Conv2d):
        return backend.conv_tensor(name, x, dense_conv(module), None, activate=False)
    if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
        for i, child in enumerate(module):
            x = run_module(child, x, backend, f'{name}.{i}')
        return x
    if kind == 'Bottleneck':
        y = run_module(module.cv1, x, backend, name + '.cv1')
        y = run_module(module.cv2, y, backend, name + '.cv2')
        return backend.add(name + '.add', x, y) if module.add else y
    if kind == 'C3k':
        a = run_module(module.cv1, x, backend, name + '.cv1')
        a = run_module(module.m, a, backend, name + '.m')
        b = run_module(module.cv2, x, backend, name + '.cv2')
        return run_module(module.cv3, backend.concat(name + '.cat', [a, b]), backend, name + '.cv3')
    if kind in ('C3k2', 'C2PSA'):
        y = run_module(module.cv1, x, backend, name + '.cv1')
        a = select(backend, name + '.split0', y, 0, module.c)
        b = select(backend, name + '.split1', y, module.c, module.c)
        pieces = [a, b]
        for i, block in enumerate(module.m):
            b = run_module(block, b, backend, f'{name}.m.{i}')
            if kind == 'C3k2':
                pieces.append(b)
        if kind == 'C2PSA':
            pieces = [a, b]
        return run_module(module.cv2, backend.concat(name + '.cat', pieces), backend, name + '.cv2')
    if kind == 'SPPF':
        pieces = [run_module(module.cv1, x, backend, name + '.cv1')]
        for i in range(getattr(module, 'n', 3)):
            pieces.append(backend.maxpool(f'{name}.pool{i}', pieces[-1], kernel=module.m.kernel_size,
                                          stride=module.m.stride, pad=module.m.padding))
        y = run_module(module.cv2, backend.concat(name + '.cat', pieces), backend, name + '.cv2')
        return backend.add(name + '.add', x, y) if getattr(module, 'add', False) else y
    if kind == 'PSABlock':
        y = run_module(module.attn, x, backend, name + '.attn')
        x = backend.add(name + '.attn_add', x, y) if module.add else y
        y = run_module(module.ffn, x, backend, name + '.ffn')
        return backend.add(name + '.ffn_add', x, y) if module.add else y
    if kind == 'Attention':
        if module.head_dim != 64 or module.key_dim != 32:
            raise ValueError(f'{name}: unsupported YOLO26n attention dimensions')
        qkv = run_module(module.qkv, x, backend, name + '.qkv')
        attended, values = [], []
        for h in range(module.num_heads):
            offset = h * (module.head_dim + 2 * module.key_dim)
            q = select(backend, f'{name}.q{h}', qkv, offset, module.key_dim, 64)
            k = select(backend, f'{name}.k{h}', qkv, offset + module.key_dim, module.key_dim, 64)
            v = select(backend, f'{name}.v{h}', qkv, offset + 2 * module.key_dim, module.head_dim)
            attended.append(backend.attention(f'{name}.head{h}', q, k, v, module.scale))
            values.append(v)
        value = backend.concat(name + '.vcat', values)
        pe = run_module(module.pe, value, backend, name + '.pe')
        y = backend.add(name + '.pe_add', backend.concat(name + '.acat', attended), pe)
        return run_module(module.proj, y, backend, name + '.proj')
    if kind == 'Upsample':
        if module.scale_factor != 2 or module.mode != 'nearest':
            raise ValueError(f'{name}: unsupported upsampling')
        return backend.upsample2x(name, x)
    if kind == 'Concat':
        return backend.concat(name, x)
    raise ValueError(f'{name}: unsupported module {kind}')


def execute_model(model, image, backend):
    outputs = []
    x = image
    for i, module in enumerate(model.model):
        if isinstance(module.f, int):
            x = x if module.f == -1 else outputs[module.f]
        else:
            x = [x if j == -1 else outputs[j] for j in module.f]
        if type(module).__name__ == 'Detect':
            heads = []
            for j, feature in enumerate(x):
                boxes = run_module(module.one2one_cv2[j], feature, backend, f'model.{i}.one2one_cv2.{j}')
                scores = run_module(module.one2one_cv3[j], feature, backend, f'model.{i}.one2one_cv3.{j}')
                heads.append(backend.concat(f'head{j}', [boxes, scores]))
            return heads
        x = run_module(module, x, backend, f'model.{i}')
        outputs.append(x)
    raise RuntimeError('Missing detection head')


class FloatBackend(TorchBackend):
    # Use the tensor's spatial shape, not model-global resolution.
    def attention(self, name, q, k, v, scale):
        shape = v.shape
        q, k, v = (t.flatten(1).T.float() for t in (q, k, v))
        out = (((q * scale) @ k.T).softmax(-1) @ v).T.reshape(shape)
        return out.to(torch.bfloat16) if self.quantized else out


def decode_heads(heads, names, *, confidence=.25, max_det=300):
    """One-to-one head: anchor distances + class sigmoid + two-stage top-k; no NMS."""
    if not 0 <= confidence <= 1 or max_det < 1 or max_det > 300:
        raise ValueError('confidence must be in [0,1], max_det in [1,300]')
    if len(heads) != 3:
        raise ValueError('Expected three YOLO26 detection heads')
    boxes, scores = [], []
    for raw, stride in zip(heads, (8, 16, 32)):
        if raw.ndim != 3 or raw.shape[0] != 84 or not torch.isfinite(raw).all():
            raise ValueError('Invalid YOLO26 detection head')
        _, h, w = raw.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        anchor = torch.stack((x + .5, y + .5)).reshape(2, -1)
        distance = raw[:4].float().reshape(4, -1)
        boxes.append(torch.cat((anchor - distance[:2], anchor + distance[2:])).T * stride)
        scores.append(raw[4:].float().reshape(80, -1).T.sigmoid())
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    count = min(max_det, len(boxes))
    selected = scores.amax(1).topk(count).indices
    values, indices = scores[selected].flatten().topk(count)
    rows, labels = selected[indices // 80], indices % 80
    return [Detection(tuple(float(x) for x in boxes[row]), float(score), int(label), names[int(label)])
            for row, label, score in zip(rows, labels, values) if score >= confidence]
