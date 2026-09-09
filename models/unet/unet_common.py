"""Carvana U-Net inference graph and Andromeda layer adapter.

Topology/checkpoint names follow milesial/Pytorch-UNet (bilinear=False).
The legacy Backend adapter is retained for layer diagnostics. The public
hardware runner uses unet_precompiled's whole-graph image. CPU execution is a numerical
reference, not a bit-exact model of the BF19/BF20 arithmetic pipeline.
"""
from pathlib import Path
import hashlib
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import user_dma_core as udc

FORMAT = 'andromeda.unet.layer-artifact-v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantize(weight):
    """Symmetric IF8 INT weights, one BF16 magnitude per output channel.

    A channel-uniform scale also works with gather's different block order.
    Negative hardware scales select the INT8 decoder.
    """
    magnitude = (weight.abs().flatten(1).amax(1) / 127).clamp_min(2**-126)
    magnitude = magnitude.to(torch.bfloat16).float()
    codes = (weight / magnitude[:, None, None, None]).round().clamp(-127, 127)
    return codes.to(torch.int8), magnitude.to(torch.bfloat16)


def phase_weights(weight):
    if weight.ndim != 4 or tuple(weight.shape[2:]) != (2, 2):
        raise ValueError('expected ConvTranspose weight [IC, OC, 2, 2]')
    return [weight[:, :, a, b].T.contiguous()[:, :, None, None]
            for a in range(2) for b in range(2)]


def interleave(parts):
    c, h, w = parts[0].shape
    output = parts[0].new_empty(c, h * 2, w * 2)
    for index, part in enumerate(parts):
        output[:, index // 2::2, index % 2::2] = part
    return output


def build_layers(state):
    """Validate the official two-class topology and fold inference BN."""
    layers = {}
    consumed = set()

    def tensor(key):
        consumed.add(key)
        value = state[key].detach().cpu().float()
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite checkpoint tensor: {key}')
        return value

    def add(name, weight, bias, relu, pad):
        codes, scales = quantize(weight)
        layers[name] = dict(weight=weight, bias=bias, codes=codes,
                            scales=scales, relu=relu, pad=pad)

    def double(prefix, input_c, output_c):
        for offset, ic in ((0, input_c), (3, output_c)):
            base = f'{prefix}.double_conv'
            w = tensor(f'{base}.{offset}.weight')
            if tuple(w.shape) != (output_c, ic, 3, 3):
                raise ValueError(f'{prefix}: unsupported topology {tuple(w.shape)}')
            bn = f'{base}.{offset + 1}'
            variance = tensor(f'{bn}.running_var')
            if (variance < 0).any():
                raise ValueError(f'{bn}: negative variance')
            alpha = tensor(f'{bn}.weight') / (variance + 1e-5).sqrt()
            bias = tensor(f'{bn}.bias') - alpha * tensor(f'{bn}.running_mean')
            consumed.add(f'{bn}.num_batches_tracked')
            add(f'{prefix}.{offset}', w * alpha[:, None, None, None], bias, True, 1)

    double('inc', 3, 64)
    for level, (ic, oc) in enumerate(((64,128),(128,256),(256,512),(512,1024)), 1):
        double(f'down{level}.maxpool_conv.1', ic, oc)
    for level, ic in enumerate((1024,512,256,128), 1):
        w = tensor(f'up{level}.up.weight')
        b = tensor(f'up{level}.up.bias')
        if tuple(w.shape) != (ic, ic//2, 2, 2) or tuple(b.shape) != (ic//2,):
            raise ValueError('only bilinear=False Carvana U-Net is supported')
        for phase, weight in enumerate(phase_weights(w)):
            add(f'up{level}.phase{phase}', weight, b, False, 0)
        double(f'up{level}.conv', ic, ic//2)
    w = tensor('outc.conv.weight')
    b = tensor('outc.conv.bias')
    if tuple(w.shape) != (2,64,1,1) or tuple(b.shape) != (2,):
        raise ValueError('initial U-Net adapter requires two output classes')
    add('outc', w, b, False, 0)
    extra = set(state) - consumed - {'mask_values'}
    if extra:
        raise ValueError(f'unexpected checkpoint keys: {sorted(extra)}')
    return layers


class Backend:
    def __init__(self, kind='cpu-quantized', engine=None, timeout=300):
        self.kind, self.engine, self.timeout = kind, engine, timeout
        self.cycles = 0
        self.kicks = 0
        self.metrics = []

    def conv(self, x, layer):
        if self.kind == 'cpu-fp32':
            weight, bias = layer['weight'], layer['bias']
        else:
            x = x.to(torch.bfloat16)
            weight = layer['codes'].float() * layer['scales'].float()[:,None,None,None]
            bias = layer['bias'].to(torch.bfloat16).float()
        if self.kind == 'hardware':
            ue = self.engine
            codes = layer['codes']
            oc,c,kh,kw = codes.shape
            chunks = (kh*kw*c + 63)//64
            gather = c <= 255 and chunks <= 4 and max(kh*kw*((c+3)//4), oc*chunks) < oc*kh*kw*((c+63)//64)
            blocks = chunks if gather else kh*kw*((c+63)//64)
            scales = -layer['scales'][:,None].expand(oc, blocks).contiguous()
            y = ue.run_conv2d_layer(x, codes, stride_s=1, pad=layer['pad'],
                block_scales=scales, bias=layer['bias'].to(torch.bfloat16),
                relu_enable=layer['relu'], data_type=udc.TYPE.IF8,
                gather=gather, timeout_s=self.timeout)
            self._finish(ue.last_conv_cycles)
            return y
        y = F.conv2d(x.float()[None], weight, bias, padding=layer['pad'])[0]
        if layer['relu']:
            y = y.relu()
        return y if self.kind == 'cpu-fp32' else y.to(torch.bfloat16)

    def pool(self, x):
        if self.kind == 'hardware':
            y = self.engine.run_maxpool2d_layer(x, kernel=2, stride_s=2, pad=0,
                                               timeout_s=self.timeout)
            self._finish(self.engine.last_maxpool_cycles)
            return y
        return F.max_pool2d(x, 2)

    def _finish(self, cycles):
        self.cycles += int(cycles)
        self.kicks += 1
        # Each layer returns its output to host before these arenas are reused.
        self.engine.reset_params_dram_addr()
        self.engine.reset_tensor_dram_addr()
        self.engine.reset_program_dram_addr()


def execute(layers, x, backend, observer=None):
    if x.ndim != 3 or x.shape[0] != 3 or min(x.shape[1:]) < 16:
        raise ValueError('expected RGB CHW input, height and width >=16')
    def conv(name, value):
        result = backend.conv(value, layers[name])
        if observer:
            observer(name, result)
        return result
    def double(prefix, value):
        return conv(prefix+'.3', conv(prefix+'.0', value))
    x = double('inc', x)
    skips = [x]
    for level in range(1,5):
        x = double(f'down{level}.maxpool_conv.1', backend.pool(x))
        if level < 4:
            skips.append(x)
    for level in range(1,5):
        x = interleave([conv(f'up{level}.phase{phase}', x) for phase in range(4)])
        skip = skips.pop()
        dy, dx = skip.shape[1]-x.shape[1], skip.shape[2]-x.shape[2]
        x = F.pad(x, (dx//2, dx-dx//2, dy//2, dy-dy//2))
        x = double(f'up{level}.conv', torch.cat((skip,x), dim=0))
    return conv('outc', x)


def load_artifact(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    from unet_precompiled import FORMAT as WHOLE_FORMAT, validate
    if payload.get('format') == WHOLE_FORMAT:
        if payload.get('precision') != 'IF8-INT/BF16' or payload.get('full_graph') is not True:
            raise ValueError('invalid whole-graph U-Net artifact')
        layers = payload['layers']
        validate(layers,payload['hardware'])
        return payload,layers
    if payload.get('format') != FORMAT or payload.get('precision') != 'IF8-INT/BF16':
        raise ValueError('unsupported U-Net artifact')
    # Rebuild from checkpoint tensors to validate graph and derived quantization.
    layers = build_layers(payload['state_dict'])
    return payload, layers
