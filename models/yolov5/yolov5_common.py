"""YOLOv5 graph, weight preparation, and post-processing helpers.

The official v7.0 checkpoint contains pickled module objects.  We load it with
``weights_only=True`` and a deliberately small set of placeholder classes, so
running YOLOv5 does not require cloning or importing the upstream repository.
Only tensors and the module topology are consumed; forward execution is
implemented here against either PyTorch or the Andromeda layer primitives.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import quant_lib
import user_dma_core


@dataclass(frozen=True)
class YOLOv5Variant:
    key: str
    model_name: str
    yaml_file: str
    checkpoint_url: str
    checkpoint_sha256: str
    parameter_count: int
    detect_input_channels: tuple[int, int, int]


YOLOV5_VARIANTS = {
    "s": YOLOv5Variant(
        key="s",
        model_name="YOLOv5s",
        yaml_file="yolov5s.yaml",
        checkpoint_url=(
            "https://github.com/ultralytics/yolov5/releases/download/v7.0/"
            "yolov5s.pt"),
        checkpoint_sha256=(
            "8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee"),
        parameter_count=7_235_389,
        detect_input_channels=(128, 256, 512),
    ),
    "n": YOLOv5Variant(
        key="n",
        model_name="YOLOv5n",
        yaml_file="yolov5n.yaml",
        checkpoint_url=(
            "https://github.com/ultralytics/yolov5/releases/download/v7.0/"
            "yolov5n.pt"),
        checkpoint_sha256=(
            "4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3"),
        parameter_count=1_872_157,
        detect_input_channels=(64, 128, 256),
    ),
}


def get_yolov5_variant(value: str = "s") -> YOLOv5Variant:
    """Resolve ``s``/``n`` and their common model-name spellings."""
    key = str(value).strip().lower()
    aliases = {
        "s": "s", "yolov5s": "s",
        "n": "n", "yolov5n": "n",
    }
    try:
        return YOLOV5_VARIANTS[aliases[key]]
    except KeyError as exc:
        choices = ", ".join(sorted(YOLOV5_VARIANTS))
        raise ValueError(
            f"unsupported YOLOv5 variant {value!r}; choose one of {choices}") from exc


def load_yolov5_config(
        value: str = "s", config_path: Optional[Path] = None,
        ) -> tuple[YOLOv5Variant, dict, Path]:
    """Load a variant's pinned config and return its resource directory."""
    profile = get_yolov5_variant(value)
    if config_path is None:
        models_dir = Path(__file__).resolve().parent.parent
        model_dirname = "yolov5" if profile.key == "s" else "yolov5n"
        config_filename = ("yolov5_config.json" if profile.key == "s"
                           else "yolov5n_config.json")
        config_path = models_dir / model_dirname / config_filename
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("model", {}).get("name") != profile.model_name:
        raise RuntimeError(
            f"{config_path} does not describe {profile.model_name}")
    source = config.get("source", {})
    if (source.get("weights_url") != profile.checkpoint_url
            or source.get("weights_sha256") != profile.checkpoint_sha256):
        raise RuntimeError(
            f"{config_path} does not pin the verified {profile.model_name} checkpoint")
    return profile, config, config_path.parent


YOLOV5S_V7_URL = YOLOV5_VARIANTS["s"].checkpoint_url
YOLOV5S_V7_SHA256 = YOLOV5_VARIANTS["s"].checkpoint_sha256
YOLOV5N_V7_URL = YOLOV5_VARIANTS["n"].checkpoint_url
YOLOV5N_V7_SHA256 = YOLOV5_VARIANTS["n"].checkpoint_sha256


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def ensure_checkpoint(path: Path, *, url: str = YOLOV5S_V7_URL,
                      sha256: str = YOLOV5S_V7_SHA256) -> Path:
    """Download the pinned official checkpoint if absent and verify its hash.

    An existing file with the wrong hash is never overwritten.  Downloads go
    to a temporary sibling and are atomically installed only after validation.
    """
    path = path.expanduser().resolve()
    if path.is_file():
        actual = sha256_file(path)
        if actual != sha256:
            raise RuntimeError(
                f"checkpoint checksum mismatch for {path}: got {actual}, "
                f"expected {sha256}; remove or replace the file explicitly")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Apex-Compute-unified-engine/YOLOv5"})
    tmp_name: Optional[str] = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                    prefix=f".{path.name}.", suffix=".download",
                    dir=path.parent, delete=False) as tmp:
                tmp_name = tmp.name
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
        tmp_path = Path(tmp_name)
        actual = sha256_file(tmp_path)
        if actual != sha256:
            raise RuntimeError(
                f"downloaded checkpoint checksum mismatch: got {actual}, "
                f"expected {sha256}")
        os.replace(tmp_path, path)
        tmp_name = None
        return path
    finally:
        if tmp_name is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(tmp_name).unlink()


def _placeholder(module: types.ModuleType, name: str) -> type:
    cls = type(name, (torch.nn.Module,), {})
    cls.__module__ = module.__name__
    cls.__qualname__ = name
    setattr(module, name, cls)
    return cls


@contextlib.contextmanager
def _yolov5_pickle_modules():
    """Temporarily provide only the class names referenced by v7.0 weights."""
    package = types.ModuleType("models")
    package.__path__ = []
    common = types.ModuleType("models.common")
    yolo = types.ModuleType("models.yolo")
    package.common = common
    package.yolo = yolo

    safe = []
    for name in ("Conv", "C3", "Bottleneck", "SPPF", "Concat"):
        safe.append(_placeholder(common, name))
    for name in ("Model", "Detect"):
        safe.append(_placeholder(yolo, name))
    safe.extend((
        set,
        collections.OrderedDict,
        torch.nn.Sequential,
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.SiLU,
        torch.nn.MaxPool2d,
        torch.nn.Upsample,
        torch.nn.ModuleList,
    ))

    names = ("models", "models.common", "models.yolo")
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules.update({
        "models": package,
        "models.common": common,
        "models.yolo": yolo,
    })
    try:
        yield safe
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def load_official_yolov5(checkpoint_path: Path, *,
                         variant: str = "s") -> torch.nn.Module:
    """Load and structurally validate a pinned official YOLOv5 v7.0 model."""
    profile = get_yolov5_variant(variant)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    actual = sha256_file(checkpoint_path)
    if actual != profile.checkpoint_sha256:
        raise RuntimeError(
            f"refusing to load unverified checkpoint {checkpoint_path}: "
            f"got SHA-256 {actual}, expected {profile.checkpoint_sha256} "
            f"for {profile.model_name} v7.0")

    with _yolov5_pickle_modules() as safe:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            # PyTorch 2.4 is the oldest supported version.  Older builds do not
            # expose the allow-list context, so the verified official artifact
            # uses the legacy loader while the temporary modules are installed.
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False)
        else:
            with safe_globals(safe):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("YOLOv5 checkpoint is not a dictionary")
    model = checkpoint.get("ema") or checkpoint.get("model")
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("YOLOv5 checkpoint has neither an EMA nor model module")
    model = model.float().eval()
    model.requires_grad_(False)

    modules = list(getattr(model, "model", ()))
    expected = (
        "Conv", "Conv", "C3", "Conv", "C3", "Conv", "C3", "Conv",
        "C3", "SPPF", "Conv", "Upsample", "Concat", "C3", "Conv",
        "Upsample", "Concat", "C3", "Conv", "Concat", "C3", "Conv",
        "Concat", "C3", "Detect",
    )
    actual_types = tuple(type(m).__name__ for m in modules)
    if actual_types != expected:
        raise RuntimeError(
            f"checkpoint topology is not canonical {profile.model_name} v7.0: "
            f"got {actual_types}")
    if getattr(model, "yaml_file", None) != profile.yaml_file:
        raise RuntimeError(
            f"checkpoint identifies {getattr(model, 'yaml_file', None)!r}, "
            f"expected {profile.yaml_file!r}")
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != profile.parameter_count:
        raise RuntimeError(
            f"unexpected {profile.model_name} parameter count {parameter_count}; "
            f"expected {profile.parameter_count}")
    strides = tuple(int(v) for v in model.stride.tolist())
    if strides != (8, 16, 32):
        raise RuntimeError(f"unexpected detection strides {strides}")
    if len(model.names) != 80:
        raise RuntimeError(f"expected 80 COCO classes, got {len(model.names)}")
    detect_channels = tuple(int(conv.in_channels) for conv in modules[-1].m)
    if detect_channels != profile.detect_input_channels:
        raise RuntimeError(
            f"unexpected {profile.model_name} Detect inputs {detect_channels}; "
            f"expected {profile.detect_input_channels}")
    return model


def load_official_yolov5s(checkpoint_path: Path) -> torch.nn.Module:
    """Compatibility wrapper for the pinned official YOLOv5s checkpoint."""
    return load_official_yolov5(checkpoint_path, variant="s")


def load_official_yolov5n(checkpoint_path: Path) -> torch.nn.Module:
    """Load the pinned official YOLOv5n v7.0 checkpoint."""
    return load_official_yolov5(checkpoint_path, variant="n")


def _pair(value, name: str) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"{name} must contain two values, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def fold_conv_bn(conv: torch.nn.Conv2d,
                 bn: Optional[torch.nn.BatchNorm2d]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return float32 Conv2d weights/bias with an optional BatchNorm folded."""
    weight = conv.weight.detach().cpu().float()
    conv_bias = (torch.zeros(conv.out_channels, dtype=torch.float32)
                 if conv.bias is None else conv.bias.detach().cpu().float())
    if bn is None:
        bias = None if conv.bias is None else conv_bias
        return weight.contiguous(), None if bias is None else bias.contiguous()

    gamma = bn.weight.detach().cpu().float()
    beta = bn.bias.detach().cpu().float()
    mean = bn.running_mean.detach().cpu().float()
    variance = bn.running_var.detach().cpu().float()
    factor = gamma * torch.rsqrt(variance + float(bn.eps))
    weight = weight * factor[:, None, None, None]
    bias = beta + factor * (conv_bias - mean)
    return weight.contiguous(), bias.contiguous()


def _unpack_nibbles(data: bytes, rows: int) -> torch.Tensor:
    packed = torch.frombuffer(bytearray(data), dtype=torch.uint8).view(rows, 32)
    codes = torch.empty(rows, 64, dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    return codes


def _bf16_from_bytes(data: bytes) -> torch.Tensor:
    bits = torch.frombuffer(bytearray(data), dtype=torch.uint16).clone()
    return bits.view(torch.bfloat16)


@dataclass
class QuantizedConv:
    """One folded convolution in an Andromeda native quantized layout."""

    codes: torch.Tensor
    block_scales: torch.Tensor
    bias: Optional[torch.Tensor]
    data_type: user_dma_core.TYPE = user_dma_core.TYPE.IF4
    gather: bool = False
    dequant_weight: Optional[torch.Tensor] = None


def quantize_conv_if4(conv: torch.nn.Conv2d,
                      bn: Optional[torch.nn.BatchNorm2d], *,
                      include_dequant: bool = False) -> QuantizedConv:
    """Fold BN and MixMSE-quantize each ``[oc,ky,kx,ct]`` 64-value block."""
    if conv.groups != 1:
        raise ValueError(f"grouped convolution is unsupported (groups={conv.groups})")
    weight, bias = fold_conv_bn(conv, bn)
    oc, channels, kh, kw = weight.shape
    ct = (channels + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
    padded = torch.zeros(
        oc, ct * user_dma_core.UE_VECTOR_SIZE, kh, kw, dtype=torch.float32)
    padded[:, :channels] = weight
    blocks = (padded.view(oc, ct, user_dma_core.UE_VECTOR_SIZE, kh, kw)
              .permute(0, 3, 4, 1, 2).contiguous())
    flat_blocks = blocks.view(-1, user_dma_core.UE_VECTOR_SIZE)

    data_bytes, scale_bytes = quant_lib.quantize_if4(
        flat_blocks, block_size=user_dma_core.UE_VECTOR_SIZE,
        int_variant=None)
    raw_blocks = _unpack_nibbles(data_bytes, flat_blocks.shape[0])
    codes = (raw_blocks.view(oc, kh, kw, ct, user_dma_core.UE_VECTOR_SIZE)
             .permute(0, 3, 4, 1, 2).contiguous()
             .view(oc, ct * user_dma_core.UE_VECTOR_SIZE, kh, kw)
             [:, :channels].contiguous())
    scales = _bf16_from_bytes(scale_bytes).view(oc, kh * kw * ct).contiguous()

    dequant_weight = None
    if include_dequant:
        dequant_blocks = quant_lib.dequantize_if4(
            data_bytes, scale_bytes, flat_blocks.shape[0],
            user_dma_core.UE_VECTOR_SIZE,
            block_size=user_dma_core.UE_VECTOR_SIZE)
        dequant_weight = (
            dequant_blocks.view(oc, kh, kw, ct, user_dma_core.UE_VECTOR_SIZE)
            .permute(0, 3, 4, 1, 2).contiguous()
            .view(oc, ct * user_dma_core.UE_VECTOR_SIZE, kh, kw)
            [:, :channels].contiguous())

    return QuantizedConv(
        codes=codes,
        block_scales=scales,
        bias=None if bias is None else bias.to(torch.bfloat16).contiguous(),
        data_type=user_dma_core.TYPE.IF4,
        gather=False,
        dequant_weight=dequant_weight,
    )


def quantize_conv_gather_if8(
        conv: torch.nn.Conv2d,
        bn: Optional[torch.nn.BatchNorm2d], *,
        include_dequant: bool = False) -> QuantizedConv:
    """Fold BN and MixMSE-quantize gather blocks in ``[ky,kx,c]`` order.

    IF8 is intentional for the small-channel gather path: reblocking a whole
    image patch into only a few 64-value blocks is much more sensitive to IF4
    error than the normal per-tap channels layout.  The wider codes still cut
    the padded RGB-stem work by an order of magnitude.
    """
    if conv.groups != 1:
        raise ValueError(f"grouped convolution is unsupported (groups={conv.groups})")
    weight, bias = fold_conv_bn(conv, bn)
    oc, channels, kh, kw = weight.shape
    taps = kh * kw * channels
    chunks = (taps + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
    if channels > 255 or chunks > 4:
        raise ValueError(
            f"gather convolution needs C<=255 and <=4 chunks, got C={channels}, "
            f"Kh*Kw*C={taps} ({chunks} chunks)")

    flat = weight.permute(0, 2, 3, 1).reshape(oc, taps)
    padded = torch.zeros(
        oc, chunks * user_dma_core.UE_VECTOR_SIZE, dtype=torch.float32)
    padded[:, :taps] = flat
    data_bytes, scale_bytes = quant_lib.quantize_if8(
        padded, block_size=user_dma_core.UE_VECTOR_SIZE, int_variant=None)
    raw = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).view(
        oc, chunks * user_dma_core.UE_VECTOR_SIZE)
    codes = (raw[:, :taps].view(oc, kh, kw, channels)
             .permute(0, 3, 1, 2).contiguous())
    scales = _bf16_from_bytes(scale_bytes).view(oc, chunks).contiguous()

    dequant_weight = None
    if include_dequant:
        dequant = quant_lib.dequantize_if8(
            data_bytes, scale_bytes, oc,
            chunks * user_dma_core.UE_VECTOR_SIZE,
            block_size=user_dma_core.UE_VECTOR_SIZE)
        dequant_weight = (dequant[:, :taps].view(oc, kh, kw, channels)
                          .permute(0, 3, 1, 2).contiguous())

    return QuantizedConv(
        codes=codes,
        block_scales=scales,
        bias=None if bias is None else bias.to(torch.bfloat16).contiguous(),
        data_type=user_dma_core.TYPE.IF8,
        gather=True,
        dequant_weight=dequant_weight,
    )


def _gather_is_profitable(conv: torch.nn.Conv2d) -> bool:
    """Whether gather reduces 64-lane blocks for this convolution."""
    kh, kw = _pair(conv.kernel_size, "kernel_size")
    channels = int(conv.in_channels)
    patch_taps = kh * kw * channels
    if channels > 255 or patch_taps > 256:
        return False
    gather_blocks = (patch_taps + user_dma_core.UE_VECTOR_SIZE - 1) // \
        user_dma_core.UE_VECTOR_SIZE
    channel_blocks = kh * kw * (
        (channels + user_dma_core.UE_VECTOR_SIZE - 1)
        // user_dma_core.UE_VECTOR_SIZE)
    return gather_blocks < channel_blocks


def quantize_conv_for_andromeda(
        conv: torch.nn.Conv2d,
        bn: Optional[torch.nn.BatchNorm2d], *,
        include_dequant: bool = False) -> QuantizedConv:
    """Choose the fastest accuracy-safe native layout for one YOLO conv."""
    if _gather_is_profitable(conv):
        return quantize_conv_gather_if8(
            conv, bn, include_dequant=include_dequant)
    return quantize_conv_if4(conv, bn, include_dequant=include_dequant)


def dequantize_conv_if4(prepared: QuantizedConv) -> torch.Tensor:
    """Dequantize a stored native-layout convolution without float weights."""
    codes = prepared.codes.detach().cpu().to(torch.uint8)
    oc, channels, kh, kw = codes.shape
    ct = (channels + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
    padded = torch.zeros(
        oc, ct * user_dma_core.UE_VECTOR_SIZE, kh, kw, dtype=torch.uint8)
    padded[:, :channels] = codes
    blocks = (padded.view(oc, ct, user_dma_core.UE_VECTOR_SIZE, kh, kw)
              .permute(0, 3, 4, 1, 2).contiguous()
              .view(-1, user_dma_core.UE_VECTOR_SIZE))
    packed = (blocks[:, 0::2] | (blocks[:, 1::2] << 4)).contiguous()
    data_bytes = packed.numpy().tobytes()
    scale_bytes = (prepared.block_scales.detach().cpu().contiguous()
                   .view(torch.uint8).numpy().tobytes())
    dequantized = quant_lib.dequantize_if4(
        data_bytes, scale_bytes, blocks.shape[0],
        user_dma_core.UE_VECTOR_SIZE,
        block_size=user_dma_core.UE_VECTOR_SIZE)
    return (dequantized.view(
        oc, kh, kw, ct, user_dma_core.UE_VECTOR_SIZE)
        .permute(0, 3, 4, 1, 2).contiguous()
        .view(oc, ct * user_dma_core.UE_VECTOR_SIZE, kh, kw)
        [:, :channels].contiguous())


def dequantize_conv(prepared: QuantizedConv) -> torch.Tensor:
    """Dequantize either native channels-IF4 or gather-IF8 layout."""
    if not prepared.gather and prepared.data_type == user_dma_core.TYPE.IF4:
        return dequantize_conv_if4(prepared)
    if not prepared.gather or prepared.data_type != user_dma_core.TYPE.IF8:
        raise ValueError(
            f"unsupported convolution layout/data type: gather={prepared.gather}, "
            f"data_type={prepared.data_type}")

    codes = prepared.codes.detach().cpu().to(torch.uint8)
    oc, channels, kh, kw = codes.shape
    taps = kh * kw * channels
    chunks = (taps + user_dma_core.UE_VECTOR_SIZE - 1) // user_dma_core.UE_VECTOR_SIZE
    flat = codes.permute(0, 2, 3, 1).reshape(oc, taps)
    padded = torch.zeros(
        oc, chunks * user_dma_core.UE_VECTOR_SIZE, dtype=torch.uint8)
    padded[:, :taps] = flat
    scale_bytes = (prepared.block_scales.detach().cpu().contiguous()
                   .view(torch.uint8).numpy().tobytes())
    dequantized = quant_lib.dequantize_if8(
        padded.numpy().tobytes(), scale_bytes, oc,
        chunks * user_dma_core.UE_VECTOR_SIZE,
        block_size=user_dma_core.UE_VECTOR_SIZE)
    return (dequantized[:, :taps].view(oc, kh, kw, channels)
            .permute(0, 3, 1, 2).contiguous())


def _activation_is_silu(module: torch.nn.Module) -> bool:
    name = type(module).__name__
    if name == "SiLU":
        return True
    if name == "Identity":
        return False
    raise ValueError(f"unsupported YOLO activation {name}")


class TorchBackend:
    """Reference backend for graph and post-processing validation."""

    def __init__(self, *, quantized: bool = False):
        self.quantized = quantized
        self._quantized: dict[int, QuantizedConv] = {}
        self._compiled_dequant: dict[str, torch.Tensor] = {}

    def conv(self, name: str, conv: torch.nn.Conv2d,
             bn: Optional[torch.nn.BatchNorm2d], *, activate: bool) -> torch.Tensor:
        raise AssertionError("TorchBackend.conv needs an activation argument; call conv_tensor")

    def conv_tensor(self, name: str, x: torch.Tensor, conv: torch.nn.Conv2d,
                    bn: Optional[torch.nn.BatchNorm2d], *, activate: bool) -> torch.Tensor:
        if self.quantized:
            prepared = self._quantized.get(id(conv))
            if prepared is None:
                prepared = quantize_conv_for_andromeda(
                    conv, bn, include_dequant=True)
                self._quantized[id(conv)] = prepared
            return self.conv_prepared(
                name, x, prepared,
                stride=_pair(conv.stride, "stride")[0],
                pad=_pair(conv.padding, "padding")[0],
                dilation=_pair(conv.dilation, "dilation")[0],
                activate=activate)

        weight, bias = fold_conv_bn(conv, bn)
        y = F.conv2d(
            x.float().unsqueeze(0), weight, bias,
            stride=conv.stride, padding=conv.padding,
            dilation=conv.dilation, groups=conv.groups)[0]
        if activate:
            y = F.silu(y)
        return y

    def conv_prepared(self, name: str, x: torch.Tensor,
                      prepared: QuantizedConv, *, stride: int, pad: int,
                      dilation: int, activate: bool) -> torch.Tensor:
        weight = prepared.dequant_weight
        if weight is None:
            weight = self._compiled_dequant.get(name)
            if weight is None:
                weight = dequantize_conv(prepared)
                self._compiled_dequant[name] = weight
        bias = None if prepared.bias is None else prepared.bias.float()
        y = F.conv2d(
            x.to(torch.bfloat16).float().unsqueeze(0), weight.float(), bias,
            stride=stride, padding=pad, dilation=dilation)[0]
        if activate:
            y = F.silu(y)
        return y.to(torch.bfloat16)

    def maxpool(self, name: str, x: torch.Tensor, *, kernel: int,
                stride: int, pad: int) -> torch.Tensor:
        y = F.max_pool2d(
            x.float().unsqueeze(0), kernel_size=kernel,
            stride=stride, padding=pad)[0]
        return y.to(torch.bfloat16) if self.quantized else y

    def upsample2x(self, name: str, x: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(x.unsqueeze(0).float(), scale_factor=2.0,
                          mode="nearest")[0]
        return y.to(torch.bfloat16) if self.quantized else y

    def add(self, name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = x.float() + y.float()
        return out.to(torch.bfloat16) if self.quantized else out

    def concat(self, name: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(values), dim=0)


class AndromedaBackend:
    """YOLO operations backed by Andromeda's full-tensor primitives."""

    def __init__(self, ue: user_dma_core.UnifiedEngine, *,
                 known_hw_versions: Iterable[int],
                 allow_unknown_hardware: bool = False,
                 timeout_s: float = 300.0):
        self.ue = ue
        self.timeout_s = timeout_s
        self._quantized: dict[int, QuantizedConv] = {}
        self.cycles: dict[str, int] = collections.defaultdict(int)
        self.instruction_bytes: dict[str, int] = collections.defaultdict(int)
        self.max_params_scratch_bytes = 0
        self.max_tensor_scratch_bytes = 0
        self.max_program_scratch_bytes = 0

        hw_version = int(ue.hw_version) & 0xFFFFFFFF
        known = {int(v) & 0xFFFFFFFF for v in known_hw_versions}
        if hw_version not in known and not allow_unknown_hardware:
            versions = ", ".join(f"0x{v:08x}" for v in sorted(known))
            raise RuntimeError(
                f"FPGA build 0x{hw_version:08x} is not known to implement the "
                f"required optimized CONV2D path. Known builds: "
                f"{versions or '(none)'}. The "
                "repository's shipped cf133b89 image predates these opcodes. "
                "Program a pcie_conv_maxpool build, or pass "
                "--allow-unknown-hardware only after verifying RTL compatibility.")
        self.hw_version = hw_version

    def _temporary_arenas(self, kind: str, call: Callable[[], torch.Tensor], *,
                          cycle_attr: str, instruction_attr: str) -> torch.Tensor:
        params_mark = self.ue._next_params_dram_addr
        tensor_mark = self.ue._tensor_dram_addr
        program_mark = self.ue._next_program_dram_addr
        try:
            result = call()
            self.cycles[kind] += int(getattr(self.ue, cycle_attr, 0))
            self.instruction_bytes[kind] += int(
                getattr(self.ue, instruction_attr, 0))
            self.max_params_scratch_bytes = max(
                self.max_params_scratch_bytes,
                self.ue._next_params_dram_addr - params_mark)
            self.max_tensor_scratch_bytes = max(
                self.max_tensor_scratch_bytes,
                self.ue._tensor_dram_addr - tensor_mark)
            self.max_program_scratch_bytes = max(
                self.max_program_scratch_bytes,
                self.ue._next_program_dram_addr - program_mark)
            return result
        finally:
            # Every layer helper has completed its DMA readback before returning,
            # so all three arenas are scratch and may be reused by the next node.
            self.ue._next_params_dram_addr = params_mark
            self.ue._tensor_dram_addr = tensor_mark
            self.ue._next_program_dram_addr = program_mark
            self.ue.is_capture_on = False
            self.ue.clear_capture_buffer()
            self.ue.reset_isa_reg_counter()
            self.ue.reset_inst_ptr_counter()

    def conv_tensor(self, name: str, x: torch.Tensor, conv: torch.nn.Conv2d,
                    bn: Optional[torch.nn.BatchNorm2d], *, activate: bool) -> torch.Tensor:
        prepared = self._quantized.get(id(conv))
        if prepared is None:
            prepared = quantize_conv_for_andromeda(conv, bn)
            self._quantized[id(conv)] = prepared

        kh, kw = _pair(conv.kernel_size, "kernel_size")
        sh, sw = _pair(conv.stride, "stride")
        ph, pw = _pair(conv.padding, "padding")
        dh, dw = _pair(conv.dilation, "dilation")
        if kh != kw or sh != sw or ph != pw or dh != dw:
            raise ValueError(
                f"{name}: Andromeda YOLO path requires square symmetric Conv2d, "
                f"got k={conv.kernel_size}, s={conv.stride}, p={conv.padding}, "
                f"d={conv.dilation}")
        if conv.groups != 1:
            raise ValueError(f"{name}: grouped convolution is unsupported")

        return self.conv_prepared(
            name, x, prepared, stride=sh, pad=ph,
            dilation=dh, activate=activate)

    def conv_prepared(self, name: str, x: torch.Tensor,
                      prepared: QuantizedConv, *, stride: int, pad: int,
                      dilation: int, activate: bool) -> torch.Tensor:

        return self._temporary_arenas(
            "conv",
            lambda: self.ue.run_conv2d_layer(
                x.to(torch.bfloat16), prepared.codes,
                stride_s=stride, pad=pad, dilation=dilation,
                block_scales=prepared.block_scales,
                bias=prepared.bias,
                silu_enable=activate,
                data_type=prepared.data_type,
                gather=prepared.gather,
                timeout_s=self.timeout_s),
            cycle_attr="last_conv_cycles",
            instruction_attr="last_conv_inst_bytes")

    def maxpool(self, name: str, x: torch.Tensor, *, kernel: int,
                stride: int, pad: int) -> torch.Tensor:
        return self._temporary_arenas(
            "maxpool",
            lambda: self.ue.run_maxpool2d_layer(
                x.to(torch.bfloat16), kernel=kernel,
                stride_s=stride, pad=pad, timeout_s=self.timeout_s),
            cycle_attr="last_maxpool_cycles",
            instruction_attr="last_maxpool_inst_bytes")

    def upsample2x(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self._temporary_arenas(
            "upsample",
            lambda: self.ue.run_nn_upsample_2x(
                x.to(torch.bfloat16), timeout_s=self.timeout_s),
            cycle_attr="last_upsample_cycles",
            instruction_attr="last_upsample_inst_bytes")

    def add(self, name: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._temporary_arenas(
            "add",
            lambda: self.ue.run_eltwise_add_layer(
                x.to(torch.bfloat16), y.to(torch.bfloat16),
                timeout_s=self.timeout_s),
            cycle_attr="last_eltwise_cycles",
            instruction_attr="last_eltwise_inst_bytes")

    def concat(self, name: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tuple(values), dim=0)


def _run_standard_conv(module: torch.nn.Module, x: torch.Tensor,
                       backend, name: str) -> torch.Tensor:
    activate = _activation_is_silu(module.act)
    return backend.conv_tensor(
        name, x, module.conv, module.bn, activate=activate)


def _run_bottleneck(module: torch.nn.Module, x: torch.Tensor,
                    backend, name: str) -> torch.Tensor:
    branch = _run_standard_conv(module.cv1, x, backend, f"{name}.cv1")
    branch = _run_standard_conv(module.cv2, branch, backend, f"{name}.cv2")
    return backend.add(f"{name}.add", x, branch) if bool(module.add) else branch


def _run_c3(module: torch.nn.Module, x: torch.Tensor,
            backend, name: str) -> torch.Tensor:
    left = _run_standard_conv(module.cv1, x, backend, f"{name}.cv1")
    for index, bottleneck in enumerate(module.m):
        left = _run_bottleneck(
            bottleneck, left, backend, f"{name}.m.{index}")
    right = _run_standard_conv(module.cv2, x, backend, f"{name}.cv2")
    return _run_standard_conv(
        module.cv3, backend.concat(f"{name}.cat", (left, right)),
        backend, f"{name}.cv3")


def _run_sppf(module: torch.nn.Module, x: torch.Tensor,
              backend, name: str) -> torch.Tensor:
    x = _run_standard_conv(module.cv1, x, backend, f"{name}.cv1")
    kernel = _pair(module.m.kernel_size, "SPPF kernel")[0]
    stride = _pair(module.m.stride, "SPPF stride")[0]
    pad = _pair(module.m.padding, "SPPF padding")[0]
    y1 = backend.maxpool(f"{name}.m.0", x, kernel=kernel,
                         stride=stride, pad=pad)
    y2 = backend.maxpool(f"{name}.m.1", y1, kernel=kernel,
                         stride=stride, pad=pad)
    y3 = backend.maxpool(f"{name}.m.2", y2, kernel=kernel,
                         stride=stride, pad=pad)
    return _run_standard_conv(
        module.cv2, backend.concat(f"{name}.cat", (x, y1, y2, y3)),
        backend, f"{name}.cv2")


def execute_yolov5(model: torch.nn.Module, image_chw: torch.Tensor,
                   backend, *, progress: bool = False) -> list[torch.Tensor]:
    """Execute a canonical YOLOv5 graph and return its raw detection maps."""
    if image_chw.dim() != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected RGB CHW input, got {tuple(image_chw.shape)}")
    outputs: list[torch.Tensor] = []
    current = image_chw

    def source(index):
        if isinstance(index, (list, tuple)):
            return [current if int(i) == -1 else outputs[int(i)] for i in index]
        return current if int(index) == -1 else outputs[int(index)]

    for index, module in enumerate(model.model):
        name = f"model.{index}"
        incoming = source(module.f)
        kind = type(module).__name__
        if kind == "Conv":
            result = _run_standard_conv(module, incoming, backend, name)
        elif kind == "C3":
            result = _run_c3(module, incoming, backend, name)
        elif kind == "SPPF":
            result = _run_sppf(module, incoming, backend, name)
        elif kind == "Upsample":
            if module.mode != "nearest" or float(module.scale_factor) != 2.0:
                raise ValueError(
                    f"{name}: only nearest 2x upsampling is supported")
            result = backend.upsample2x(name, incoming)
        elif kind == "Concat":
            if int(module.d) != 1:
                raise ValueError(f"{name}: expected NCHW channel concat, d={module.d}")
            result = backend.concat(name, incoming)
        elif kind == "Detect":
            raw = []
            for head, (feature, conv) in enumerate(zip(incoming, module.m)):
                raw.append(backend.conv_tensor(
                    f"{name}.m.{head}", feature, conv, None, activate=False))
            if progress:
                print(f"  [{index:02d}] {name:<12} Detect -> "
                      f"{[tuple(v.shape) for v in raw]}")
            return raw
        else:
            raise ValueError(f"{name}: unsupported module type {kind}")

        current = result
        outputs.append(result)
        if progress:
            print(f"  [{index:02d}] {name:<12} {kind:<8} -> {tuple(result.shape)}")

    raise RuntimeError("YOLOv5 graph ended without a Detect module")


def execute_yolov5s(model: torch.nn.Module, image_chw: torch.Tensor,
                    backend, *, progress: bool = False) -> list[torch.Tensor]:
    """Compatibility wrapper for callers that previously targeted YOLOv5s."""
    return execute_yolov5(model, image_chw, backend, progress=progress)


def execute_yolov5n(model: torch.nn.Module, image_chw: torch.Tensor,
                    backend, *, progress: bool = False) -> list[torch.Tensor]:
    """Compatibility wrapper for explicit YOLOv5n callers."""
    return execute_yolov5(model, image_chw, backend, progress=progress)


@dataclass(frozen=True)
class LetterboxInfo:
    original_height: int
    original_width: int
    ratio: float
    pad_left: int
    pad_top: int


def letterbox_image(path: Path, image_size: int = 256) -> tuple[Image.Image, torch.Tensor, LetterboxInfo]:
    """Load RGB, preserve aspect ratio, and pad to a square with value 114."""
    if image_size <= 0 or image_size % 32:
        raise ValueError(f"image_size must be a positive multiple of 32, got {image_size}")
    with Image.open(path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = original.size
    ratio = min(image_size / height, image_size / width)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    resized = original.resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR)
    pad_x = image_size - resized_width
    pad_y = image_size - resized_height
    left = round(pad_x / 2 - 0.1)
    top = round(pad_y / 2 - 0.1)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (left, top))

    pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
    tensor = (pixels.view(image_size, image_size, 3)
              .permute(2, 0, 1).contiguous().float() / 255.0)
    return original, tensor, LetterboxInfo(
        original_height=height, original_width=width,
        ratio=ratio, pad_left=left, pad_top=top)


def _class_names(model: torch.nn.Module) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[i]) for i in range(len(names))]
    return [str(name) for name in names]


def decode_yolov5(raw_heads: Sequence[torch.Tensor], model: torch.nn.Module) -> torch.Tensor:
    """Decode three raw heads into dynamic ``(anchors, 85)`` prediction rows."""
    detect = model.model[-1]
    anchors = detect.anchors.detach().cpu().float()
    strides = model.stride.detach().cpu().float()
    num_classes = len(_class_names(model))
    outputs = []
    for level, raw in enumerate(raw_heads):
        channels, height, width = raw.shape
        anchors_per_level = anchors.shape[1]
        values_per_anchor = num_classes + 5
        expected = anchors_per_level * values_per_anchor
        if channels != expected:
            raise ValueError(
                f"head {level} has {channels} channels, expected {expected}")
        prediction = (raw.float().view(
            anchors_per_level, values_per_anchor, height, width)
            .permute(0, 2, 3, 1).contiguous().sigmoid())
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32), indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0) - 0.5
        anchor_grid = (anchors[level] * strides[level]).view(
            anchors_per_level, 1, 1, 2)
        xy = (prediction[..., 0:2] * 2.0 + grid) * strides[level]
        wh = (prediction[..., 2:4] * 2.0).square() * anchor_grid
        decoded = torch.cat((xy, wh, prediction[..., 4:]), dim=-1)
        outputs.append(decoded.view(-1, values_per_anchor))
    return torch.cat(outputs, dim=0)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(boxes)
    result[:, 0:2] = boxes[:, 0:2] - boxes[:, 2:4] / 2
    result[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4] / 2
    return result


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area_a = (box[2:] - box[:2]).clamp(min=0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (area_a + area_b - intersection).clamp(min=1e-9)


def greedy_nms(boxes: torch.Tensor, scores: torch.Tensor,
               iou_threshold: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    kept = []
    while order.numel():
        first = order[0]
        kept.append(first)
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[
            box_iou_one_to_many(boxes[first], boxes[rest]) <= iou_threshold]
    return (torch.stack(kept) if kept
            else torch.empty(0, dtype=torch.long))


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str


def non_max_suppression(decoded: torch.Tensor, model: torch.nn.Module, *,
                        conf_threshold: float = 0.25,
                        iou_threshold: float = 0.45,
                        max_det: int = 300,
                        max_candidates: int = 30000) -> list[Detection]:
    """Apply one-label-per-anchor, class-aware greedy NMS."""
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be in [0, 1]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_det <= 0:
        raise ValueError("max_det must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    names = _class_names(model)
    class_scores, class_ids = (
        decoded[:, 5:] * decoded[:, 4:5]).max(dim=1)
    keep = class_scores >= conf_threshold
    if not keep.any():
        return []
    boxes = xywh_to_xyxy(decoded[keep, :4])
    scores = class_scores[keep]
    class_ids = class_ids[keep]
    if scores.numel() > max_candidates:
        top = scores.topk(max_candidates).indices
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

    selected = []
    for class_id in class_ids.unique(sorted=True):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        local = greedy_nms(boxes[indices], scores[indices], iou_threshold)
        selected.append(indices[local])
    selected_indices = torch.cat(selected)
    selected_indices = selected_indices[
        scores[selected_indices].argsort(descending=True)[:max_det]]
    return [
        Detection(
            box=tuple(float(v) for v in boxes[index]),
            score=float(scores[index]),
            class_id=int(class_ids[index]),
            label=names[int(class_ids[index])],
        )
        for index in selected_indices
    ]


def restore_boxes(detections: Sequence[Detection],
                  info: LetterboxInfo) -> list[Detection]:
    """Map letterboxed coordinates back to the original image."""
    restored = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        x1 = max(0.0, min(info.original_width, (x1 - info.pad_left) / info.ratio))
        x2 = max(0.0, min(info.original_width, (x2 - info.pad_left) / info.ratio))
        y1 = max(0.0, min(info.original_height, (y1 - info.pad_top) / info.ratio))
        y2 = max(0.0, min(info.original_height, (y2 - info.pad_top) / info.ratio))
        restored.append(Detection(
            box=(x1, y1, x2, y2), score=detection.score,
            class_id=detection.class_id, label=detection.label))
    return restored


def draw_detections(image: Image.Image, detections: Sequence[Detection],
                    output_path: Path) -> None:
    """Write a simple annotated RGB image."""
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = (
            (37 * detection.class_id + 71) % 256,
            (17 * detection.class_id + 149) % 256,
            (29 * detection.class_id + 211) % 256,
        )
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        caption = f"{detection.label} {detection.score:.2f}"
        text_box = draw.textbbox((x1, y1), caption, font=font)
        text_height = text_box[3] - text_box[1]
        text_width = text_box[2] - text_box[0]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, label_y, x1 + text_width + 4,
                        label_y + text_height + 4), fill=color)
        draw.text((x1 + 2, label_y + 2), caption, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
