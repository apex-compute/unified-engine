#!/usr/bin/env python3
"""
MobileNetV2-SSD-FPNLite (640x640) — CPU pythonic implementation.

Architecture follows the TensorFlow Object Detection API config
`ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8`:

  Backbone: MobileNetV2-1.0 (input-size-agnostic)
            taps C3 (stride 8, 32ch), C4 (stride 16, 96ch), C5 (stride 32, 320ch)
  Neck:     FPN-Lite, 128 channels, depthwise-separable 3x3 smoothing,
            top-down add + nearest upsample, two extra levels P6/P7 from P5.
  Head:     SSDLite shared cls+box, depthwise-separable, 6 anchors/loc,
            91 classes (90 COCO + background).

This file only builds the network and runs a forward pass to validate
output shapes. Weight conversion from the TF checkpoint, anchor generation,
box decode, and NMS are next steps.

Backbone primitives are reused from mobilenetv2_cpu_test.py (same dir).

Usage:
    python mobilenetv2_ssd_fpnlite_640_cpu_test.py
    python mobilenetv2_ssd_fpnlite_640_cpu_test.py --size 1056
"""

import argparse
import os
import sys
import tarfile
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mobilenetv2_cpu_test import ConvBNReLU6, InvertedResidual, _load_config

# TF Object Detection Zoo checkpoint (CC-BY 4.0). Same arch we mirror here.
TF_CKPT_NAME = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
TF_CKPT_URL = (
    "http://download.tensorflow.org/models/object_detection/tf2/"
    "20200711/" + TF_CKPT_NAME + ".tar.gz"
)
BIN_DIR = os.path.join(SCRIPT_DIR, "mobilenetv2_ssd_fpnlite_bin")


def _ensure_tf_checkpoint() -> str:
    """Download + extract TF SSD-MBV2-FPNLite-640 checkpoint if not present.

    Returns the prefix path passed to tf.train.load_checkpoint (without
    .index / .data-... suffix).
    """
    ckpt_dir = os.path.join(BIN_DIR, TF_CKPT_NAME, "checkpoint")
    ckpt_prefix = os.path.join(ckpt_dir, "ckpt-0")
    if os.path.exists(ckpt_prefix + ".index"):
        return ckpt_prefix

    os.makedirs(BIN_DIR, exist_ok=True)
    tar_path = os.path.join(BIN_DIR, TF_CKPT_NAME + ".tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading {TF_CKPT_URL} -> {tar_path}")
        urllib.request.urlretrieve(TF_CKPT_URL, tar_path)
        print("Download complete.")

    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(BIN_DIR)
    assert os.path.exists(ckpt_prefix + ".index"), f"missing {ckpt_prefix}.index after extract"
    return ckpt_prefix


# ---------------------------------------------------------------------------
# Multi-scale MobileNetV2 backbone (returns C3, C4, C5)
# ---------------------------------------------------------------------------

class MobileNetV2BackboneSSD(nn.Module):
    """MBV2-1.0 with head conv. Returns C3 (32ch s8), C4 (96ch s16), C5 (1280ch s32).

    TF SSD-FPNLite uses the post-Conv_1 (1280ch) tensor as C5, not the raw block-16
    (320ch) output. We include the head conv and tap its output for C5.

    Block index -> post-block stride / channels:
        blocks[ 0]  16ch  stride 2
        blocks[1-2] 24ch  stride 4
        blocks[3-5] 32ch  stride 8     <- C3 (after block 5)
        blocks[6-9] 64ch  stride 16
        blocks[10-12] 96ch stride 16   <- C4 (after block 12)
        blocks[13-15] 160ch stride 32
        blocks[16]  320ch stride 32
        head:       1280ch stride 32   <- C5
    """

    C3_BLOCK_IDX = 5
    C4_BLOCK_IDX = 12
    C3_CH, C4_CH, C5_CH = 32, 96, 1280

    def __init__(self, cfg: dict):
        super().__init__()
        b = cfg["backbone"]
        stem_ch = b["stem_out_channels"]
        head_ch = b["head_out_channels"]
        irs = b["inverted_residual_setting"]

        self.stem = ConvBNReLU6(3, stem_ch, kernel_size=3, stride=2)

        blocks = []
        in_ch = stem_ch
        for t, c, n, s in irs:
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
        self.blocks = nn.ModuleList(blocks)
        self.head = ConvBNReLU6(in_ch, head_ch, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c3 = c4 = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.C3_BLOCK_IDX:
                c3 = x
            elif i == self.C4_BLOCK_IDX:
                c4 = x
        c5 = self.head(x)
        return c3, c4, c5


# ---------------------------------------------------------------------------
# TF-style separable conv (matches tf.keras.layers.SeparableConv2D)
# ---------------------------------------------------------------------------

class SeparableConvBNReLU6(nn.Module):
    """TF SeparableConv2D + BN + ReLU6.

    Layout: depthwise 3x3 -> pointwise 1x1 -> BN -> ReLU6. Single BN at the end,
    no BN/activation between dw and pw (this matches TF Keras SeparableConv2D,
    NOT the MBV2 inverted-residual depthwise-separable pattern).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self._k = kernel_size
        self._use_tf_pad = stride > 1 and kernel_size > 1
        sym = 0 if self._use_tf_pad else (kernel_size - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride, sym, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_tf_pad:
            x = F.pad(x, (0, self._k - 1, 0, self._k - 1))
        return F.relu6(self.bn(self.pw(self.dw(x))))


# ---------------------------------------------------------------------------
# FPN-Lite neck
# ---------------------------------------------------------------------------

class FPNLiteNeck(nn.Module):
    """TF SSD-FPNLite top-down neck.

    Architecture (matches checkpoint):
      lateral_c5 = Conv1x1(C5,  out_ch) + bias                     # no BN/act
      lateral_c4 = Conv1x1(C4,  out_ch) + bias
      lateral_c3 = Conv1x1(C3,  out_ch) + bias
      P5 = lateral_c5
      P4 = SepConv3x3+BN+ReLU6 ( lateral_c4 + nearest_upsample(P5) )
      P3 = SepConv3x3+BN+ReLU6 ( lateral_c3 + nearest_upsample(P4) )
      P6 = SepConv3x3+BN+ReLU6 ( P5, stride=2 )
      P7 = SepConv3x3+BN+ReLU6 ( P6, stride=2 )

    Note: P5 has no smoothing conv in TF FPNLite.
    """

    def __init__(self, c3_ch: int, c4_ch: int, c5_ch: int, out_ch: int = 128):
        super().__init__()
        self.out_ch = out_ch
        # Laterals: plain 1x1 Conv2D with bias, no BN, no activation.
        self.lat_c5 = nn.Conv2d(c5_ch, out_ch, 1, 1, 0, bias=True)
        self.lat_c4 = nn.Conv2d(c4_ch, out_ch, 1, 1, 0, bias=True)
        self.lat_c3 = nn.Conv2d(c3_ch, out_ch, 1, 1, 0, bias=True)
        # Smoothing (only P4 and P3; not P5).
        self.smooth_p4 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=1)
        self.smooth_p3 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=1)
        # Coarse extra levels.
        self.coarse_p6 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=2)
        self.coarse_p7 = SeparableConvBNReLU6(out_ch, out_ch, kernel_size=3, stride=2)

    def forward(self, c3, c4, c5):
        p5 = self.lat_c5(c5)
        p4 = self.smooth_p4(self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth_p3(self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p6 = self.coarse_p6(p5)
        p7 = self.coarse_p7(p6)
        return [p3, p4, p5, p6, p7]


# ---------------------------------------------------------------------------
# SSDLite WeightSharedConvolutionalBoxPredictor head
# ---------------------------------------------------------------------------

class SSDLiteHead(nn.Module):
    """WeightSharedConvolutionalBoxPredictor (use_depthwise=True, share_prediction_tower=True).

    Single shared 4-DSConv tower across all FPN levels. Tower convs (depthwise+pointwise)
    are shared; the BN statistics inside the tower are PER LEVEL. After the tower,
    a shared SeparableConv produces cls logits (A*num_classes) and another produces
    box deltas (A*4); both have bias on the pointwise (final) conv and no BN/activation.
    """

    def __init__(self, in_ch: int = 128, num_anchors: int = 6, num_classes: int = 91,
                 n_levels: int = 5, n_tower: int = 4):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.n_levels = n_levels
        self.n_tower = n_tower

        # Shared tower DSConv kernels (one set, applied at every level).
        self.tower_dw = nn.ModuleList(
            [nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False) for _ in range(n_tower)]
        )
        self.tower_pw = nn.ModuleList(
            [nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False) for _ in range(n_tower)]
        )
        # Per-level BN: tower_bn[level][stage]
        self.tower_bn = nn.ModuleList([
            nn.ModuleList([nn.BatchNorm2d(in_ch, eps=1e-3) for _ in range(n_tower)])
            for _ in range(n_levels)
        ])

        # Shared final predictors (DSConv with bias on pw, no BN, no activation).
        self.cls_dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.cls_pw = nn.Conv2d(in_ch, num_anchors * num_classes, 1, 1, 0, bias=True)
        self.box_dw = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.box_pw = nn.Conv2d(in_ch, num_anchors * 4, 1, 1, 0, bias=True)

    def _tower(self, x: torch.Tensor, level: int) -> torch.Tensor:
        for s in range(self.n_tower):
            x = self.tower_pw[s](self.tower_dw[s](x))
            x = self.tower_bn[level][s](x)
            x = F.relu6(x)
        return x

    def forward(self, feats):
        cls_outs, box_outs = [], []
        for level, f in enumerate(feats):
            t = self._tower(f, level)
            cls = self.cls_pw(self.cls_dw(t))   # (B, A*K, H, W)
            box = self.box_pw(self.box_dw(t))   # (B, A*4, H, W)
            B, _, H, W = cls.shape
            cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, self.num_classes)
            box = box.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 4)
            cls_outs.append(cls)
            box_outs.append(box)
        return torch.cat(cls_outs, dim=1), torch.cat(box_outs, dim=1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MobileNetV2_SSD_FPNLite(nn.Module):
    def __init__(self, cfg: dict, fpn_ch: int = 128, num_anchors: int = 6,
                 num_classes: int = 91):
        super().__init__()
        self.backbone = MobileNetV2BackboneSSD(cfg)
        self.neck = FPNLiteNeck(
            c3_ch=self.backbone.C3_CH,
            c4_ch=self.backbone.C4_CH,
            c5_ch=self.backbone.C5_CH,
            out_ch=fpn_ch,
        )
        self.head = SSDLiteHead(in_ch=fpn_ch, num_anchors=num_anchors, num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        feats = self.neck(c3, c4, c5)
        cls, box = self.head(feats)
        return cls, box, feats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pure-Python TF v2 checkpoint reader (no tensorflow dependency).
#
# The TF checkpoint is two files:
#   ckpt-0.index               - LevelDB-style SSTable (uncompressed in TF):
#                                maps variable-name string -> BundleEntryProto bytes.
#   ckpt-0.data-00000-of-00001 - concatenated raw tensor bytes.
#
# We parse the SSTable (footer -> index block -> data blocks -> prefix-compressed
# entries), parse the BundleEntryProto (only the dtype/shape/offset/size fields),
# and slice each tensor out of the data file via numpy.frombuffer.
# ---------------------------------------------------------------------------

import struct
import numpy as np

_TF_MAGIC = 0xdb4775248b80fb57   # LevelDB SSTable kTableMagicNumber
_TF_DTYPE = {
    1:  np.dtype("<f4"),   # DT_FLOAT
    2:  np.dtype("<f8"),   # DT_DOUBLE
    3:  np.dtype("<i4"),   # DT_INT32
    4:  np.dtype("u1"),    # DT_UINT8
    5:  np.dtype("<i2"),   # DT_INT16
    6:  np.dtype("i1"),    # DT_INT8
    9:  np.dtype("<i8"),   # DT_INT64
    19: np.dtype("<f2"),   # DT_HALF
}


def _read_varint(buf, pos):
    shift = 0; result = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 70:
            raise ValueError("varint too long")


def _read_block_handle(buf, pos):
    off, pos = _read_varint(buf, pos)
    sz, pos = _read_varint(buf, pos)
    return off, sz, pos


def _snappy_decompress(data: bytes) -> bytes:
    """Pure-Python Snappy frame-less decoder. See:
       https://github.com/google/snappy/blob/main/format_description.txt
    """
    out_len, pos = _read_varint(data, 0)
    out = bytearray()
    n = len(data)
    while pos < n:
        tag = data[pos]; pos += 1
        kind = tag & 0x03
        if kind == 0:                           # literal
            ll = tag >> 2
            if ll < 60:
                length = ll + 1
            else:
                nbytes = ll - 59
                length = int.from_bytes(data[pos:pos + nbytes], "little") + 1
                pos += nbytes
            out.extend(data[pos:pos + length]); pos += length
        else:                                   # copy
            if kind == 1:                       # 1-byte offset
                length = ((tag >> 2) & 0x07) + 4
                offset = ((tag >> 5) & 0x07) << 8 | data[pos]; pos += 1
            elif kind == 2:                     # 2-byte offset
                length = (tag >> 2) + 1
                offset = int.from_bytes(data[pos:pos + 2], "little"); pos += 2
            else:                               # kind == 3, 4-byte offset
                length = (tag >> 2) + 1
                offset = int.from_bytes(data[pos:pos + 4], "little"); pos += 4
            start = len(out) - offset
            # Spec requires byte-at-a-time copy (overlap is legal & common).
            for i in range(length):
                out.append(out[start + i])
    if len(out) != out_len:
        raise ValueError(f"snappy length mismatch: got {len(out)}, expected {out_len}")
    return bytes(out)


def _read_block(file_bytes: bytes, off: int, sz: int) -> bytes:
    """Read a LevelDB SSTable block, transparently snappy-decompressing if needed.

    The 5-byte trailer immediately follows the block contents: 1 byte compression
    type (0 = none, 1 = snappy) + 4 byte CRC32C (skipped).
    """
    raw = file_bytes[off:off + sz]
    comp = file_bytes[off + sz]
    if comp == 0:
        return raw
    if comp == 1:
        return _snappy_decompress(raw)
    raise ValueError(f"unsupported block compression type {comp}")


def _iter_block_entries(block):
    """Yield (key, value) from a LevelDB SSTable block (prefix-compressed)."""
    n = len(block)
    restart_count = struct.unpack_from("<I", block, n - 4)[0]
    entries_end = n - 4 - 4 * restart_count
    pos = 0
    prev_key = b""
    while pos < entries_end:
        shared, pos = _read_varint(block, pos)
        unshared, pos = _read_varint(block, pos)
        vlen, pos = _read_varint(block, pos)
        key = prev_key[:shared] + block[pos:pos + unshared]
        pos += unshared
        val = block[pos:pos + vlen]
        pos += vlen
        yield key, val
        prev_key = key


def _parse_shape(buf):
    """Parse TensorShapeProto -> tuple of dim sizes."""
    dims = []
    pos = 0; n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field = tag >> 3; wt = tag & 7
        if wt == 2:
            ln, pos = _read_varint(buf, pos)
            if field == 2:  # repeated TensorShapeProto.Dim
                end = pos + ln
                size = 0
                while pos < end:
                    t2, pos = _read_varint(buf, pos)
                    f2 = t2 >> 3; w2 = t2 & 7
                    if w2 == 0:
                        v, pos = _read_varint(buf, pos)
                        if f2 == 1:
                            size = v
                    elif w2 == 2:
                        ln2, pos = _read_varint(buf, pos)
                        pos += ln2  # skip Dim.name
                    else:
                        raise ValueError(f"shape dim wire type {w2}")
                dims.append(size)
            else:
                pos += ln
        elif wt == 0:
            _, pos = _read_varint(buf, pos)
        elif wt == 1:
            pos += 8
        elif wt == 5:
            pos += 4
        else:
            raise ValueError(f"shape wire type {wt}")
    return tuple(dims)


def _parse_bundle_entry(buf):
    """Parse BundleEntryProto -> (dtype, shape, offset, size). Other fields ignored."""
    dtype = 0; offset = 0; size = 0; shape = ()
    pos = 0; n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field = tag >> 3; wt = tag & 7
        if wt == 0:
            val, pos = _read_varint(buf, pos)
            if field == 1: dtype = val
            elif field == 4: offset = val
            elif field == 5: size = val
        elif wt == 2:
            ln, pos = _read_varint(buf, pos)
            if field == 2:
                shape = _parse_shape(bytes(buf[pos:pos + ln]))
            pos += ln
        elif wt == 1:
            pos += 8
        elif wt == 5:
            pos += 4
        else:
            raise ValueError(f"bundle entry wire type {wt}")
    return dtype, shape, offset, size


def _read_tf_checkpoint(prefix: str):
    """Pure-Python TF v2 checkpoint reader.

    Returns {variable_name: numpy_array_with_TF_shape}. No tensorflow import.
    """
    with open(prefix + ".index", "rb") as f:
        idx = f.read()
    with open(prefix + ".data-00000-of-00001", "rb") as f:
        data = f.read()

    if len(idx) < 48:
        raise ValueError("index file too small to contain SSTable footer")
    magic = struct.unpack_from("<Q", idx, len(idx) - 8)[0]
    if magic != _TF_MAGIC:
        raise ValueError(f"bad SSTable magic 0x{magic:x}")
    footer = idx[len(idx) - 48:]
    _meta_off, _meta_sz, p = _read_block_handle(footer, 0)
    idx_off, idx_sz, _ = _read_block_handle(footer, p)

    # The index block lists (last_key_in_block -> BlockHandle) for each data block.
    index_block = _read_block(idx, idx_off, idx_sz)
    data_block_handles = []
    for _key, handle_bytes in _iter_block_entries(index_block):
        off, sz, _ = _read_block_handle(handle_bytes, 0)
        data_block_handles.append((off, sz))

    raw_entries = {}
    for off, sz in data_block_handles:
        block = _read_block(idx, off, sz)
        for key, val in _iter_block_entries(block):
            raw_entries[bytes(key).decode("utf-8")] = bytes(val)

    tensors = {}
    for name, entry in raw_entries.items():
        if not entry:
            continue
        dtype, shape, offset, size = _parse_bundle_entry(entry)
        if dtype not in _TF_DTYPE:
            continue  # skip non-numeric (e.g. DT_STRING object-graph)
        np_dtype = _TF_DTYPE[dtype]
        count = size // np_dtype.itemsize
        arr = np.frombuffer(data, dtype=np_dtype, count=count, offset=offset).copy()
        if shape:
            arr = arr.reshape(shape)
        tensors[name] = arr
    return tensors


# ---------------------------------------------------------------------------
# TF checkpoint -> our PyTorch modules
# ---------------------------------------------------------------------------

def _tf_kernel_to_torch(t):
    """TF Conv2D kernel (H,W,Cin,Cout) -> PyTorch (Cout,Cin,H,W)."""
    return torch.from_numpy(t).permute(3, 2, 0, 1).contiguous()


def _tf_dwkernel_to_torch(t):
    """TF DepthwiseConv2D / SeparableConv2D depthwise kernel
       (H,W,Cin,multiplier=1) -> PyTorch (Cin,1,H,W)."""
    return torch.from_numpy(t).permute(2, 3, 0, 1).contiguous()


def _load_tf_weights(model: "MobileNetV2_SSD_FPNLite", ckpt_prefix: str, verbose: bool = False):
    """Copy TF SSD-MBV2-FPNLite-640 weights into our PyTorch model in-place.

    Uses the pure-Python checkpoint reader above; no tensorflow import required.
    """
    ckpt = _read_tf_checkpoint(ckpt_prefix)
    def G(name):
        return ckpt[name + "/.ATTRIBUTES/VARIABLE_VALUE"]

    # ---------- Backbone ----------
    # MBV2 1.0 layer_with_weights numbering (each conv = 1 idx, each BN = 1 idx):
    #   0: stem conv          1: stem BN
    #   2: blk0.dw_conv       3: blk0.dw_BN
    #   4: blk0.pw_conv       5: blk0.pw_BN
    #   For i in 1..16, base = 6 + (i-1)*6:
    #     base:    blk[i] expand_pw_conv      base+1: BN
    #     base+2:  blk[i] depthwise_conv      base+3: BN
    #     base+4:  blk[i] project_pw_conv     base+5: BN
    #   102: head conv        103: head BN
    fe = "model/_feature_extractor/classification_backbone/layer_with_weights-"

    def cp_conv2d(idx: int, m: nn.Conv2d):
        m.weight.data.copy_(_tf_kernel_to_torch(G(f"{fe}{idx}/kernel")))

    def cp_dwconv2d(idx: int, m: nn.Conv2d):
        m.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{fe}{idx}/depthwise_kernel")))

    def cp_bn(idx: int, m: nn.BatchNorm2d):
        m.weight.data.copy_(torch.from_numpy(G(f"{fe}{idx}/gamma")))
        m.bias.data.copy_(torch.from_numpy(G(f"{fe}{idx}/beta")))
        m.running_mean.data.copy_(torch.from_numpy(G(f"{fe}{idx}/moving_mean")))
        m.running_var.data.copy_(torch.from_numpy(G(f"{fe}{idx}/moving_variance")))

    bb = model.backbone
    cp_conv2d(0, bb.stem.conv); cp_bn(1, bb.stem.bn)

    # blk0: [dw_ConvBNReLU6, project_Conv2d, project_BN]
    blk0 = bb.blocks[0].block
    cp_dwconv2d(2, blk0[0].conv); cp_bn(3, blk0[0].bn)
    cp_conv2d(4, blk0[1]);        cp_bn(5, blk0[2])

    for i in range(1, 17):
        base = 6 + (i - 1) * 6
        blk = bb.blocks[i].block  # [expand_ConvBNReLU6, dw_ConvBNReLU6, project_Conv2d, project_BN]
        cp_conv2d(base + 0, blk[0].conv); cp_bn(base + 1, blk[0].bn)
        cp_dwconv2d(base + 2, blk[1].conv); cp_bn(base + 3, blk[1].bn)
        cp_conv2d(base + 4, blk[2]);     cp_bn(base + 5, blk[3])

    cp_conv2d(102, bb.head.conv); cp_bn(103, bb.head.bn)

    # ---------- FPN ----------
    # Laterals: 1x1 with bias.
    n = model.neck
    fpn = "model/_feature_extractor/_fpn_features_generator"
    # top_layers/0 = lateral for C5 (1280->128).
    n.lat_c5.weight.data.copy_(_tf_kernel_to_torch(G(f"{fpn}/top_layers/0/kernel")))
    n.lat_c5.bias.data.copy_(torch.from_numpy(G(f"{fpn}/top_layers/0/bias")))
    # residual_blocks/0 = lateral for C4 (96->128); /1 = lateral for C3 (32->128).
    n.lat_c4.weight.data.copy_(_tf_kernel_to_torch(G(f"{fpn}/residual_blocks/0/0/kernel")))
    n.lat_c4.bias.data.copy_(torch.from_numpy(G(f"{fpn}/residual_blocks/0/0/bias")))
    n.lat_c3.weight.data.copy_(_tf_kernel_to_torch(G(f"{fpn}/residual_blocks/1/0/kernel")))
    n.lat_c3.bias.data.copy_(torch.from_numpy(G(f"{fpn}/residual_blocks/1/0/bias")))

    # conv_layers/0 = smooth P4; conv_layers/1 = smooth P3.
    def cp_sepconv(prefix: str, m: SeparableConvBNReLU6):
        m.dw.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{prefix}/0/depthwise_kernel")))
        m.pw.weight.data.copy_(_tf_kernel_to_torch(G(f"{prefix}/0/pointwise_kernel")))
        m.bn.weight.data.copy_(torch.from_numpy(G(f"{prefix}/1/gamma")))
        m.bn.bias.data.copy_(torch.from_numpy(G(f"{prefix}/1/beta")))
        m.bn.running_mean.data.copy_(torch.from_numpy(G(f"{prefix}/1/moving_mean")))
        m.bn.running_var.data.copy_(torch.from_numpy(G(f"{prefix}/1/moving_variance")))

    cp_sepconv(f"{fpn}/conv_layers/0", n.smooth_p4)
    cp_sepconv(f"{fpn}/conv_layers/1", n.smooth_p3)
    # Coarse: _coarse_feature_layers/0 -> P6, /1 -> P7.
    coarse = "model/_feature_extractor/_coarse_feature_layers"
    cp_sepconv(f"{coarse}/0", n.coarse_p6)
    cp_sepconv(f"{coarse}/1", n.coarse_p7)

    # ---------- Head (tower + final cls/box) ----------
    bp = "model/_box_predictor"
    h = model.head
    # Shared tower DSConv kernels (depthwise + pointwise) per stage 0..3.
    for s in range(h.n_tower):
        h.tower_dw[s].weight.data.copy_(
            _tf_dwkernel_to_torch(G(f"{bp}/_head_scope_conv_layers/PredictionTower/{s}/depthwise_kernel"))
        )
        h.tower_pw[s].weight.data.copy_(
            _tf_kernel_to_torch(G(f"{bp}/_head_scope_conv_layers/PredictionTower/{s}/pointwise_kernel"))
        )
    # Per-level BN: levels 0..4 (P3..P7), BN slot indices 1, 4, 7, 10 (one per tower stage).
    bn_slots = [1, 4, 7, 10]
    for lvl in range(h.n_levels):
        for s, slot in enumerate(bn_slots):
            bn = h.tower_bn[lvl][s]
            base = f"{bp}/_base_tower_layers_for_heads/box_encodings/{lvl}/{slot}"
            bn.weight.data.copy_(torch.from_numpy(G(f"{base}/gamma")))
            bn.bias.data.copy_(torch.from_numpy(G(f"{base}/beta")))
            bn.running_mean.data.copy_(torch.from_numpy(G(f"{base}/moving_mean")))
            bn.running_var.data.copy_(torch.from_numpy(G(f"{base}/moving_variance")))

    # Final cls predictor: dw + pw + bias.
    cls_p = f"{bp}/_prediction_heads/class_predictions_with_background/_class_predictor_layers/0"
    h.cls_dw.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{cls_p}/depthwise_kernel")))
    h.cls_pw.weight.data.copy_(_tf_kernel_to_torch(G(f"{cls_p}/pointwise_kernel")))
    h.cls_pw.bias.data.copy_(torch.from_numpy(G(f"{cls_p}/bias")))
    # Final box predictor.
    box_p = f"{bp}/_box_prediction_head/_box_encoder_layers/0"
    h.box_dw.weight.data.copy_(_tf_dwkernel_to_torch(G(f"{box_p}/depthwise_kernel")))
    h.box_pw.weight.data.copy_(_tf_kernel_to_torch(G(f"{box_p}/pointwise_kernel")))
    h.box_pw.bias.data.copy_(torch.from_numpy(G(f"{box_p}/bias")))

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded TF weights into model: {n_params/1e6:.2f}M params")


# ---------------------------------------------------------------------------
# SSD postprocess: anchors, box decode, NMS
# ---------------------------------------------------------------------------

# Matches `multiscale_anchor_generator` in pipeline.config.
_ANCHOR_LEVELS = (3, 4, 5, 6, 7)
_ASPECT_RATIOS = (1.0, 2.0, 0.5)
_SCALES_PER_OCTAVE = 2
_ANCHOR_SCALE = 4.0
# Faster R-CNN box coder scale factors (y, x, h, w).
_BOX_SCALES = (10.0, 10.0, 5.0, 5.0)


def generate_anchors(image_size: int) -> torch.Tensor:
    """Generate multiscale SSD anchors in input-pixel coords.

    Returns: (N, 4) tensor of (cy, cx, h, w). Anchor order matches the head's
    flattened output: levels P3..P7, then row-major (i, j), then anchor index.
    """
    octave_scales = [2 ** (k / _SCALES_PER_OCTAVE) for k in range(_SCALES_PER_OCTAVE)]
    anchors = []
    for level in _ANCHOR_LEVELS:
        stride = 1 << level                   # 8, 16, 32, 64, 128
        base = _ANCHOR_SCALE * stride         # 32, 64, 128, 256, 512
        feat_size = image_size // stride
        # Anchor center grid (cy, cx) in input-pixel coords.
        yy = (torch.arange(feat_size, dtype=torch.float32) + 0.5) * stride
        xx = (torch.arange(feat_size, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(yy, xx, indexing="ij")
        cy = cy.reshape(-1, 1)                # (HW, 1)
        cx = cx.reshape(-1, 1)
        # Each anchor: (scale * sqrt(ratio), scale / sqrt(ratio)) for (w, h).
        hs, ws = [], []
        for octave in octave_scales:
            for ratio in _ASPECT_RATIOS:
                hs.append(base * octave / (ratio ** 0.5))
                ws.append(base * octave * (ratio ** 0.5))
        # Order TF uses: outer loop aspect ratio? Actually multiscale_anchor_generator
        # iterates `for scale in scales: for aspect in aspect_ratios:` which matches above.
        h = torch.tensor(hs).reshape(1, -1)   # (1, A)
        w = torch.tensor(ws).reshape(1, -1)
        # Broadcast to (HW, A) then flatten.
        cy = cy.expand(-1, h.shape[1])
        cx = cx.expand(-1, w.shape[1])
        h = h.expand(cy.shape[0], -1)
        w = w.expand(cx.shape[0], -1)
        anchors.append(torch.stack([cy, cx, h, w], dim=-1).reshape(-1, 4))
    return torch.cat(anchors, dim=0)


def decode_boxes(box_deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode (ty, tx, th, tw) deltas against (cy, cx, h, w) anchors.

    Returns (N, 4) boxes as (ymin, xmin, ymax, xmax) in input-pixel coords.
    """
    ty = box_deltas[..., 0] / _BOX_SCALES[0]
    tx = box_deltas[..., 1] / _BOX_SCALES[1]
    th = box_deltas[..., 2] / _BOX_SCALES[2]
    tw = box_deltas[..., 3] / _BOX_SCALES[3]
    cy_a, cx_a, h_a, w_a = anchors.unbind(-1)
    cy = ty * h_a + cy_a
    cx = tx * w_a + cx_a
    h = torch.exp(th) * h_a
    w = torch.exp(tw) * w_a
    ymin = cy - 0.5 * h
    xmin = cx - 0.5 * w
    ymax = cy + 0.5 * h
    xmax = cx + 0.5 * w
    return torch.stack([ymin, xmin, ymax, xmax], dim=-1)


def nms_detections(boxes: torch.Tensor, scores: torch.Tensor,
                   score_thresh: float = 0.3, iou_thresh: float = 0.6,
                   max_total: int = 100):
    """Per-class NMS. scores: (N, K) for K foreground classes. Returns lists."""
    from torchvision.ops import nms
    keep_boxes, keep_scores, keep_labels = [], [], []
    K = scores.shape[1]
    for c in range(K):
        s = scores[:, c]
        m = s > score_thresh
        if not m.any():
            continue
        b, s = boxes[m], s[m]
        keep = nms(b, s, iou_thresh)
        keep_boxes.append(b[keep])
        keep_scores.append(s[keep])
        keep_labels.append(torch.full((keep.numel(),), c + 1, dtype=torch.int64))  # +1: skip background
    if not keep_boxes:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
    b = torch.cat(keep_boxes); s = torch.cat(keep_scores); l = torch.cat(keep_labels)
    if s.numel() > max_total:
        top = torch.topk(s, max_total).indices
        b, s, l = b[top], s[top], l[top]
    return b, s, l


# COCO 1..90 class names (TF mscoco_label_map.pbtxt; channel 0 = background).
COCO_LABELS = [
    "background",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _dump_tf_vars(ckpt_prefix: str):
    """Print every TF model variable (name, shape, dtype). One-shot inspection."""
    ckpt = _read_tf_checkpoint(ckpt_prefix)
    keys = sorted(ckpt.keys())
    print(f"TF checkpoint: {len(keys)} model variables (pure-python reader)")
    for k in keys:
        a = ckpt[k]
        print(f"  {k}  {tuple(a.shape)}  {a.dtype.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=640,
                        help="Input image size (square). Supported: 640, 1056.")
    parser.add_argument("--num-classes", type=int, default=91)
    parser.add_argument("--num-anchors", type=int, default=6)
    parser.add_argument("--fpn-ch", type=int, default=128)
    parser.add_argument("--dump-tf-vars", action="store_true",
                        help="Download TF checkpoint (if needed) and dump variable names + shapes.")
    parser.add_argument("--image",
                        default=os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)),
                                             "test_samples", "vette.jpg"),
                        help="Input image path (default: test_samples/vette.jpg)")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.6)
    args = parser.parse_args()

    if args.dump_tf_vars:
        ckpt = _ensure_tf_checkpoint()
        _dump_tf_vars(ckpt)
        return

    cfg = _load_config()
    model = MobileNetV2_SSD_FPNLite(
        cfg,
        fpn_ch=args.fpn_ch,
        num_anchors=args.num_anchors,
        num_classes=args.num_classes,
    ).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV2-SSD-FPNLite ({args.size}x{args.size}): {n_params/1e6:.2f}M params")

    # Auto-load TF weights when present (downloads on first run).
    ckpt = _ensure_tf_checkpoint()
    _load_tf_weights(model, ckpt, verbose=True)

    # Load image (resize to square; no letterboxing — TF model trained on resize).
    from PIL import Image
    import numpy as np
    img = Image.open(args.image).convert("RGB")
    W0, H0 = img.size
    img_r = img.resize((args.size, args.size), Image.BILINEAR)
    arr = np.asarray(img_r, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # MBV2 normalize: [-1, 1]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        _ = model(x)  # warmup
        t0 = time.perf_counter()
        cls, box, feats = model(x)
        t_inf = time.perf_counter() - t0

    print(f"Input  shape: {tuple(x.shape)}  (orig {W0}x{H0})")
    for i, f in enumerate(feats, start=3):
        print(f"  P{i} shape: {tuple(f.shape)}")
    print(f"Inference (pure CPU): {t_inf * 1000:.3f} ms")

    # Postprocess: anchors + decode + per-class NMS.
    anchors = generate_anchors(args.size)
    assert anchors.shape[0] == cls.shape[1], (anchors.shape, cls.shape)
    boxes = decode_boxes(box[0], anchors)               # (N, 4) in input-px coords
    # Skip background channel (class 0) for sigmoid scores.
    scores = torch.sigmoid(cls[0, :, 1:])                # (N, 90)
    keep_boxes, keep_scores, keep_labels = nms_detections(
        boxes, scores,
        score_thresh=args.score_thresh,
        iou_thresh=args.iou_thresh,
    )

    # Rescale boxes from input_size back to original image coords.
    sx = W0 / args.size; sy = H0 / args.size
    print(f"\nDetections (score >= {args.score_thresh}):")
    if keep_scores.numel() == 0:
        print("  (none)")
        order = torch.empty(0, dtype=torch.long)
    else:
        order = torch.argsort(keep_scores, descending=True)
        for idx in order:
            ymin, xmin, ymax, xmax = keep_boxes[idx].tolist()
            label = COCO_LABELS[int(keep_labels[idx])]
            print(f"  {keep_scores[idx].item()*100:5.1f}%  {label:18s}  "
                  f"[{xmin*sx:7.1f}, {ymin*sy:7.1f}, {xmax*sx:7.1f}, {ymax*sy:7.1f}]")

    # Draw boxes on the original-resolution image and save next to this script.
    from PIL import ImageDraw, ImageFont
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  max(12, int(min(W0, H0) * 0.02)))
    except OSError:
        font = ImageFont.load_default()
    # Deterministic color per class (HSV around the wheel).
    import colorsys
    def _color(class_id: int):
        r, g, b = colorsys.hsv_to_rgb((class_id * 0.137) % 1.0, 0.85, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)

    line_w = max(2, int(min(W0, H0) * 0.004))
    for idx in order:
        ymin, xmin, ymax, xmax = keep_boxes[idx].tolist()
        x0, y0 = xmin * sx, ymin * sy
        x1, y1 = xmax * sx, ymax * sy
        cls_id = int(keep_labels[idx])
        color = _color(cls_id)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_w)
        text = f"{COCO_LABELS[cls_id]} {keep_scores[idx].item()*100:.1f}%"
        # Text background for legibility.
        tb = draw.textbbox((x0, y0), text, font=font)
        pad = 2
        draw.rectangle([tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad], fill=color)
        draw.text((x0, y0), text, fill=(0, 0, 0), font=font)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(SCRIPT_DIR, f"{stem}_detections.jpg")
    out.save(out_path, quality=92)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
