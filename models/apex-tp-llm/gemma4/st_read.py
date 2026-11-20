"""Read single tensors from .safetensors files with plain file reads (no mmap of the whole file:
the 50 GB shard does not fit the address-space limit on this host)."""
import json
import os
import struct

import torch

_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64,
       "I32": torch.int32, "U8": torch.uint8, "I8": torch.int8, "BOOL": torch.bool}


class SafeDir:
    def __init__(self, d):
        self.d = d
        self.index = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self._hdr = {}

    def _header(self, fn):
        if fn not in self._hdr:
            with open(os.path.join(self.d, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._hdr[fn] = (8 + n, json.loads(f.read(n)))
        return self._hdr[fn]

    def keys(self):
        return sorted(self.index)

    def info(self, key):
        fn = self.index[key]
        base, h = self._header(fn)
        m = h[key]
        return m["shape"], m["dtype"]

    def get(self, key):
        fn = self.index[key]
        base, h = self._header(fn)
        m = h[key]
        a, b = m["data_offsets"]
        with open(os.path.join(self.d, fn), "rb") as f:
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
        t = torch.frombuffer(buf, dtype=_DT[m["dtype"]]) if b > a else torch.empty(0, dtype=_DT[m["dtype"]])
        return t.reshape(m["shape"])
