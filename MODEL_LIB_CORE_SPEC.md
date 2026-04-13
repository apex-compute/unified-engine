# model_lib_core.py — Implementation Specification (self-contained)

This document is the complete specification for creating `/home/rohit/unified-engine/model_lib_core.py`. It is self-contained: the implementing AI should be able to produce the file and migrate every call site from this document alone, without re-reading the source files whose contents are pasted below. Any ambiguity that remains is flagged with **VERIFY** and must be resolved by direct inspection of the named source lines; guessing is not acceptable.

This spec pairs with `HARDWARE_BEHAVIORS.md` at the repo root. Every chunking loop and every URAM-A/URAM-B address choice in the function bodies below is a consequence of the constraints documented in that file; the implementer should read it once before starting so the rationale for each loop is clear.

---

## 0. Design decisions (locked; do not reinterpret)

1. **File path:** `/home/rohit/unified-engine/model_lib_core.py`, at the repo root, sibling to `user_dma_core.py`.
2. **Suffix convention:** every hardware-touching helper uses the `_core_dram` suffix. There is no bare `_dram` suffix in this library. Functions currently named `*_dram` in the sam codebase (`dram_zero_fill`, `dram_copy`, `eltwise_add_dram`, `broadcast_mul_dram`) are renamed at lift time to `zero_fill_core_dram`, `copy_core_dram`, `eltwise_add_core_dram`, `broadcast_mul_core_dram` respectively. Functions already ending in `_core_dram` (`eltwise_add_core_dram`, `eltwise_mul_core_dram`, `rms_norm_core_dram_post_add`, `layer_norm_core_dram`, `layer_norm_core_dram_post_add`) keep their names.
3. **Leading-underscore names become public:** `_quantize_bf16_to_int4_packed` → `quantize_bf16_to_int4_packed`. No other public helper begins with an underscore.
4. **Fused variants stay as separate functions, not kwargs.** `rms_norm_core_dram_post_add` is its own function; do NOT introduce a `fuse_residual_add=True` parameter on a bare `rms_norm_core_dram`. The `_post_*` suffix pattern is the convention going forward.
5. **Host-side helpers keep their names unchanged** (`store_weight`, `store_quantized_weight`, `load_weight_cache`, `quantize_bf16_to_int4_packed`, `quantize_q4_64`, `quantize_fp4_64`). They do NOT get a `_core_dram` suffix because they are not core-hardware invocations.
6. **No class hierarchy, no registry, no logging, no metrics.** Flat file of top-level functions.
7. **No import-time side effects.** No DMA, no print, no device open on import.
8. **No bug fixes while lifting.** Lift verbatim; if a bug is found, file it and fix in a follow-up commit.

---

## 1. Library file layout

The file is assembled in this exact order:

1. Module docstring (§2)
2. Imports (§3)
3. Module-level constants (§4)
4. Section comment `# --- Host-side weight tooling ---` followed by the six host functions (§5.1–§5.6)
5. Section comment `# --- DRAM elementwise and movement helpers ---` followed by the six DRAM helpers (§6.1–§6.6)
6. Section comment `# --- Normalization helpers ---` followed by the three norm helpers (§7.1–§7.3)
7. Section comment `# --- Attention helper ---` followed by the one attention helper (§8.1)

---

## 2. Module docstring (paste verbatim)

```python
"""Shared helpers for model bring-up, sitting on top of user_dma_core.

Naming convention:
  - Hardware-touching helpers use the `_core_dram` suffix.
  - Fused variants are separate functions with a `_post_<op>` suffix
    (e.g. rms_norm_core_dram_post_add) rather than kwargs on a base function.
  - Host-side helpers (quantization, weight I/O) have no suffix.

See HARDWARE_BEHAVIORS.md at the repo root for the constraints that shape
every chunking loop and URAM-A / URAM-B address choice below.
"""
```

---

## 3. Imports (paste verbatim)

```python
import json

import numpy as np
import torch

from user_dma_core import (
    DMA_DEVICE_H2C,
    URAM_NEAR_FULL_ELEMENTS,
    UnifiedEngine,
)
```

**VERIFY (import completeness):** the above covers every symbol used by the function bodies in §5–§8 as pasted. If during implementation any `NameError` appears, the missing symbol must be added to the `from user_dma_core import (...)` block; do NOT introduce `import user_dma_core` wildcard access.

---

## 4. Module-level constants

### 4.1 `_FP4_E2M1_TABLE`

`quantize_fp4_64` (§5.6) references a module-level tensor named `_FP4_E2M1_TABLE`. This constant is defined in `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` and in `models/smolvlm2/smolvlm2_test.py` as the FP4 E2M1 code table.

**VERIFY:** grep both files for `_FP4_E2M1_TABLE =` and confirm the two definitions are byte-identical. If so, paste the definition into `model_lib_core.py` immediately after the imports. If they differ, stop and surface the difference — the library cannot ship with two candidates. The constant must retain its leading underscore (it is an implementation detail of `quantize_fp4_64`, not part of the public API).

---

## 5. Host-side weight tooling

### 5.1 `store_weight`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 40–53. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 42–55.

```python
def store_weight(ue, tensor, padded_shape=None):
    """Pad, convert to bf16, DMA to device DRAM. Returns DRAM address."""
    bf16 = tensor.to(torch.bfloat16)
    if padded_shape is not None:
        padded = torch.zeros(padded_shape, dtype=torch.bfloat16)
        slices = tuple(slice(0, s) for s in bf16.shape)
        padded[slices] = bf16
        bf16 = padded
    bf16 = bf16.contiguous()
    nbytes = bf16.numel() * 2
    addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, addr, bf16.flatten(), nbytes)
    ue.allocate_params_dram(nbytes)
    return addr
```

### 5.2 `store_quantized_weight`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 55–74. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 56–75.

```python
def store_quantized_weight(ue, raw_data):
    """Store Q4_64 raw GGUF data as scale + packed in DRAM. Returns (scale_addr, data_addr)."""
    raw_bytes = raw_data.tobytes() if hasattr(raw_data, 'tobytes') else bytes(raw_data)
    n_blocks = len(raw_bytes) // 34
    scales_size = n_blocks * 2
    data_size = n_blocks * 32

    scales_np = np.frombuffer(raw_bytes[:scales_size], dtype=np.uint16).copy()
    scale_tensor = torch.from_numpy(scales_np).view(torch.bfloat16)
    scale_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, scale_addr, scale_tensor, scales_size)
    ue.allocate_params_dram(scales_size)

    data_np = np.frombuffer(raw_bytes[scales_size:scales_size + data_size], dtype=np.uint8).copy()
    data_tensor = torch.from_numpy(data_np)
    data_addr = ue.get_params_dram_addr()
    ue.dma_write(DMA_DEVICE_H2C, data_addr, data_tensor, data_size)
    ue.allocate_params_dram(data_size)

    return scale_addr, data_addr
```

### 5.3 `load_weight_cache`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 76–86. Byte-identical copy in `models/smolvlm2/smolvlm2_test.py` lines 76–86 (confirmed by follow-up grep).

```python
def load_weight_cache(bin_path):
    """Load bin+json weight file. Returns {tensor_name: raw_numpy_data}."""
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    with open(json_path) as f:
        manifest = json.load(f)
    with open(bin_path, 'rb') as f:
        raw = f.read()
    cache = {}
    for name, meta in manifest.items():
        cache[name] = np.frombuffer(raw[meta['offset']:meta['offset'] + meta['size']], dtype=np.uint8).copy()
    return cache
```

### 5.4 `quantize_bf16_to_int4_packed` (renamed from `_quantize_bf16_to_int4_packed`)

Source of truth: `models/gemma3/gemma3_numeric.py` lines 66–81. Byte-identical to `models/llama3.2_1b/llama3.2_1b_test.py` lines 71–86 and `models/qwen3_1.7b/qwen3_1.7b_test.py` lines 77–92. **Drop the leading underscore at lift time.**

```python
def quantize_bf16_to_int4_packed(weight_bf16: torch.Tensor, block_size: int = 64) -> tuple[bytes, bytes]:
    """Quantize bf16 weight (N_w, K_w) to INT4 packed + scale per block of 64 along K. Returns (data_bytes, scale_bytes)."""
    w = weight_bf16.detach().cpu().float().reshape(-1)
    N_w, K_w = weight_bf16.shape
    assert K_w % block_size == 0
    w_blocks = w.reshape(N_w, K_w // block_size, block_size)
    scale = w_blocks.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
    scale_bf16 = scale.to(torch.bfloat16)
    w_int8 = (w_blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    w_nibbles = w_int8.numpy().astype(np.int16) & 0x0F
    low = w_nibbles[:, :, 0::2].reshape(N_w, -1)
    high = w_nibbles[:, :, 1::2].reshape(N_w, -1)
    packed = (high << 4) | low
    data_bytes = packed.astype(np.uint8).tobytes()
    scale_bytes = scale_bf16.contiguous().view(torch.uint8).numpy().tobytes()
    return (data_bytes, scale_bytes)
```

### 5.5 `quantize_q4_64`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 363–376. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 1009–1022.

```python
def quantize_q4_64(tensor):
    """INT4 quantization with 64-element blocks. Returns (packed_bytes, n_blocks)."""
    data = tensor.flatten().cpu().float().numpy()
    n_blocks = int(np.ceil(len(data) / 64))
    padded = np.pad(data, (0, n_blocks * 64 - len(data)))
    blocks = padded.reshape(n_blocks, 64)
    scales = np.max(np.abs(blocks), axis=1)
    scales[scales == 0] = 1.0
    scales /= 7.0
    quantized = np.clip(np.round(blocks / scales[:, None]), -8, 7).astype(np.int8)
    pairs = (quantized.astype(np.uint8) & 0x0F).reshape(n_blocks, 32, 2)
    packed = pairs[:, :, 0] | (pairs[:, :, 1] << 4)
    scale_bytes = torch.tensor(scales, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy()
    return np.frombuffer(scale_bytes.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks
```

### 5.6 `quantize_fp4_64`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 378–394. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 1023–1039. Depends on `_FP4_E2M1_TABLE` — see §4.1.

```python
def quantize_fp4_64(tensor):
    """FP4 E2M1 quantization with 64-element blocks. Returns (packed_bytes, n_blocks)."""
    x = tensor.to(torch.bfloat16).cpu().flatten()
    n_blocks = int(np.ceil(x.numel() / 64))
    if x.numel() % 64 != 0:
        x = torch.nn.functional.pad(x, (0, n_blocks * 64 - x.numel()))
    blocks = x.view(n_blocks, 64)
    fp4_max = torch.tensor(6.0, dtype=torch.bfloat16)
    scales = (blocks.abs().max(dim=1).values / fp4_max).clamp(min=1e-8).to(torch.bfloat16)
    scaled = (blocks / scales[:, None]).to(torch.bfloat16)
    codes = torch.argmin(torch.abs(scaled.unsqueeze(-1) - _FP4_E2M1_TABLE), dim=-1).to(torch.uint8)
    codes_np = codes.numpy().flatten()
    if len(codes_np) % 2 != 0:
        codes_np = np.pad(codes_np, (0, 1))
    packed = (codes_np[0::2] & 0x0F) | ((codes_np[1::2] & 0x0F) << 4)
    scales_np = scales.view(torch.uint16).numpy()
    return np.frombuffer(scales_np.tobytes() + packed.tobytes(), dtype=np.uint8), n_blocks
```

---

## 6. DRAM elementwise and movement helpers

All six functions below stage through SRAM using URAM-A at `0x00000` and URAM-B at `0x80000`, chunked at `URAM_NEAR_FULL_ELEMENTS`. This is the pattern mandated by `HARDWARE_BEHAVIORS.md` §3 (128-byte row granularity) and §2 (URAM A/B convention).

### 6.1 `zero_fill_core_dram` (renamed from `dram_zero_fill`)

Source of truth: `models/sam/sam1-vit-b/sam1_vit_b_test.py` lines 361–377. Byte-identical to `models/sam/sam3.1/sam3.1_test.py` lines 526–542. The swin copy at `models/swin/swin_test.py` lines 363–389 has the same logic but uses a local `from user_dma_core import ...` inside the function; the canonical version uses the module-level import only. **Rename at lift time.**

```python
def zero_fill_core_dram(ue: UnifiedEngine, DRAM_ADDR: int, num_elements: int) -> None:
    """Fill a DRAM region with zeros using SRAM as staging."""
    bpe = 2
    chunk = min(URAM_NEAR_FULL_ELEMENTS, num_elements)
    zeros = torch.zeros(chunk, dtype=torch.bfloat16)
    z_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(chunk * bpe)
    ue.dma_write(DMA_DEVICE_H2C, z_addr, zeros, chunk * bpe)
    ue.accelerator_memory_to_sram(accelerator_dram_address=z_addr,
                                   sram_address=0x00000, element_size=chunk)
    offset = 0
    while offset < num_elements:
        take = min(chunk, num_elements - offset)
        ue.sram_to_accelerator_memory(sram_address=0x00000,
                                       accelerator_dram_address=DRAM_ADDR + offset * bpe,
                                       element_size=take)
        offset += take
```

### 6.2 `copy_core_dram` (renamed from `dram_copy`)

Source of truth: `models/sam/sam1-vit-b/sam1_vit_b_test.py` lines 697–708. Byte-identical to `models/sam/sam3.1/sam3.1_test.py` lines 862–873. **Rename at lift time.**

```python
def copy_core_dram(ue: UnifiedEngine, SRC: int, DST: int, num_elements: int) -> None:
    """Copy num_elements bf16 values from SRC to DST in DRAM."""
    bpe = 2
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=SRC + offset * bpe,
            sram_address=0x00000, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=DST + offset * bpe,
            element_size=take)
```

### 6.3 `eltwise_add_core_dram`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 97–120. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 143–166.

**Resolved diff vs. sam's `eltwise_add_dram`:** the sam version (`sam1_vit_b_test.py` lines 655–675) is behaviorally equivalent — same URAM addresses, same write-back to URAM-A at `0x00000`, same three-step stage/compute/writeback pattern — with only stylistic differences (sam uses a raw `range()` loop, qwen uses `ue.chunk_ranges()`; sam returns `None`, qwen returns `size`; arg order differs — see §9 migration table). The canonical version below is qwen's, which means **every sam call site must be reordered** to the `(ue, size, A, B, OUTPUT)` signature.

```python
def eltwise_add_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A + B (DRAM)."""
    bytes_per_element = 2
    uram_a_addr = 0x00000
    uram_b_addr = 0x80000
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS, size)

    for start, take in ue.chunk_ranges(size, chunk_size):
        offset_bytes = start * bytes_per_element
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR + offset_bytes,
            sram_address=uram_a_addr, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_DRAM_ADDR + offset_bytes,
            sram_address=uram_b_addr, element_size=take)
        ue.eltwise_add_core(
            vector_A_sram_start_addr=uram_a_addr,
            vector_B_sram_start_addr=uram_b_addr,
            vector_C_sram_wb_addr=uram_a_addr, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=uram_a_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + offset_bytes,
            element_size=take)
    return size
```

### 6.4 `eltwise_mul_core_dram`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 122–145. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 167–190.

```python
def eltwise_mul_core_dram(ue, size: int, A_DRAM_ADDR: int, B_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int) -> int:
    """OUTPUT = A * B (DRAM)."""
    bytes_per_element = 2
    uram_a_addr = 0x00000
    uram_b_addr = 0x80000
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS, size)

    for start, take in ue.chunk_ranges(size, chunk_size):
        offset_bytes = start * bytes_per_element
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=A_DRAM_ADDR + offset_bytes,
            sram_address=uram_a_addr, element_size=take)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=B_DRAM_ADDR + offset_bytes,
            sram_address=uram_b_addr, element_size=take)
        ue.eltwise_mul_core(
            vector_A_sram_start_addr=uram_a_addr,
            vector_B_sram_start_addr=uram_b_addr,
            vector_C_sram_wb_addr=uram_a_addr, element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=uram_a_addr,
            accelerator_dram_address=OUTPUT_DRAM_ADDR + offset_bytes,
            element_size=take)
    return size
```

### 6.5 `broadcast_mul_core_dram` (renamed from `broadcast_mul_dram`)

Source of truth: `models/sam/sam1-vit-b/sam1_vit_b_test.py` lines 678–694. Byte-identical to `models/sam/sam3.1/sam3.1_test.py` lines 843–859. **Rename at lift time.**

**Note:** this function currently has **zero call sites** in the repo. It is being included because (a) it is already written and duplicated across both sam variants, and (b) its semantic niche (in-place scalar multiply) is distinct enough from `eltwise_mul_core_dram` that merging them is not warranted. If the implementer discovers that `broadcast_mul` (the underlying `ue.broadcast_mul` primitive) has been removed or renamed in `user_dma_core`, stop and surface — do not silently skip lifting.

```python
def broadcast_mul_core_dram(ue: UnifiedEngine, ADDR: int, scalar: float, num_elements: int) -> None:
    """In-place scalar multiply on DRAM buffer."""
    bpe = 2
    for offset in range(0, num_elements, URAM_NEAR_FULL_ELEMENTS):
        take = min(URAM_NEAR_FULL_ELEMENTS, num_elements - offset)
        ue.accelerator_memory_to_sram(
            accelerator_dram_address=ADDR + offset * bpe,
            sram_address=0x00000, element_size=take)
        ue.broadcast_mul(
            scalar=scalar,
            sram_start_addr=0x00000, sram_wb_addr=0x00000,
            element_size=take)
        ue.sram_to_accelerator_memory(
            sram_address=0x00000,
            accelerator_dram_address=ADDR + offset * bpe,
            element_size=take)
```

### 6.6 (reserved — no §6.6 function)

---

## 7. Normalization helpers

All three norms place gamma/beta (and, for layer_norm, a persistent zero vector) at the base of URAM-B at `0x80000`, then chunk activations through URAM-A at `0x00000`.

### 7.1 `rms_norm_core_dram_post_add`

Source of truth: `models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py` lines 315–358. Byte-identical to `models/smolvlm2/smolvlm2_test.py` lines 209–252.

```python
def rms_norm_core_dram_post_add(ue, M: int, N: int,
                                 A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                 ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                 GAMMA_DRAM_ADDR: int = None) -> int:
    """rms_norm(A + B) with residual output."""
    gamma_sram_addr = 0x80000
    params_sram_addr = gamma_sram_addr

    if GAMMA_DRAM_ADDR is not None:
        ue.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                    sram_address=gamma_sram_addr, element_size=N)
        params_sram_addr += N * 2
    else:
        gamma_sram_addr = None

    vector_A_sram_addr = 0x00000
    vector_B_sram_addr = params_sram_addr
    uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
    assert chunk_size >= 1 and chunk_size <= M

    for i, m_take in ue.chunk_ranges(M, chunk_size):
        chunk_elements = m_take * N
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                    sram_address=vector_A_sram_addr, element_size=chunk_elements)
        ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                    sram_address=vector_B_sram_addr, element_size=chunk_elements)
        for j in range(m_take):
            ue.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2,
                                vector_A_sram_addr + j * N * 2, N)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                        accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                        element_size=chunk_elements)
        for j in range(m_take):
            ue.rms_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2,
                             N, gamma_sram_addr)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                        accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                        element_size=chunk_elements)

    total_flops = M * N + 3 * M * N
    if gamma_sram_addr is not None:
        total_flops += M * N
    return total_flops
```

### 7.2 `layer_norm_core_dram`

Source of truth: `models/smolvlm2/smolvlm2_test.py` lines 253–285. SINGLETON definition, but there are 31 call sites across parakeet, gpt2, sam1-vit-b, sam3.1, smolvlm2, and swin (see §9.12) — the call sites exist because those models import or expect this helper. The implementer must confirm none of the callers define a local shadow; if any does, reconcile before lifting.

```python
def layer_norm_core_dram(ue, M: int, N: int,
                         A_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                         GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> int:
    """OUTPUT = LayerNorm(A)."""
    zeros_sram = 0x80000
    zeros_dram = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * 2)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram, torch.zeros(N, dtype=torch.bfloat16), N * 2)
    ue.accelerator_memory_to_sram(zeros_dram, zeros_sram, N)
    params_sram = zeros_sram + N * 2

    gamma_sram = None
    if GAMMA_DRAM_ADDR is not None:
        gamma_sram = params_sram
        ue.accelerator_memory_to_sram(GAMMA_DRAM_ADDR, gamma_sram, N)
        params_sram += N * 2
    beta_sram = None
    if BETA_DRAM_ADDR is not None:
        beta_sram = params_sram
        ue.accelerator_memory_to_sram(BETA_DRAM_ADDR, beta_sram, N)
        params_sram += N * 2

    vector_sram = 0x00000
    uram_b_used = (params_sram - 0x80000) // 2
    chunk = min(URAM_NEAR_FULL_ELEMENTS // N, (URAM_NEAR_FULL_ELEMENTS - uram_b_used) // N, M)
    assert chunk >= 1

    for i, m_take in ue.chunk_ranges(M, chunk):
        ue.accelerator_memory_to_sram(A_DRAM_ADDR + i * N * 2, vector_sram, m_take * N)
        for j in range(m_take):
            ue.layer_norm_core(vector_sram + j * N * 2, vector_sram + j * N * 2, N, zeros_sram, gamma_sram, beta_sram)
        ue.sram_to_accelerator_memory(vector_sram, OUTPUT_DRAM_ADDR + i * N * 2, m_take * N)
    return 5 * M * N + (M * N if gamma_sram else 0) + (M * N if beta_sram else 0)
```

**VERIFY:** the 31 call-site files listed in §9.12 — confirm that each one currently *imports* `layer_norm_core_dram` from a shared source or has its own local definition. If they rely on a definition not already in this spec, diff against the smolvlm2 version above before accepting the lift as canonical.

### 7.3 `layer_norm_core_dram_post_add`

Source of truth: `models/smolvlm2/smolvlm2_test.py` lines 286–346. SINGLETON today; included because it is the direct counterpart of §7.1 and will be needed by any non-RMSNorm model that adds a residual.

```python
def layer_norm_core_dram_post_add(ue, M: int, N: int,
                                   A_DRAM_ADDR: int, B_DRAM_ADDR: int,
                                   ADDOUTPUT_DRAM_ADDR: int, NORMOUTPUT_DRAM_ADDR: int,
                                   GAMMA_DRAM_ADDR: int = None, BETA_DRAM_ADDR: int = None) -> int:
    """layer_norm(A + B) with residual output."""
    zeros_sram_addr = 0x80000

    zeros_dram_addr = ue.get_params_dram_addr()
    ue.allocate_params_dram(N * 2)
    ue.dma_write(DMA_DEVICE_H2C, zeros_dram_addr, torch.zeros(N, dtype=torch.bfloat16), N * 2)

    ue.accelerator_memory_to_sram(accelerator_dram_address=zeros_dram_addr,
                                    sram_address=zeros_sram_addr, element_size=N)
    params_sram_addr = zeros_sram_addr + N * 2

    gamma_sram_addr = None
    beta_sram_addr = None

    if GAMMA_DRAM_ADDR is not None:
        gamma_sram_addr = params_sram_addr
        ue.accelerator_memory_to_sram(accelerator_dram_address=GAMMA_DRAM_ADDR,
                                    sram_address=gamma_sram_addr, element_size=N)
        params_sram_addr += N * 2

    if BETA_DRAM_ADDR is not None:
        beta_sram_addr = params_sram_addr
        ue.accelerator_memory_to_sram(accelerator_dram_address=BETA_DRAM_ADDR,
                                    sram_address=beta_sram_addr, element_size=N)
        params_sram_addr += N * 2

    vector_A_sram_addr = 0x00000
    vector_B_sram_addr = params_sram_addr
    uram_b_remaining_elements = URAM_NEAR_FULL_ELEMENTS - (params_sram_addr - 0x80000) // 2
    chunk_size = min(URAM_NEAR_FULL_ELEMENTS // N, uram_b_remaining_elements // N, M)
    assert chunk_size >= 1 and chunk_size <= M

    for i, m_take in ue.chunk_ranges(M, chunk_size):
        chunk_elements = m_take * N
        ue.accelerator_memory_to_sram(accelerator_dram_address=A_DRAM_ADDR + i * N * 2,
                                    sram_address=vector_A_sram_addr, element_size=chunk_elements)
        ue.accelerator_memory_to_sram(accelerator_dram_address=B_DRAM_ADDR + i * N * 2,
                                    sram_address=vector_B_sram_addr, element_size=chunk_elements)
        for j in range(m_take):
            ue.eltwise_add_core(vector_A_sram_addr + j * N * 2, vector_B_sram_addr + j * N * 2,
                                vector_A_sram_addr + j * N * 2, N)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                        accelerator_dram_address=ADDOUTPUT_DRAM_ADDR + i * N * 2,
                                        element_size=chunk_elements)
        for j in range(m_take):
            ue.layer_norm_core(vector_A_sram_addr + j * N * 2, vector_A_sram_addr + j * N * 2,
                               N, zeros_sram_addr, gamma_sram_addr, beta_sram_addr)
        ue.sram_to_accelerator_memory(sram_address=vector_A_sram_addr,
                                        accelerator_dram_address=NORMOUTPUT_DRAM_ADDR + i * N * 2,
                                        element_size=chunk_elements)

    total_flops = M * N + 5 * M * N
    if gamma_sram_addr is not None:
        total_flops += M * N
    if beta_sram_addr is not None:
        total_flops += M * N
    return total_flops
```

---

## 8. Attention helper

### 8.1 `flash_attention_batched_core_dram` (renamed and unified from `flash_attention_batched`)

Three source copies exist: `sam1-vit-b` lines 104–137 (includes `bias_shared` kwarg), `sam3.1` lines 98–127 (no `bias_shared`), `swin` lines 164–195 (no `bias_shared`, uses longer variable names). Canonical version is sam1-vit-b's — it is the strict superset. When `bias_shared` defaults to `False`, sam3.1's and swin's behavior is reproduced exactly (BIAS stride = `seq_len * seq_len * bpe`). When `bias_shared=True`, sam1-vit-b's shared-mask-broadcast case is reproduced (bias stride = 0). **Rename at lift time.**

```python
def flash_attention_batched_core_dram(ue: UnifiedEngine, num_batches: int, head_dim: int,
                                       seq_len: int, Q_DRAM_ADDR: int, K_DRAM_ADDR: int,
                                       V_DRAM_ADDR: int, OUTPUT_DRAM_ADDR: int,
                                       SCRATCH_DRAM_ADDR: int,
                                       BIAS_DRAM_ADDR: int = None,
                                       bias_shared: bool = False) -> int:
    """Batched flash self-attention: loop over num_batches calling flash_attention_core.

    Each batch is one attention head. Q/K/V/OUTPUT are contiguous per batch with
    stride (seq_len * head_dim). BIAS stride is (seq_len * seq_len) if provided,
    or 0 if bias_shared=True (single mask broadcast across all batches).
    """
    bpe = 2
    qkv_stride = seq_len * head_dim * bpe
    out_stride = seq_len * head_dim * bpe
    scratch_stride = head_dim * seq_len * bpe + seq_len * seq_len * bpe
    bias_stride = 0 if (BIAS_DRAM_ADDR is None or bias_shared) else seq_len * seq_len * bpe

    total_flops = 0
    for b in range(num_batches):
        bias_addr = BIAS_DRAM_ADDR if (BIAS_DRAM_ADDR is not None) else None
        if bias_addr is not None and not bias_shared:
            bias_addr = BIAS_DRAM_ADDR + b * bias_stride
        total_flops += ue.flash_attention_core(
            head_dim=head_dim,
            seq_len=seq_len,
            Q_DRAM_ADDR=Q_DRAM_ADDR + b * qkv_stride,
            K_DRAM_ADDR=K_DRAM_ADDR + b * qkv_stride,
            V_DRAM_ADDR=V_DRAM_ADDR + b * qkv_stride,
            OUTPUT_DRAM_ADDR=OUTPUT_DRAM_ADDR + b * out_stride,
            SCRATCH_DRAM_ADDR=SCRATCH_DRAM_ADDR + b * scratch_stride,
            BIAS_DRAM_ADDR=bias_addr,
        )
    return total_flops
```

---

## 9. Migration: delete-local-copy and update call-sites

For each model listed below, the migration is:
(a) add `from model_lib_core import <names>` after existing imports,
(b) delete the listed local definitions,
(c) update call sites per the rename/reorder column.

Do **one model per commit** so any regression is bisectable.

### 9.1 Rename table (apply at every call site)

| Old name | New name | Arg-order change? |
|---|---|---|
| `dram_zero_fill` | `zero_fill_core_dram` | no |
| `dram_copy` | `copy_core_dram` | no |
| `broadcast_mul_dram` | `broadcast_mul_core_dram` | no |
| `eltwise_add_dram` (sam) | `eltwise_add_core_dram` | **YES** — old `(ue, A, B, OUT, n)` → new `(ue, n, A, B, OUT)` |
| `eltwise_add_core_dram` (qwen/smolvlm) | `eltwise_add_core_dram` | no |
| `eltwise_mul_core_dram` | `eltwise_mul_core_dram` | no |
| `flash_attention_batched` | `flash_attention_batched_core_dram` | no |
| `_quantize_bf16_to_int4_packed` | `quantize_bf16_to_int4_packed` | no |
| `store_weight`, `store_quantized_weight`, `load_weight_cache` | unchanged | no |
| `quantize_q4_64`, `quantize_fp4_64` | unchanged | no |
| `rms_norm_core_dram_post_add` | unchanged | no |
| `layer_norm_core_dram`, `layer_norm_core_dram_post_add` | unchanged | no |

### 9.2 gemma3 — commit 1

- Delete `_quantize_bf16_to_int4_packed` definition in `models/gemma3/gemma3_numeric.py` lines 66–81.
- Update call sites: `models/gemma3/gemma3_test.py:169`, `:213` — rename `_quantize_bf16_to_int4_packed` → `quantize_bf16_to_int4_packed`.
- Add `from model_lib_core import quantize_bf16_to_int4_packed` at the top of `gemma3_numeric.py` and `gemma3_test.py` (only where referenced).

### 9.3 llama3.2_1b — commit 2

- Delete `_quantize_bf16_to_int4_packed` definition in `models/llama3.2_1b/llama3.2_1b_test.py` lines 71–86.
- Update call sites: `models/llama3.2_1b/llama3.2_1b_test.py:216`, `:268` — rename → `quantize_bf16_to_int4_packed`.
- Add import.

### 9.4 qwen3_1.7b — commit 3

- Delete `_quantize_bf16_to_int4_packed` definition in `models/qwen3_1.7b/qwen3_1.7b_test.py` lines 77–92.
- Update call sites: `models/qwen3_1.7b/qwen3_1.7b_test.py:186`, `:228` — rename → `quantize_bf16_to_int4_packed`.
- Add import.

### 9.5 qwen2.5_vl_3b — commit 4

- Delete definitions: `store_weight` (40–53), `store_quantized_weight` (55–74), `load_weight_cache` (76–86), `eltwise_add_core_dram` (97–120), `eltwise_mul_core_dram` (122–145), `rms_norm_core_dram_post_add` (315–358), `quantize_q4_64` (363–376), `quantize_fp4_64` (378–394). Also delete any local `_FP4_E2M1_TABLE` definition.
- Call sites (no rename needed, just the import):
  - `store_weight`: 1261, 1262, 1268, 1271, 1273, 1274, 1280, 1319, 1321, 1324, 1328, 1330, 1331, 1338, 1341, 1343
  - `store_quantized_weight`: 1259, 1286, 1318, 1320, 1323, 1327, 1335, 1340, 1342
  - `load_weight_cache`: 1238, 1302
  - `eltwise_add_core_dram`: 1806
  - `quantize_q4_64`: 515, 527, 533, 543, 552, 565, 573
  - `quantize_fp4_64`: 1285
  - `rms_norm_core_dram_post_add`: 1763
- Add `from model_lib_core import (store_weight, store_quantized_weight, load_weight_cache, eltwise_add_core_dram, eltwise_mul_core_dram, rms_norm_core_dram_post_add, quantize_q4_64, quantize_fp4_64)`.

### 9.6 smolvlm2 — commit 5

- Delete definitions: `store_weight` (42–55), `store_quantized_weight` (56–75), `load_weight_cache` (76–86), `eltwise_add_core_dram` (143–166), `eltwise_mul_core_dram` (167–190), `rms_norm_core_dram_post_add` (209–252), `layer_norm_core_dram` (253–285), `layer_norm_core_dram_post_add` (286–346), `quantize_q4_64` (1009–1022), `quantize_fp4_64` (1023–1039). Also delete any local `_FP4_E2M1_TABLE`.
- Call sites (no rename):
  - `store_weight`: 1269, 1275, 1277, 1278, 1279, 1282, 1284, 1294, 1306, 1309, 1340, 1341, 1344, 1345, 1348, 1349, 1353, 1354, 1357, 1359, 1360, 1363, 1378, 1383, 1387, 1389, 1391, 1395, 1396, 1398
  - `store_quantized_weight`: 1303, 1311, 1376, 1381, 1401
  - `load_weight_cache`: 1289, 1369
  - `eltwise_add_core_dram`: 1540, 1588, 1718
  - `eltwise_mul_core_dram`: 1712
  - `rms_norm_core_dram_post_add`: 1665, 1706, 1822
  - `layer_norm_core_dram`: 1556, 1592
  - `layer_norm_core_dram_post_add`: 1582
- Add the corresponding import line.

### 9.7 sam1-vit-b — commit 6

- Delete definitions: `flash_attention_batched` (104–137), `dram_zero_fill` (361–377), `eltwise_add_dram` (655–675), `broadcast_mul_dram` (678–694), `dram_copy` (697–708).
- Call sites (rename + reorder where noted):
  - `dram_zero_fill` → `zero_fill_core_dram`: 406, 722, 1312
  - `dram_copy` → `copy_core_dram`: 726, 736, 1315, 1391
  - `eltwise_add_dram` → `eltwise_add_core_dram` **(arg reorder)**: 1216, 1443, 1485 — at each of these, rewrite the call from `eltwise_add_dram(ue, A, B, OUT, n)` to `eltwise_add_core_dram(ue, n, A, B, OUT)`.
  - `flash_attention_batched` → `flash_attention_batched_core_dram`: 1374
  - `layer_norm_core_dram` call sites (already-matching name, just import): 1245, 1448, 1517, 1537, 2255
- Add `from model_lib_core import (zero_fill_core_dram, copy_core_dram, eltwise_add_core_dram, flash_attention_batched_core_dram, layer_norm_core_dram, broadcast_mul_core_dram)` (include `broadcast_mul_core_dram` even though unused — so re-introduction doesn't require a re-import).

### 9.8 sam3.1 — commit 7

- Delete definitions: `flash_attention_batched` (98–127), `dram_zero_fill` (526–542), `eltwise_add_dram` (820–840), `broadcast_mul_dram` (843–859), `dram_copy` (862–873).
- Call sites (rename + reorder where noted):
  - `dram_zero_fill` → `zero_fill_core_dram`: 571, 2817
  - `dram_copy` → `copy_core_dram`: 2950
  - `eltwise_add_dram` → `eltwise_add_core_dram` **(arg reorder)**: 1856, 2022, 2054, 2871, 2913, 3007, 3033, 3106
  - `flash_attention_batched` → `flash_attention_batched_core_dram`: 1978
  - `layer_norm_core_dram` call sites (import only): 1860, 1893, 2026, 2794, 2882, 2924, 2961, 2981, 3024
- Add the import line.

### 9.9 swin — commit 8

- Delete definitions: `flash_attention_batched` (164–195), `dram_zero_fill` (363–389).
- Call sites:
  - `dram_zero_fill` → `zero_fill_core_dram`: 431, 1129, 1161, 1373
  - `flash_attention_batched` → `flash_attention_batched_core_dram`: 1162
  - `layer_norm_core_dram` call sites (import only): 1054, 1084, 1275, 1343, 1363
- Add the import line.

### 9.10 gpt2 — commit 9 (call-site-only; no deletions)

- `gpt2_test.py` already calls `layer_norm_core_dram` at 763, 844, 936, 1011, 1034 but **does not define it locally** (VERIFY — grep `gpt2_test.py` for `def layer_norm_core_dram`). If so, gpt2 is currently broken and only compiles because of an import from elsewhere. Add `from model_lib_core import layer_norm_core_dram` at the top.

### 9.11 parakeet — commit 10 (call-site-only; no deletions)

- `parakeet_test.py` calls `layer_norm_core_dram` at 1588, 1622, 1722, 1787, 1813 but **does not define it locally** (VERIFY as in §9.10). Add `from model_lib_core import layer_norm_core_dram`.

### 9.12 Complete `layer_norm_core_dram` call-site index

For completeness (same sites already listed above in context):
- parakeet: 1588, 1622, 1722, 1787, 1813
- smolvlm2: 1556, 1592
- gpt2: 763, 844, 936, 1011, 1034
- sam3.1: 1860, 1893, 2026, 2794, 2882, 2924, 2961, 2981, 3024
- sam1-vit-b: 1245, 1448, 1517, 1537, 2255
- swin: 1054, 1084, 1275, 1343, 1363

---

## 10. Out-of-scope (explicit non-goals for this task)

Do NOT include any of the following in this extraction. Each has known divergence or insufficient duplication to justify lifting now:

- `window_partition_dram`, `window_reverse_dram` — structural divergence between swin and sam copies (swin lacks URAM-A row batching).
- Any RoPE variant (`rope_hf_core_dram`, `rope_2d_heads_dram`, `rope_2d_vectorized_dram`, `_rope_kv_perm`, `_make_rope_perm`, `precompute_rope_2d`).
- `multihead_reshape_dram`, `multihead_merge_dram`, `multihead_pad_and_permute`.
- Conv family: `conv2d_1x1_dram`, `conv2d_3x3_dram`, `conv_transpose2d_2x2_dram`, `nearest_upsample_2x_dram`.
- Permute family: `bf16_permute_core_v2`, `bf16_smart_permute_core`, `chunked_transpose_core_dram`, `bf16_patching_core`.
- Parakeet-only helpers: `compute_mel_spectrogram`, `frame_waveform`, `rel_shift`, `batch_norm_core_dram`, `batch_norm_fuse_params`, `silu_core_dram`, `tanh_core_dram`, `glu_core_dram`, `half_step_residual_core_dram`, `allocate_identity`, `pad_to_multiple`, `conv2d_outsize`.
- Host-side config utilities (`_load_config`, `_ensure_hf_model`, `_parse_offset`, `quiet_print`, `weight_bin_generate`) — these belong in a future `model_host_util.py`, NOT in `model_lib_core.py`. Leave them where they are.
- Any modification to `user_dma_core.py` or `user_dma_core.c` — off-limits.

---

## 11. Verification after each commit

There is no pytest suite. Verification is per-model functional parity:

1. Before commit N, run the model's existing bring-up entry point and capture its stdout and any numerical SNR line.
2. After commit N, run again and diff.
3. If output changes, **stop**. Do not "fix" by modifying the library to match the old per-model behavior — other models depend on the canonical version.
4. If a model's baseline is not green on main at migration time (currently suspected for models on feature branches), skip that model's migration and record it in the PR description under "Deferred migrations".

---

## 12. Definition of done

1. `/home/rohit/unified-engine/model_lib_core.py` exists with exactly the contents of §2–§8 above.
2. `_FP4_E2M1_TABLE` has been confirmed identical across both source files and pasted into the library (§4.1).
3. Every model file listed in §9.2–§9.11 has had its local duplicates removed and its call sites renamed/reordered.
4. Every model whose baseline was verifiable produces numerically identical output to its pre-extraction run.
5. The PR description explicitly records: (a) the `_FP4_E2M1_TABLE` verification result, (b) the gpt2 and parakeet local-definition checks from §9.10 and §9.11, (c) any deferred model migrations from §11.4.
