"""Weight-bin generation for Qwen2.5-VL-3B: HF checkpoint -> params.bin.

WHY THIS IS A SEPARATE MODULE. It runs ONCE, on a machine that has no
``params.bin`` yet, and it needs things the runtime deliberately does not:
``transformers`` to load the checkpoint and ``huggingface_hub`` to fetch it.
Keeping it out of the mixins means a normal run imports neither.

The output is a single ``params.bin`` holding the LM region followed by the
vision region, plus a ``params.json`` sidecar whose ``regions`` map gives each
region's offset, size and a manifest of region-relative tensor offsets. Weights
are stored the way the hardware wants them -- quantized at the configured
precision, and for vision pre-padded (head_dim 80->128, MLP 3420->3456) -- so
the runtime never needs the HF model.

Lifted from the pre-refactor qwen2.5_vl_3b_test.py (commit 0047b4ca^), which was
the only thing that produced these files; the refactor dropped it and left the
model unable to bootstrap on a fresh checkout. Kept byte-for-byte where possible
so bins generated before and after the refactor are identical.
"""
import json
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download

# quant_lib lives at the repo root, two levels up from models/<model>/. Added
# here rather than relied on from the caller so the module is importable on its
# own -- it is also a usable standalone generator.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import quant_lib


def _load_config(script_dir: str) -> dict:
    """The model config. Only ``paths`` and ``precision`` are used here."""
    with open(os.path.join(script_dir, "qwen2.5_vl_3b_config.json")) as f:
        return json.load(f)


def _qs_pack(precision: str, tensor: torch.Tensor):
    """Run a 64-block codec via quant_lib and emit the scale-then-data
    byte layout the released wire format uses (consumed by
    store_quantized_weight() as 34 bytes per block: 2 B bf16 scale + 32 B
    nibbles). Accepts arbitrary input shape; flattens and zero-pads to a
    multiple of 64 to match the released helpers."""
    bf = tensor.detach().to(torch.bfloat16).cpu()
    # Fast path: 2D and K-aligned. Preserve (N, K) shape so the codec can
    # chunk along rows (large N inputs like the LM head would otherwise hit
    # the unbounded distance-tensor allocation when flattened to (1, N*K)).
    if bf.dim() == 2 and bf.shape[1] % 64 == 0:
        n_blocks = bf.numel() // 64
        data_bytes, scale_bytes = quant_lib.quantize(precision, bf, block_size=64)
        return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks
    # Generic path: flatten + zero-pad to a multiple of 64. Used for
    # arbitrary-rank tensors (e.g. the 5D Conv3D patch_embed weight).
    bf = bf.flatten()
    n_blocks = (bf.numel() + 63) // 64
    if bf.numel() != n_blocks * 64:
        bf = torch.nn.functional.pad(bf, (0, n_blocks * 64 - bf.numel()))
    bf = bf.view(1, -1)
    data_bytes, scale_bytes = quant_lib.quantize(precision, bf, block_size=64)
    return np.frombuffer(scale_bytes + data_bytes, dtype=np.uint8), n_blocks

_LAYER_MAP = {
    'lm': {
        'self_attn.q_proj.weight': 'attn_q.weight', 'self_attn.q_proj.bias': 'attn_q.bias',
        'self_attn.k_proj.weight': 'attn_k.weight', 'self_attn.k_proj.bias': 'attn_k.bias',
        'self_attn.v_proj.weight': 'attn_v.weight', 'self_attn.v_proj.bias': 'attn_v.bias',
        'self_attn.o_proj.weight': 'attn_output.weight',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.up_proj.weight': 'ffn_up.weight',
        'mlp.down_proj.weight': 'ffn_down.weight', 'input_layernorm.weight': 'attn_norm.weight',
        'post_attention_layernorm.weight': 'ffn_norm.weight',
    },
    'vision': {
        'attn.qkv.weight': 'attn_qkv.weight', 'attn.qkv.bias': 'attn_qkv.bias',
        'attn.proj.weight': 'attn_out.weight', 'attn.proj.bias': 'attn_out.bias',
        'mlp.gate_proj.weight': 'ffn_gate.weight', 'mlp.gate_proj.bias': 'ffn_gate.bias',
        'mlp.up_proj.weight': 'ffn_up.weight', 'mlp.up_proj.bias': 'ffn_up.bias',
        'mlp.down_proj.weight': 'ffn_down.weight', 'mlp.down_proj.bias': 'ffn_down.bias',
        'norm1.weight': 'norm1.weight', 'norm2.weight': 'norm2.weight',
    },
}
_TOP_MAP = {
    'lm': {'embed_tokens.weight': 'token_embd.weight', 'norm.weight': 'output_norm.weight'},
    'vision': {
        'patch_embed.proj.weight': 'v.patch_embd.weight',
        'patch_embed.proj.bias': 'v.patch_embd.bias',
        'merger.ln_q.weight': 'v.merger_ln_q.weight',
        'merger.mlp.0.weight': 'v.merger_mlp0.weight',
        'merger.mlp.0.bias': 'v.merger_mlp0.bias',
        'merger.mlp.2.weight': 'v.merger_mlp2.weight',
        'merger.mlp.2.bias': 'v.merger_mlp2.bias',
    },
}
# LM-side quant scope: q/k/gate/up/down (v_proj and o_proj stay BF16 for
# attention accuracy). Same set for every supported precision.
_LM_QUANT_LAYERS = {'q_proj.weight', 'k_proj.weight',
                    'gate_proj.weight', 'up_proj.weight', 'down_proj.weight'}

_VALID_PRECISIONS = ('int4', 'fp4', 'if4')

def _lm_precision(cfg: dict) -> str:
    """Read LM precision from the config, defaulting to the eval-winner 'if4'.
    Validates against the codecs the quant_lib wrapper actually supports."""
    p = cfg.get("precision", {}).get("lm", "if4")
    if p not in _VALID_PRECISIONS:
        raise ValueError(f"config precision.lm={p!r} not in {_VALID_PRECISIONS}")
    return p

def _vision_precision(cfg: dict) -> str:
    """Read vision precision from the config. Default 'int4' matches the
    legacy released vision codec (Q4_64 = pure INT4 codes, all-negative
    bf16 scales — HW INT4 dispatch). Same precision string drives both the
    generator's codec call and the manifest suffix the loader looks for."""
    p = cfg.get("precision", {}).get("vision", "int4")
    if p not in _VALID_PRECISIONS:
        raise ValueError(f"config precision.vision={p!r} not in {_VALID_PRECISIONS}")
    return p

def _weight_key(hf_name, mode):
    """Map HF param name to short weight key. mode='lm' or 'vision'."""
    name = hf_name
    for pfx in ('model.', 'visual.'):
        if name.startswith(pfx):
            name = name[len(pfx):]
            break
    if name in _TOP_MAP[mode]:
        return _TOP_MAP[mode][name]
    if mode == 'lm' and name.startswith('layers.'):
        p = name.split('.'); comp = '.'.join(p[2:])
        if comp in _LAYER_MAP['lm']:
            return f'blk.{p[1]}.{_LAYER_MAP["lm"][comp]}'
    elif mode == 'vision' and name.startswith('blocks.'):
        p = name.split('.'); comp = '.'.join(p[2:])
        if comp in _LAYER_MAP['vision']:
            return f'v.blk.{p[1]}.{_LAYER_MAP["vision"][comp]}'
    return name

def _write_weight_bin(bin_path, model, param_filter, mode, suffix, quant_layers, qfn):
    """Write a weight bin + json manifest. Params whose name ends with one
    of ``quant_layers`` go through ``qfn`` (returning packed scale+nibble
    bytes) and get a ``.{suffix}`` manifest key; everything else is stored
    BF16 with no suffix."""
    json_path = bin_path.rsplit('.', 1)[0] + '.json'
    manifest = {}
    count = 0
    with open(bin_path, 'wb') as f:
        for pname, param in model.named_parameters():
            if not param_filter(pname):
                continue
            key = _weight_key(pname, mode)
            t = param.data
            if any(pname.endswith(s) for s in quant_layers):
                data, _ = qfn(t)
                raw = data.tobytes()
                key = f'{key}.{suffix}'
            else:
                raw = t.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell()
            f.write(raw)
            manifest[key] = {'offset': offset, 'size': len(raw)}
            count += 1
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"Weights: {count} tensors, {os.path.getsize(bin_path)/1048576:.1f} MB -> {bin_path}")

def generate_lm_weights(model, output_path, precision: str = "if4"):
    """Generate LM weight bin using the given quant_lib precision
    ('int4' / 'fp4' / 'if4'). The precision string is also the manifest
    suffix the runtime loader looks for.

    Also pre-quantizes the LM head (tied to the input embedding) and
    appends it to the bin as ``lm_head.weight.{precision}``. The runtime
    loads those bytes directly instead of re-quantizing 300M+ weights at
    weight_init, which OOMs memory-constrained devices like Pi 5."""
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"precision={precision!r} not in {_VALID_PRECISIONS}")
    _write_weight_bin(output_path, model,
        lambda n: 'model.layers' in n or 'model.embed_tokens' in n or 'model.norm' in n,
        'lm', precision, _LM_QUANT_LAYERS, lambda t: _qs_pack(precision, t))

    # Pre-quantize the LM head (tied to embedding) and append to the bin.
    embed_w = model.get_input_embeddings().weight.detach().to(torch.bfloat16)
    combined, _ = _qs_pack(precision, embed_w)
    combined_bytes = combined.tobytes()
    json_path = output_path.rsplit('.', 1)[0] + '.json'
    with open(json_path) as f:
        manifest = json.load(f)
    with open(output_path, 'ab') as f:
        offset = f.tell()
        f.write(combined_bytes)
    manifest[f'lm_head.weight.{precision}'] = {'offset': offset, 'size': len(combined_bytes)}
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"LM head ({precision}) appended: {len(combined_bytes)/1048576:.1f} MB at offset 0x{offset:X}")

def generate_vision_weights(model, output_path, precision: str = "int4"):
    """Generate vision weight bin with pre-padded QKV (80→128) and MLP
    (3420→3456). The given quant_lib precision ('int4' / 'fp4' / 'if4')
    drives both the codec and the manifest suffix.

    The binary stores padded weights so the runtime doesn't need the HF model.
    QKV is stored as separate qk_padded (rearranged 128-dim) and v_padded
    (sequential 128-dim).
    """
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"precision={precision!r} not in {_VALID_PRECISIONS}")
    sfx = precision  # manifest suffix tag = precision string
    qpack = lambda t: _qs_pack(precision, t)
    VN, VD, VD_PAD, VH = 16, 80, 128, 1280
    VI = 3420
    VIS_MLP_PAD = ((VI + 63) // 64) * 64
    half_d = VD // 2
    json_path = output_path.rsplit('.', 1)[0] + '.json'
    manifest = {}
    count = 0
    with open(output_path, 'wb') as f:
        for i, block in enumerate(model.visual.blocks):
            prefix = f'visual.blocks.{i}'
            # QKV: pad and split into qk_padded (rearranged) + v_padded (sequential)
            qkv_w = block.attn.qkv.weight.detach().to(torch.bfloat16)
            qkv_b = block.attn.qkv.bias.detach().to(torch.bfloat16)
            qkv_w_3d = qkv_w.view(3, VN, VD, VH)
            qkv_b_3d = qkv_b.view(3, VN, VD)
            # Q/K rearranged padded
            qk_padded_w = torch.zeros(2 * VN * VD_PAD, VH, dtype=torch.bfloat16)
            qk_padded_b = torch.zeros(2 * VN * VD_PAD, dtype=torch.bfloat16)
            for proj in range(2):
                for h in range(VN):
                    hs = (proj * VN + h) * VD_PAD
                    qk_padded_w[hs:hs+half_d, :] = qkv_w_3d[proj, h, :half_d, :]
                    qk_padded_w[hs+64:hs+64+half_d, :] = qkv_w_3d[proj, h, half_d:, :]
                    qk_padded_b[hs:hs+half_d] = qkv_b_3d[proj, h, :half_d]
                    qk_padded_b[hs+64:hs+64+half_d] = qkv_b_3d[proj, h, half_d:]
            data, _ = qpack(qk_padded_w)
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.qk_padded.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = qk_padded_b.contiguous().view(torch.uint16).cpu().numpy().tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.qk_padded.bias'] = {'offset': offset, 'size': len(raw)}
            # V sequential padded
            v_padded_w = torch.zeros(VN * VD_PAD, VH, dtype=torch.bfloat16)
            v_padded_b = torch.zeros(VN * VD_PAD, dtype=torch.bfloat16)
            for h in range(VN):
                hs = h * VD_PAD
                v_padded_w[hs:hs+VD, :] = qkv_w_3d[2, h, :, :]
                v_padded_b[hs:hs+VD] = qkv_b_3d[2, h, :]
            data, _ = qpack(v_padded_w)
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.v_padded.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = v_padded_b.contiguous().view(torch.uint16).cpu().numpy().tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.v_padded.bias'] = {'offset': offset, 'size': len(raw)}
            # O proj (no padding needed)
            data, _ = qpack(block.attn.proj.weight.detach().to(torch.bfloat16))
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.proj.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = block.attn.proj.bias.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.attn.proj.bias'] = {'offset': offset, 'size': len(raw)}
            # MLP: pad 3420→3456
            for proj_name in ['gate_proj', 'up_proj']:
                w = getattr(block.mlp, proj_name).weight.detach().to(torch.bfloat16)
                w_padded = torch.zeros(VIS_MLP_PAD, VH, dtype=torch.bfloat16); w_padded[:w.shape[0]] = w
                data, _ = qpack(w_padded)
                raw = data.tobytes(); offset = f.tell(); f.write(raw)
                manifest[f'{prefix}.mlp.{proj_name}.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
                b = getattr(block.mlp, proj_name).bias.detach().to(torch.bfloat16)
                b_padded = torch.zeros(VIS_MLP_PAD, dtype=torch.bfloat16); b_padded[:b.shape[0]] = b
                raw = b_padded.contiguous().view(torch.uint16).cpu().numpy().tobytes(); offset = f.tell(); f.write(raw)
                manifest[f'{prefix}.mlp.{proj_name}.bias'] = {'offset': offset, 'size': len(raw)}
            down_w = block.mlp.down_proj.weight.detach().to(torch.bfloat16)
            down_padded = torch.zeros(VH, VIS_MLP_PAD, dtype=torch.bfloat16); down_padded[:, :down_w.shape[1]] = down_w
            data, _ = qpack(down_padded)
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.mlp.down_proj.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = block.mlp.down_proj.bias.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell(); f.write(raw)
            manifest[f'{prefix}.mlp.down_proj.bias'] = {'offset': offset, 'size': len(raw)}
            # Norms
            for norm_name in ['norm1', 'norm2']:
                raw = getattr(block, norm_name).weight.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
                offset = f.tell(); f.write(raw)
                manifest[f'{prefix}.{norm_name}.weight'] = {'offset': offset, 'size': len(raw)}
            count += 1
        # Patch embed (no padding needed)
        data, _ = qpack(model.visual.patch_embed.proj.weight.detach().to(torch.bfloat16))
        raw = data.tobytes(); offset = f.tell(); f.write(raw)
        manifest[f'visual.patch_embed.proj.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
        # Merger
        raw = model.visual.merger.ln_q.weight.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
        offset = f.tell(); f.write(raw)
        manifest['visual.merger.ln_q.weight'] = {'offset': offset, 'size': len(raw)}
        for mlp_idx in [0, 2]:
            data, _ = qpack(model.visual.merger.mlp[mlp_idx].weight.detach().to(torch.bfloat16))
            raw = data.tobytes(); offset = f.tell(); f.write(raw)
            manifest[f'visual.merger.mlp.{mlp_idx}.weight.{sfx}'] = {'offset': offset, 'size': len(raw)}
            raw = model.visual.merger.mlp[mlp_idx].bias.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().tobytes()
            offset = f.tell(); f.write(raw)
            manifest[f'visual.merger.mlp.{mlp_idx}.bias'] = {'offset': offset, 'size': len(raw)}
    with open(json_path, 'w') as f:
        json.dump(manifest, f)
    print(f"Vision weights (padded {precision}): {count} layers + merger, {os.path.getsize(output_path)/1048576:.1f} MB -> {output_path}")


def _ensure_hf_model(script_dir: str, cfg: dict):
    """Ensure HF model is downloaded and loaded. Returns (model, model_dir)."""
    model_dir = os.path.join(script_dir, cfg["paths"]["hf_model_dir"])
    hf_repo = cfg["paths"]["hf_model_repo"]
    config_path = os.path.join(model_dir, "config.json")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    missing_required = not os.path.exists(config_path)
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        required_files = set(weight_map.values())
        missing_required = missing_required or any(
            not os.path.exists(os.path.join(model_dir, name))
            for name in required_files)
    if missing_required:
        print(f"Downloading HF model {hf_repo} to {os.path.abspath(model_dir)} ...")
        snapshot_download(repo_id=hf_repo, local_dir=model_dir, local_dir_use_symlinks=False)
        print("Download complete.")
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True
    )
    # Newer transformers nests vision encoder under model.model.visual;
    # expose it as model.visual so the rest of the code can access it directly.
    if not hasattr(model, 'visual') and hasattr(model, 'model') and hasattr(model.model, 'visual'):
        model.visual = model.model.visual
    return model, model_dir

def weight_bin_generate(script_dir: str | None = None, output_params: str | None = None) -> str:
    """Generate the unified params.bin (LM + vision) from a Hugging Face model.

    LM and vision weight bytes are generated into temporary sidecar files,
    then concatenated into a single ``params.bin`` whose ``params.json``
    sidecar carries a ``regions`` map ({lm,vision} → {offset, size,
    manifest}) where each region's manifest holds region-relative tensor
    offsets. Returns the unified params.bin path."""
    script_dir = script_dir or os.path.dirname(os.path.abspath(__file__))
    cfg = _load_config(script_dir)
    paths = cfg["paths"]
    params_path = output_params or os.path.join(script_dir, paths["params"])
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    bin_dir = os.path.dirname(params_path)

    lm_tmp = os.path.join(bin_dir, "_params_lm.tmp.bin")
    vis_tmp = os.path.join(bin_dir, "_params_vision.tmp.bin")

    model, model_dir = _ensure_hf_model(script_dir, cfg)
    generate_lm_weights(model, lm_tmp, precision=_lm_precision(cfg))
    generate_vision_weights(model, vis_tmp, precision=_vision_precision(cfg))
    del model

    lm_json = lm_tmp.rsplit('.', 1)[0] + '.json'
    vis_json = vis_tmp.rsplit('.', 1)[0] + '.json'
    with open(lm_json) as f:
        lm_manifest = json.load(f)
    with open(vis_json) as f:
        vis_manifest = json.load(f)
    lm_size = os.path.getsize(lm_tmp)
    vis_size = os.path.getsize(vis_tmp)

    # Concatenate LM then vision into params.bin (raw region dump).
    CHUNK = 4 * 1024 * 1024
    with open(params_path, 'wb') as out:
        for tmp in (lm_tmp, vis_tmp):
            with open(tmp, 'rb') as src:
                while True:
                    buf = src.read(CHUNK)
                    if not buf:
                        break
                    out.write(buf)

    regions = {
        "lm":     {"offset": 0,       "size": lm_size,  "manifest": lm_manifest},
        "vision": {"offset": lm_size, "size": vis_size, "manifest": vis_manifest},
    }
    params_json = params_path.rsplit('.', 1)[0] + '.json'
    with open(params_json, 'w') as f:
        json.dump({"regions": regions}, f)

    for tmp in (lm_tmp, vis_tmp, lm_json, vis_json):
        try:
            os.remove(tmp)
        except OSError:
            pass
    print(f"Unified params.bin: lm={lm_size/1048576:.1f} MB + vision={vis_size/1048576:.1f} MB -> {params_path}")
    return params_path


def ensure_params_bin(script_dir: str, verbose: bool = True) -> str:
    """Return the params.bin path, generating it (and params.json) if absent.

    Called from weight init rather than from main() so EVERY entry point that
    needs weights gets them -- the runtime, the numeric harness, an import from
    a notebook. Both files are checked: a half-written pair from an interrupted
    generation is not usable, and regenerating is the only repair.
    """
    cfg = _load_config(script_dir)
    params_path = os.path.join(script_dir, cfg["paths"]["params"])
    json_path = params_path.rsplit(".", 1)[0] + ".json"
    if os.path.exists(params_path) and os.path.exists(json_path):
        return params_path
    if verbose:
        missing = [p for p in (params_path, json_path) if not os.path.exists(p)]
        print(f"  weight bin missing ({', '.join(os.path.basename(m) for m in missing)}); "
              f"generating from the HF checkpoint -- this downloads ~8 GB and "
              f"takes several minutes, once.")
    return weight_bin_generate(script_dir)
