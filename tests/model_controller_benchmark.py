"""Verify identical tokens and compare single/multiple engine model timing.

Weights must be available through the model's normal HF/params preparation.
Example: python tests/model_controller_benchmark.py --dev xdma1 --engines 2 --json results.json
"""

import argparse
import builtins
import sys
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
from pathlib import Path
import socket
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import user_dma_core as core
from andromeda_hw_info import decode_hardware_info
from multi_engine_decode import reset_engine_queues

PROMPT = "Solve 2x + 3 = 7. Reply with only the value of x."


def load_model(name):
    paths = {"llama": "models/llama3.2_1b/llama3.2_1b_test.py",
             "qwen": "models/qwen3_0.6b/qwen3_0.6b_test.py",
             "qwen_2b": "models/qwen3.5_2b/qwen3.5_2b_test.py",
             "qwen_vl": "models/qwen2.5_vl_3b/qwen2.5_vl_3b_test.py",
             "gemma": "models/gemma3/gemma3_test.py",
             "e2b": "models/gemma4_e2b/gemma4_e2b_test.py",
             "e4b": "models/gemma4_e4b/gemma4_e4b_test.py"}
    spec = importlib.util.spec_from_file_location(f"controller_benchmark_{name}", ROOT / paths[name])
    module = importlib.util.module_from_spec(spec)
    # Gemma4 has sibling mixins and changes builtins.print at import time.
    original_path, original_print = sys.path[:], builtins.print
    try:
        sys.path.insert(0, str((ROOT / paths[name]).parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
        builtins.print = original_print
    return module


def token_output(ue, tokens):
    tokens = [int(token) for token in tokens]
    return dict(token_ids=tokens, decoded_tokens=len(tokens),
                decoded_text_without_special_tokens=ue.tokenizer.decode(tokens, skip_special_tokens=True),
                pad_token_id=ue.tokenizer.pad_token_id)


def valid_token_output(result):
    tokens = result["token_ids"]
    return bool(tokens and result["decoded_text_without_special_tokens"].strip()
                and any(token != result.get("pad_token_id") for token in tokens))


def run_llama(module, engines, prompt, max_tokens):
    ue = module.Llama32_1b_UnifiedEngine(multi_core=engines)
    ue.prefill_seq = tuple(ue.tokenizer.encode(module._minimal_chat_prompt(prompt),
                                             add_special_tokens=False))
    ue.fpga_penalty = False
    ue.max_new_tokens = max_tokens
    ue._generated_tokens = list(ue.prefill_seq)
    ue.compile_llama()
    result = ue.run_llama()
    result.update(token_output(ue, result.pop("generated_token_ids")))
    result["model_base"] = ue._params_dram_base
    return result


def run_qwen(module, engines, prompt, max_tokens):
    ue = module.Qwen3_0_6b_UnifiedEngine(multi_core=engines)
    conversation = [{"role": "system", "content": ue._cfg.get("default_system_prompt", "You are a helpful assistant.")},
                    {"role": "user", "content": prompt}]
    rendered = ue.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    tokens = tuple(ue.tokenizer.encode(rendered, add_special_tokens=False))
    ue.fpga_penalty = False
    ue.max_new_tokens = max_tokens
    ue._generated_tokens = list(tokens)
    meta = ue.compile_instructions()
    base, size = ue.load_program_instructions_from_file(ue.instruction_paths()[0])
    module._validate_program_embedding_separation(
        size, ue.DRAM_ADDR_TOKEN_EMBEDDING, ue.EMBEDDING_DISPATCH_TABLE_SIZE, ue._program_base)
    if base != ue._program_base:
        raise RuntimeError("Qwen instruction image was loaded at the wrong base")
    preamble = ue.get_program_dram_addr()
    ue.allocate_program_dram(module._RUNTIME_PREAMBLE_BYTES)
    ue.build_embedding_dispatch_table(preamble)
    prefill_flops = meta["prefill_template_flops"] * (len(tokens) - 1) // max(meta["prefill_template_seq_len"], 1)
    ue.run_prefill(module._parse_offset(meta["prefill_program_start_addr"]), preamble,
                   prefill_seq=tokens, gflops=prefill_flops)
    timings = []
    original = ue.program_execute
    def measured_execute(*args, **kwargs):
        result = original(*args, **kwargs)
        timings.append(result[0])
        return result
    ue.program_execute = measured_execute
    ue.run_decoder(module._parse_offset(meta["decoder_program_start_addr"]), preamble,
                   token_id=tokens[-1], gflops_per_token=meta["decoder_total_flops"])
    generated = ue._generated_tokens[len(tokens):]
    return dict(**token_output(ue, generated), decoded_text=ue.tokenizer.decode(generated), decode_first_hw_ms=timings[0] / 1000,
                decode_avg_hw_ms=sum(timings) / len(timings) / 1000,
                model_base=ue._params_dram_base,
                private_windows=ue.controller_decoder.scheduler.arena.windows
                if ue.controller_decoder is not None else None)


def run_qwen_vl(module, engines, prompt, max_tokens):
    # Match the model CLI's known initial DRAM state before weight uploads.
    module.clean_dram()
    ue = module.Qwen25VL_UnifiedEngine(multi_core=engines)
    ue.lm_weight_init()
    ue.lm_tensor_init()
    rendered = ue.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    tokens = ue.tokenizer(rendered)["input_ids"]
    context, seed = tokens[:-1], tokens[-1]
    ue.compile_prefill(len(context))
    ue.compile_decoder()
    ue.check_master_isa()
    ue.run_prefill(context)
    generated = []
    original = ue._decode_token

    def measured_argmax():
        token = int(original())
        generated.append(token)
        return token

    ue._decode_token = measured_argmax
    _, text = ue.run_decoder(seed, max_new_tokens=max_tokens)
    timings = ue._decode_step_us
    if not timings or len(generated) != len(timings):
        raise RuntimeError("Qwen VL decode tokens and hardware timings do not match")
    return dict(**token_output(ue, generated), decoded_text=text,
                decode_first_hw_ms=timings[0] / 1000,
                decode_avg_hw_ms=sum(timings) / len(timings) / 1000,
                model_base=ue.PARAMS_BASE, private_windows=ue._board_windows)


def run_qwen_2b(module, engines, prompt, max_tokens):
    import torch
    from transformers import AutoTokenizer

    script_dir = module.CONFIG_PATH.parent
    config = json.loads(module.CONFIG_PATH.read_text())
    model_dir = module._ensure_hf_model(str(script_dir), config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    weights_path = script_dir / config["paths"]["weights_bin"]
    if weights_path.exists():
        weights = torch.load(weights_path, weights_only=False, map_location="cpu")
    else:
        hf, text_model = module._load_hf_model(model_dir)
        weights = module._extract_all_weights(
            text_model, set(config["model"]["linear_attn_layer_indices"]))
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights, weights_path)
        del hf, text_model

    ue = module.Qwen3_5_2b_UnifiedEngine(
        device="cpu", init_unified_engine=False, multi_core=engines)
    ue.fpga_penalty = False
    ue._preallocate_identity_matrix()
    ue.prepare_inference(weights, max_context=config["model"]["max_context_size"])
    # A benchmark must compile both engine counts for this board and context.
    # Keep transient output separate from the model CLI's packaged programs.
    with tempfile.TemporaryDirectory(prefix="qwen35-benchmark-") as directory:
        ue._decoder_bin_path = str(Path(directory) / "decoder.bin")
        ue._decoder_meta_path = str(Path(directory) / "decoder.json")
        path, _, _ = ue.compile_decoder()
        ue.load_instructions(path)
    ids, _ = module._tokenize_with_chat_template(tokenizer, prompt)
    if len(ids) + max_tokens > ue.max_context:
        raise ValueError("Qwen3.5 prompt plus generated tokens exceeds the configured context")
    first_token = module.prefill_via_decode(ue, ids)
    result = ue.run_decoder(tokenizer, first_token, max_new_tokens=max_tokens)
    timings = ue._decode_step_us
    if not timings:
        raise RuntimeError("Qwen3.5 2B produced no decode timings")
    if len(timings) != len(result["token_ids"]) - 1:
        raise RuntimeError("Qwen3.5 2B tokens and decode timings do not match")
    if any(not math.isfinite(value) or value <= 0 for value in timings):
        raise RuntimeError("Qwen3.5 2B reported an invalid hardware timing")
    return dict(**token_output(ue, result["token_ids"]),
                decoded_text=result["generated_text"],
                decode_first_hw_ms=timings[0] / 1000,
                decode_avg_hw_ms=sum(timings) / len(timings) / 1000,
                timed_decode_steps=len(timings), model_base=ue._params_dram_base,
                private_windows=ue._decode_windows)


def run_gemma(module, engines, prompt, max_tokens):
    ue = module.Gemma3_UnifiedEngine(multi_core=engines)
    ue.set_prefill_seq(prompt)
    ue.setup_multi_core()
    ue.compile_gemma3(bin_reuse=False)
    tokens = []
    owner = ue if engines == 1 else ue.shard_group
    method = "get_arg_max_index" if engines == 1 else "global_argmax"
    original = getattr(owner, method)
    def measured_argmax(*args, **kwargs):
        token = int(original(*args, **kwargs))
        tokens.append(token)
        return token
    setattr(owner, method, measured_argmax)
    result = ue.run_gemma3()
    result.update(token_output(ue, tokens))
    result.update(model_base=ue._model_base, private_windows=ue._board_windows)
    return result


def run_e2b(module, engines, prompt, max_tokens):
    module.poison_dram("zero", quiet=True)
    ue = module.Gemma4_UnifiedEngine(multi_core=engines, prefill_kernel="matmatmul")
    ue.set_prefill_seq(prompt)
    ue.max_new_tokens = max_tokens
    ue.compile_gemma4(layer_size=ue.LAYER_SIZE)
    ue.run_gemma4()
    return dict(**token_output(ue, ue._decoded_token_ids), decoded_text=ue._decoded_text,
                decode_first_hw_ms=1000 / ue._decode_peak_toks,
                decode_avg_hw_ms=ue._decode_hw_latency_us / ue._decode_generated_n / 1000,
                model_base=ue._params_dram_base, dram_layout=ue.dram_layout,
                private_windows=(ue.mc_arena.windows or [(r.base, ue.mc_arena.stride) for r in ue.mc_arena.regions])
                if engines > 1 else None)


def run_e4b(module, engines, prompt, max_tokens):
    # Start with deterministic scratch/unused KV contents on both runs.
    raw = core.UnifiedEngine(init_unified_engine=False)
    zeros = bytes(64 << 20)
    for address in range(0, core.AVAILABLE_DRAM_SIZE_GB << 30, len(zeros)):
        written = raw.dma_write(core.DMA_DEVICE_H2C, address, zeros, len(zeros))
        if written != len(zeros):
            raise IOError(f"DRAM initialization at 0x{address:X}: {written}/{len(zeros)} bytes")
    del raw, zeros
    # Keep the original E4B prefill, down projection and LM-head kernels.
    ue = module.Gemma4_UnifiedEngine(multi_core=engines)
    ue.max_new_tokens = max_tokens
    ue.compile_instruction_bin(layer_size=ue.LAYER_SIZE)
    manifest = ue.load_instruction_bin()
    ue.set_prefill_seq(prompt)
    ue.run_prefill_bucketed(manifest)
    ue.run_decoder([manifest["decoder_program_size"]], manifest["_decoder_addr_int"],
                   ue.prefill_seq[-1], [manifest["decoder_total_flops"]])
    timings = ue._decode_step_us
    if not timings:
        raise RuntimeError("Gemma4 E4B produced no decode timings")
    return dict(**token_output(ue, ue._decoded_token_ids), decoded_text=ue._decoded_text,
                decode_first_hw_ms=timings[0] / 1000,
                decode_avg_hw_ms=sum(timings) / len(timings) / 1000,
                model_base=ue._params_dram_base, dram_layout=ue.dram_layout,
                private_windows=ue._decode_windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", default="xdma1")
    parser.add_argument("--engines", type=int, default=2)
    parser.add_argument("--models", nargs="+", choices=("qwen", "qwen_2b", "qwen_vl", "llama", "gemma", "e2b", "e4b"),
                        default=["qwen", "llama", "gemma"])
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Token limit for Qwen/Llama/Gemma4; Gemma3 runs to its stop token")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    core.set_dma_device(args.dev)
    core.configure_clock_from_hardware()
    if not 2 <= args.engines <= core.ANDROMEDA_CORE_COUNT:
        parser.error(f"--engines must be 2..{core.ANDROMEDA_CORE_COUNT}")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    info = decode_hardware_info(core.HW_INFO_RAW)
    version = core.UnifiedEngine(init_unified_engine=False).read_reg32(core.UE_FPGA_VERSION_ADDR)
    payload = dict(hostname=socket.gethostname(), device=args.dev, image=f"0x{version:08x}",
                   hardware_info=asdict(info), timestamp_utc=datetime.now(timezone.utc).isoformat(),
                   prompt=args.prompt, max_new_tokens=args.max_new_tokens, models={})
    try:
        for name in args.models:
            module = load_model(name)
            runs = []
            for engines in (1, args.engines):
                reset_engine_queues(args.engines)
                start = time.monotonic()
                result = globals()[f"run_{name}"](module, engines, args.prompt, args.max_new_tokens)
                result.update(active_engines=engines, wall_seconds=time.monotonic() - start)
                runs.append(result)
                payload["models"][name] = dict(runs=runs)
                args.json.write_text(json.dumps(payload, indent=2) + "\n")
                gc.collect()
            single, multi = runs
            comparison = payload["models"][name]
            comparison.update(exact_token_match=single["token_ids"] == multi["token_ids"],
                              meaningful_output=all(valid_token_output(run) for run in runs),
                              first_token_speedup=single["decode_first_hw_ms"] / multi["decode_first_hw_ms"],
                              average_token_speedup=single["decode_avg_hw_ms"] / multi["decode_avg_hw_ms"])
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
            if not comparison["exact_token_match"] or not comparison["meaningful_output"]:
                raise AssertionError(f"{name}: single/multiple engine token mismatch or empty/padding-only output")
            print(f"{name}: exact token PASS; first-token speedup {comparison['first_token_speedup']:.3f}x", flush=True)
    except BaseException as error:
        payload["error"] = dict(type=type(error).__name__, message=str(error))
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        reset_engine_queues(args.engines)
        raise
    print(f"Results: {args.json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
