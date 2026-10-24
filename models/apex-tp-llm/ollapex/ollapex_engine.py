"""OllApeX engine: keeps a Qwen3.5-family model loaded tensor-parallel on the FPGA cards and
generates tokens on request. All model compute runs on the cards.

Three kinds of processes:
  server (ollapex_server.py)  HTTP, tokenizer, chat template
  driver (this file)          one per loaded model; owns the token loop, pinned to CPU 0
  card workers (this file)    one per card; each drives its card's slice of every layer

The driver and the workers talk through shared memory (see ../run_tp.py for why). Between
requests the workers are parked: they poll slowly instead of spinning.
"""
import ctypes
import glob
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing import shared_memory

import numpy as np
import torch

EXP = os.path.join(os.path.expanduser(os.environ.get("APEX_TP_WORK", "~/apex_tp")), "models")
LOGDIR = os.environ.get("OLLAPEX_LOGDIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
BF16 = 2
INST = 32
SEG_QUIT, SEG_RESET, SEG_PARK, SEG_VISION, SEG_ROPE = -1, -2, -3, -4, -5
VIS_REQ, VIS_OUT, VIS_META = "/dev/shm/ollapex_vis_req.json", "/dev/shm/ollapex_vis_out.npy", "/dev/shm/ollapex_vis_meta.json"
ROPE_FILE = "/dev/shm/ollapex_rope.npz"
FAILED = -1
MAX_RANK = 4                      # the card tracks its four best logits
MAX_CTX = 16384                   # ctx16k patch: one KV copy per KV head, persistent V^T, one mask row

# name -> (directory under $APEX_TP_WORK/models, sliced weight set, display details)
MODEL_MAX_CTX = {"qwen3.8:27b": 8192, "gpt-oss:20b": 8192}   # per-model cap below MAX_CTX (card memory per engine)

MODELS = {
    "qwen3.8:27b": ("q38_27b", "10fp4", {"family": "qwen3.8", "parameter_size": "27B"}),
    "qwen3.5:9b": ("q35_9b", "10fp4", {"family": "qwen3.5", "parameter_size": "9B"}),
    "gpt-oss:20b": ("gptoss20b", "10", {"family": "gpt-oss", "parameter_size": "20B (MoE, 3.6B active)",
                                        "quantization_level": "FP4"}),
}
GPTOSS_HF = os.path.expanduser(os.environ.get("GPTOSS_MODEL", "~/apex_exp/models/gpt_oss_20b/hf/gpt-oss-20b"))


def is_gptoss(name):
    return MODELS[name][0] == "gptoss20b"


def model_paths(name):
    d, tp, _ = MODELS[name]
    tp = os.environ.get("OLLAPEX_TP_" + d.upper(), tp)   # e.g. OLLAPEX_TP_Q35_9B=5v2 on a 5-card host
    base = os.path.join(EXP, d)
    if is_gptoss(name):                               # its own card runtime (gptoss_card / gptoss_tp)
        return {"base": base, "tdir": os.path.join(base, "tp" + tp), "wdir": os.path.join(base, "weights"),
                "hf": GPTOSS_HF}
    hf = os.path.dirname(sorted(glob.glob(os.path.join(base, "*_bin", "*", "config.json")))[0])
    return {"base": base, "tdir": os.path.join(base, "tp" + tp), "wdir": os.path.join(base, "weights"),
            "port": os.path.join(base, "qwen3.5_2b_test.py"), "hf": hf,
            "config": os.path.join(base, "qwen3.5_2b_config.json"),
            "vision": os.path.join(base, "vision", "vision_weights.pt")}


def available_models():
    out = []
    for name in MODELS:
        try:
            p = model_paths(name)
            if os.path.exists(os.path.join(p["tdir"], "plan.json")):
                out.append(name)
        except IndexError:
            pass
    return out


def cards_in_pci_order():
    out = []
    for u in glob.glob("/sys/class/xdma/xdma*_user"):
        dev = os.path.basename(u)[:-len("_user")]
        bdf = os.path.basename(os.path.realpath(os.path.join(u, "device")))
        rd = lambda f: open("/sys/bus/pci/devices/%s/%s" % (bdf, f)).read().strip()
        out.append((bdf, dev, "%s x%s" % (rd("current_link_speed").replace(" PCIe", ""), rd("current_link_width"))))
    return sorted(out)


def cpu_plan(P):
    """CPU of the driver and of each card process. APEX_TP_CPUS="8,9,10,..." (driver first) keeps them
    off the CPUs the xdma driver pins its completion threads to in poll mode (CPUs 0-7 on the Z420)."""
    env = os.environ.get("APEX_TP_CPUS", "")
    if env:
        c = [int(v) for v in env.split(",")]
        return c[0], [c[1 + k % (len(c) - 1)] for k in range(P)]
    n = os.cpu_count()
    return 0, [1 + k % (n - 1) for k in range(P)]


class Shared:
    """Command block written by the driver, one result row per card."""

    def __init__(self, P, H, names=None):
        spec = (("cmd", np.int64, (4,)),            # sequence number, segment, position (-1: none), candidates wanted
                ("x", np.int16, (H,)),              # input vector for the segment (bf16 bits)
                ("done", np.int64, (P,)),           # sequence number each card finished last (FAILED on error)
                ("meta", np.float64, (P, 2 + 2 * MAX_RANK)),   # card ms, round-trip ms, then (index, logit) per rank
                ("out", np.int16, (P, H)))          # each card's partial output (bf16 bits)
        self.blocks, self.names = [], []
        for i, (name, dtype, shape) in enumerate(spec):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if names is None:
                shm = shared_memory.SharedMemory(create=True, size=size)
            else:
                try:
                    shm = shared_memory.SharedMemory(name=names[i], track=False)   # the driver owns it
                except TypeError:
                    shm = shared_memory.SharedMemory(name=names[i])
            self.blocks.append(shm)
            self.names.append(shm.name)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            if names is None:
                arr[...] = 0
            setattr(self, name, arr)

    def close(self, unlink=False):
        for name in ("cmd", "x", "done", "meta", "out"):
            delattr(self, name)
        for shm in self.blocks:
            try:
                shm.close()
                if unlink:
                    shm.unlink()
            except Exception:
                pass


class LazyLayers:
    """The card's layer slices, read from disk one at a time (a 27B card holds over 1 GB of them)."""

    def __init__(self, cdir, n):
        self.cdir, self.n = cdir, n

    def __len__(self):
        return self.n

    def __getitem__(self, g):
        if isinstance(g, slice):
            return [self[i] for i in range(*g.indices(self.n))]
        if g < 0:
            g += self.n
        if not 0 <= g < self.n:
            raise IndexError(g)
        return torch.load(os.path.join(self.cdir, "layer_%02d.pt" % g), weights_only=False, map_location="cpu")


ENGINE_STRIDE = 0x10000          # AXI-Lite register block of engine e: UE_0_BASE_ADDR + e * ENGINE_STRIDE
CHANNEL_BYTES = 0x8000_0000      # Kintex-7: address bit 31 picks the DDR3 channel (2 GiB each)
PROGRAM_BYTES = 64 << 20         # per engine, at the top of its window


def engine_window(e, E, cdir):
    """DRAM layout of engine e of E on one card. With two engines, engine e lives entirely in DDR
    channel e, so both engines read weights at full speed at the same time (5.7 GB/s each)."""
    if E == 1:
        return {}                                  # single engine: the port's membase layout
    w = e * CHANNEL_BYTES
    on_disk = sum(os.path.getsize(f) for f in glob.glob(os.path.join(cdir, "*.pt")))
    params = (int(on_disk * 1.03) + (96 << 20) + (16 << 20) - 1) // (16 << 20) * (16 << 20)
    assert params < CHANNEL_BYTES - PROGRAM_BYTES - (256 << 20), "the slice does not fit one DDR channel"
    return {"params_dram_base": w, "tensor_dram_base": w + params,
            "program_dram_base": w + CHANNEL_BYTES - PROGRAM_BYTES}


def worker(c, E, P, dev, paths, n_layers, max_ctx, cpu, nap, shm_names, conn, vision_k=-1):
    """One process per card; it drives the card's E engines. Engine e of card c is TP shard k = c*E + e."""
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, "card_%02d.log" % c), "w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    sh = None
    ks = [c * E + e for e in range(E)]
    # presum: after each segment engine 0 adds engine 1's share into its own on the card, so the host
    # reads one share per card and writes one per card back (half the PCIe traffic)
    presum = E == 2 and os.environ.get("APEX_TP_PRESUM", "1") != "0"
    try:
        os.sched_setaffinity(0, {cpu})
        ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)     # PR_SET_TIMERSLACK = 1 ns: short sleeps end on time
        torch.set_num_threads(1)
        t0 = time.time()
        eng = []
        # build every engine before any weights go up: constructing one self-tests the first 16 KB of DRAM
        for e, k in enumerate(ks):
            spec = importlib.util.spec_from_file_location("ollapex_port_%d" % k, paths["port"])
            T = importlib.util.module_from_spec(spec)
            sys.modules["ollapex_port_%d" % k] = T
            spec.loader.exec_module(T)
            T.set_dma_device(dev)
            T.DMA_DEVICE_H2C = T.user_dma_core.DMA_DEVICE_H2C
            T.DMA_DEVICE_C2H = T.user_dma_core.DMA_DEVICE_C2H
            T.user_dma_core.configure_clock_from_hardware()
            cdir = os.path.join(paths["tdir"], "card_%02d" % k)
            ue = T.Qwen3_5_2b_UnifiedEngine(device="cpu", config_path=os.path.join(cdir, "config.json"),
                                            BASE_ADDR=T.user_dma_core.UE_0_BASE_ADDR + e * ENGINE_STRIDE,
                                            **engine_window(e, E, cdir))
            eng.append({"T": T, "ue": ue, "k": k, "e": e, "cdir": cdir, "tensor0": ue._tensor_dram_addr})
        T = eng[0]["T"]
        cfg = json.load(open(os.path.join(eng[0]["cdir"], "config.json")))
        H = cfg["file_info"]["hidden_size"]
        for g in eng:
            ue, cdir = g["ue"], g["cdir"]
            g["vocab"] = json.load(open(os.path.join(cdir, "config.json")))["file_info"]["embedding_vocab"]
            weights = {
                "layers": LazyLayers(cdir, n_layers),
                "final_norm_gamma": torch.load(os.path.join(paths["wdir"], "final_norm.pt"), weights_only=False),
                "embed_weight": torch.zeros(g["vocab"], 1, dtype=torch.bfloat16),   # only its row count is used
            }
            weights["lm_head_scale"], weights["lm_head_data"] = torch.load(os.path.join(cdir, "lm_head.pt"),
                                                                           weights_only=False)
            ue._tp_mode = True
            ue._tp_segments = []
            ue._tp_card_sum = True                    # residual kept and summed on the card (cardsum patch)
            ue._tp_cards = P // E if presum else P    # presum: one combined share per card
            ue.software_reset()
            ue._preallocate_identity_matrix()
            ue.prepare_inference(weights, max_context=max_ctx)
            del weights
        for g in eng[1:]:                             # one copy of the shards' partial outputs per card
            g["ue"]._tp_parts_dram = eng[0]["ue"]._tp_parts_dram
        info = []
        for g in eng:
            T_e, ue, cdir, k = g["T"], g["ue"], g["cdir"], g["k"]
            ue._decoder_bin_path = os.path.join(cdir, "decoder_ctx%d.bin" % max_ctx)
            ue._decoder_meta_path = os.path.join(cdir, "decoder_ctx%d.json" % max_ctx)
            for p in (ue._decoder_bin_path, ue._decoder_meta_path):
                if os.path.exists(p):
                    os.remove(p)
            bin_path, sizes, _ = ue.compile_decoder()
            ue.load_instructions(bin_path)
            segs = list(ue._tp_segments)
            assert len(segs) == 2 * n_layers, "expected two cuts per layer, got %d" % len(segs)
            g["starts"] = [ue._decoder_prog_addr] + [ue._decoder_prog_addr + idx * INST for idx, _ in segs]
            g["outs"] = [addr for _, addr in segs]   # segment s < 2L writes H values here; the last one is the head
            g["est"] = [0.0] * len(g["starts"])      # last measured run time of each segment, microseconds
            g["pos_prog"] = ue.get_program_dram_addr()   # fixed slot for the per-token position program
            ue.allocate_program_dram(8 * INST)
            # The LM never touches tensor memory above this address: that is where the vision encoder lives.
            lm_tensor_end = (ue._tensor_dram_addr + 0xFFFF) & ~0xFFFF
            tensor_limit = ue._program_dram_base
            g["vision"] = None
            if k == vision_k:
                import ollapex_vision
                g["vision"] = ollapex_vision.VisionCard(T_e, ue, paths["vision"], lm_tensor_end, tensor_limit)
            info.append({"dev": dev if E == 1 else "%s/e%d" % (dev, g["e"]), "segs": len(g["starts"]),
                         "setup_s": time.time() - t0, "prog_mb": sizes[0] / 1e6,
                         "params_mb": (ue._next_params_dram_addr - ue._params_dram_base) / 1e6,
                         # for aptop: K, V and V^T caches of the full-attention layers, and all tensor memory in use
                         "kv_mb": len(ue.full_attn_layers) * 3 * ue.full_num_kv_heads * ue.max_context_aligned
                                  * ue.full_head_dim * BF16 / 1e6,
                         "tensor_mb": (lm_tensor_end - g["tensor0"]) / 1e6,
                         "instructions": sizes[0] // INST, "tensor_free_mb": (tensor_limit - lm_tensor_end) / 1048576.0,
                         "clock_mhz": 1000.0 / float(getattr(ue, "_clock_period_ns", 0) or 5.042),
                         "firmware": "%08x" % ue.get_hardware_version()})
        n_seg = len(eng[0]["starts"])
        ue0 = eng[0]["ue"]
        T_al = ue0.max_context_aligned
        row_byt = ue0.full_head_dim * BF16
        rope_byt = ue0.full_rotary_dim * BF16
        rot_dim = ue0.full_rotary_dim
        add_progs = []
        if presum:
            # per segment, a tiny engine-0 program: share(engine 0) += share(engine 1), in URAM
            U, nbytes = T.user_dma_core, H * BF16
            for s in range(n_seg - 1):
                a0, a1 = eng[0]["outs"][s], eng[1]["outs"][s]
                ue0.clear_capture_buffer()
                U.UnifiedEngine.start_capture(ue0)
                ue0.ue_memcpy_from_dram(a0, nbytes, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_A.value)
                ue0.ue_memcpy_from_dram(a1, nbytes, U.MEMCPY_TYPE.URAM.value, U.URAM_START_ADDR, U.URAM_SECTION.URAM_B.value)
                ue0.eltwise_add_core(0x00000, 0x80000, 0x00000, H)
                ue0.ue_memcpy_to_dram(U.MEMCPY_TYPE.URAM.value, U.URAM_SECTION.URAM_A.value, U.URAM_START_ADDR, a0, nbytes)
                ue0.generate_instruction_halt()
                U.UnifiedEngine.stop_capture(ue0)
                addr = ue0.get_program_dram_addr()
                ue0.write_captured_instructions_to_dram(addr)
                ue0.allocate_program_dram(ue0.get_capture_instruction_size_bytes() + 64)
                ue0.clear_capture_buffer()
                add_progs.append(addr)

        sh = Shared(P, H, shm_names)
        x_t = torch.from_numpy(sh.x).view(torch.bfloat16)
        n_parts = P // E if presum else P
        parts_t = torch.from_numpy(sh.out[:n_parts]).view(torch.bfloat16)   # the shares the cards add
        # presum double-buffers the combined shares by segment parity (P = 2 x cards rows): a fast card
        # writing segment s's share never overwrites a row a slow card is still uploading for segment s
        parity_t = [torch.from_numpy(sh.out[i * n_parts:(i + 1) * n_parts]).view(torch.bfloat16)
                    for i in range(2 if presum else 1)]
        parity_out = [torch.from_numpy(sh.out[i * n_parts + c]).view(torch.bfloat16) for i in range(len(parity_t))]
        bias = torch.empty((1, T_al), dtype=torch.bfloat16)
        for g in eng:
            g["out_t"] = torch.from_numpy(sh.out[g["k"]]).view(torch.bfloat16)

        def run_all(addr_of, expect_us, limit_s):
            for g in eng:                             # start every engine, then wait for all of them
                g["ue"].start_execute_from_dram(addr_of(g))
            t_wait = time.time()
            if nap and expect_us > 2 * nap:
                time.sleep((expect_us - nap) * 1e-6)
            for g in eng:
                while g["ue"].is_queue_busy():
                    os.sched_yield()
                    if time.time() - t_wait >= limit_s:
                        raise RuntimeError("engine program at 0x%x did not finish within %.0f s" % (addr_of(g), limit_s))

        def write_position(g, pos):
            T_e, ue = g["T"], g["ue"]
            ue.dma_write(T_e.DMA_DEVICE_H2C, ue._bias_dram, bias, T_al * BF16)
            ue.reset_tensor_dram_addr()
            ue.clear_capture_buffer()
            T_e.UnifiedEngine.start_capture(ue)
            ue.generate_instruction_add_set(ue._ISA_POS_K_REG, T_e.ue_35bit_addr_shifter(pos * row_byt))
            ue.generate_instruction_add_set(ue._ISA_POS_ROPE_REG, T_e.ue_35bit_addr_shifter(pos * rope_byt))
            if hasattr(ue, "_ISA_ATTN_LEN_REG"):      # dynlen: attend over the filled cache only
                ue.generate_instruction_add_set(ue._ISA_ATTN_LEN_REG, min(T_al, (pos + 64) // 64 * 64))
            if hasattr(ue, "_ISA_BLK_V_REG"):         # ctx16k: the 64-row V block that holds pos, and its V^T columns
                blk, ch = pos // 64 * 64, ue._vt_chunk
                ue.generate_instruction_add_set(ue._ISA_BLK_V_REG, T_e.ue_35bit_addr_shifter(blk * row_byt))
                ue.generate_instruction_add_set(ue._ISA_BLK_VT_REG, T_e.ue_35bit_addr_shifter(
                    (blk // ch) * ue.full_head_dim * ch * BF16 + (blk % ch) * BF16))
            ue.generate_instruction_halt()
            T_e.UnifiedEngine.stop_capture(ue)
            ue.write_captured_instructions_to_dram(g["pos_prog"])
            ue.clear_capture_buffer()

        conn.send(("ready", info))
        last = 0
        parked = True
        cmd, done, meta = sh.cmd, sh.done, sh.meta
        with T._quiet():
            while True:
                while cmd[0] == last:
                    if parked:
                        time.sleep(0.002)             # idle: do not burn a CPU
                    else:
                        os.sched_yield()
                last = int(cmd[0]); s = int(cmd[1]); pos = int(cmd[2]); want = int(cmd[3])
                if s == SEG_QUIT:
                    break
                if s == SEG_PARK:
                    parked = True
                    for k in ks:
                        done[k] = last
                    continue
                parked = False
                if s == SEG_RESET:
                    for g in eng:
                        g["ue"].reset_state()
                    for k in ks:
                        done[k] = last
                    continue
                if s == SEG_VISION:
                    for g in eng:
                        if pos == g["k"] and g["vision"] is not None:
                            try:
                                rq = json.load(open(VIS_REQ))
                                toks, vi = g["vision"].encode(rq["image"], rq["W"], rq["H"])
                                np.save(VIS_OUT, toks)
                                json.dump(vi, open(VIS_META, "w"))
                            except Exception:
                                traceback.print_exc()
                                json.dump({"error": traceback.format_exc()[-1500:]}, open(VIS_META, "w"))
                    for k in ks:
                        done[k] = last
                    continue
                if s == SEG_ROPE:
                    # position rows of the attention layers' rotary tables, rewritten for image positions
                    z = np.load(ROPE_FILE)
                    off = int(z["i0"]) * rot_dim * BF16
                    for g in eng:
                        T_e, ue = g["T"], g["ue"]
                        for name, addr in (("cos", ue._fa_cos_pad_dram), ("nsin", ue._fa_neg_sin_pad_dram),
                                           ("sin", ue._fa_sin_hi_pad_dram)):
                            t = torch.from_numpy(np.ascontiguousarray(z[name])).view(torch.bfloat16)
                            ue.dma_write(T_e.DMA_DEVICE_H2C, addr + off, t, t.numel() * BF16)
                    for k in ks:
                        done[k] = last
                    continue
                t1 = time.perf_counter()
                if pos >= 0:                          # first segment of a token
                    bias.fill_(float("-inf"))
                    bias[:, :pos + 1] = 0.0
                    for g in eng:
                        write_position(g, pos)
                    run_all(lambda g: g["pos_prog"], 0.0, 5.0)
                    for g in eng:                     # embedding row of the token
                        g["ue"].dma_write(g["T"].DMA_DEVICE_H2C, g["ue"]._decoder_X_dram, x_t, H * BF16)
                else:                                 # every share, once per card; the engines add them
                    src = parity_t[(s - 1) % 2] if presum else parts_t
                    ue0.dma_write(T.DMA_DEVICE_H2C, ue0._tp_parts_dram, src, n_parts * H * BF16)
                run_all(lambda g: g["starts"][s], max(g["est"][s] for g in eng), 60.0)
                hw_seg = [g["ue"].report_latency_in_us() for g in eng]
                if presum and s < n_seg - 1:
                    ue0.start_execute_from_dram(add_progs[s])
                    while ue0.is_queue_busy():
                        pass
                    ue0.dma_read(T.DMA_DEVICE_C2H, eng[0]["outs"][s], parity_out[s % 2], H * BF16)
                for g, hw_us in zip(eng, hw_seg):
                    T_e, ue, k = g["T"], g["ue"], g["k"]
                    g["est"][s] = hw_us
                    if s < n_seg - 1:
                        if not presum:
                            ue.dma_read(T_e.DMA_DEVICE_C2H, g["outs"][s], g["out_t"], H * BF16)
                    else:
                        for r in range(max(1, min(want, MAX_RANK))):
                            idx = int(ue.get_arg_max_index(rank=r + 1))
                            if not 0 <= idx < g["vocab"]:
                                idx = 0
                            blk = ue.dma_from_accelerator_memory(ue.LOGITS_DRAM + (idx // 32) * 32 * BF16,
                                                                 torch.Size([1, 32]))
                            meta[k, 2 + 2 * r] = idx
                            meta[k, 3 + 2 * r] = float(blk.float()[0, idx % 32])
                    meta[k, 0] = hw_us / 1e3
                    meta[k, 1] = (time.perf_counter() - t1) * 1e3
                for k in ks:
                    done[k] = last
    except Exception:
        traceback.print_exc()
        try:
            if sh is not None:
                for k in ks:
                    sh.done[k] = FAILED
            conn.send(("error", traceback.format_exc()[-1500:]))
        except Exception:
            pass


class Loaded:
    """A model resident on the cards (driver side)."""

    def __init__(self, name, num_ctx, emit, nap_us=30.0):
        # 27B on 5 cards x 2 engines: weights take ~1.6 of each engine's 2 GiB, 16K of KV/V^T does not fit
        num_ctx = min(num_ctx, MODEL_MAX_CTX.get(name, MAX_CTX))
        self.name, self.num_ctx = name, num_ctx
        self.paths = model_paths(name)
        plan = json.load(open(os.path.join(self.paths["tdir"], "plan.json")))
        self.P, self.L, self.H = plan["P"], plan["layers"], plan["hidden"]
        self.voff = [c["vocab"][0] for c in plan["cards"]]
        self.plan = plan["cards"]
        # P is the number of TP shards; each card runs E of them, one per engine (each in its own DDR channel)
        self.E = int(os.environ.get("OLLAPEX_ENGINES", "1"))
        if self.P % self.E:
            raise RuntimeError("%d shards do not split into %d engines per card" % (self.P, self.E))
        n_cards = self.P // self.E
        cards = cards_in_pci_order()
        if len(cards) < n_cards:
            raise RuntimeError("the model needs %d cards, %d cards are visible" % (n_cards, len(cards)))
        self.phys = cards[:n_cards]
        self.cards = [self.phys[k // self.E] for k in range(self.P)]   # per shard: (bdf, dev, link) of its card
        # the image encoder lives on the last shard (it carries the fewest heads)
        self.vision_k = self.P - 1 if os.path.exists(self.paths["vision"]) else -1
        cfg = json.load(open(self.paths["config"]))
        self.rot_dim = cfg["file_info"]["full_attn_rotary_dim"]
        self.rope_theta = float(cfg["special"]["rope"]["theta_full"])
        hf = json.load(open(os.path.join(self.paths["hf"], "config.json")))
        self.mrope = hf.get("text_config", {}).get("rope_parameters", {}).get("mrope_section", [11, 11, 10])
        self.rope_from = None             # first position whose rotary row is not the default one
        ncpu = os.cpu_count()
        self.sh = Shared(self.P, self.H)
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        t0 = time.time()
        cpus = cpu_plan(n_cards)[1]
        card_pipes, card_procs = [], []
        for c, (bdf, dev, link) in enumerate(self.phys):
            pa, pb = ctx.Pipe()
            p = ctx.Process(target=worker, args=(c, self.E, self.P, dev, self.paths, self.L, num_ctx, cpus[c], nap_us,
                                                 self.sh.names, pb, self.vision_k), daemon=True)
            p.start()
            card_pipes.append(pa); card_procs.append(p)
        self.pipes = [card_pipes[k // self.E] for k in range(self.P)]   # per shard, for error reports
        self.procs = card_procs
        self.card_info = [None] * self.P
        pending = set(range(n_cards))
        while pending:
            for c in list(pending):
                if card_pipes[c].poll(0.2):
                    m = card_pipes[c].recv()
                    if m[0] != "ready":
                        raise RuntimeError("card %d (%s): %s" % (c, self.phys[c][1], m[1] if len(m) > 1 else m))
                    for e, i in enumerate(m[1]):
                        k = c * self.E + e
                        i.update({"index": k, "bdf": self.phys[c][0], "link": self.phys[c][2], "plan": self.plan[k]})
                        self.card_info[k] = i
                    pending.discard(c)
                    emit(("progress", {"ready": n_cards - len(pending), "cards": n_cards, "elapsed_s": time.time() - t0}))
                elif not card_procs[c].is_alive():
                    raise RuntimeError("the process of card %d ended unexpectedly (log: %s/card_%02d.log)" % (c, LOGDIR, c))
        self.n_seg = self.card_info[0]["segs"]
        self.load_s = time.time() - t0
        shape = json.load(open(os.path.join(self.paths["wdir"], "embed.json")))["shape"]
        # the whole table in RAM: a memmap row costs a random disk read (~28 ms on this SSD) once the page cache drops it
        self.embed = np.fromfile(os.path.join(self.paths["wdir"], "embed.bin"), dtype=np.int16).reshape(tuple(shape))
        self.seq = 0
        self.fed = []                 # what the cards' state already contains: one key per position
        self.f32 = np.empty(self.H, dtype=np.float32)
        self.command(SEG_RESET, -1)
        self.command(SEG_PARK, -1)

    def vision(self, image, W, H):
        """Run the image encoder on its card. Returns (tokens int16 [n, H], info)."""
        if self.vision_k < 0:
            raise ValueError("this model has no image support on the cards")
        for f in (VIS_OUT, VIS_META):
            if os.path.exists(f):
                os.remove(f)
        json.dump({"image": image, "W": W, "H": H}, open(VIS_REQ, "w"))
        self.command(SEG_VISION, self.vision_k, limit_s=900.0)
        meta = json.load(open(VIS_META))
        self.command(SEG_PARK, -1)
        if "error" in meta:
            raise ValueError("image encoder error: " + meta["error"].strip().split("\n")[-1])
        meta["card"] = self.cards[self.vision_k][1]
        return np.load(VIS_OUT), meta

    def set_rope(self, i0, pos3):
        """Rewrite the rotary table rows for sequence indexes i0.. on every card. pos3[j] is the
        (t, h, w) position of index i0+j; text has t == h == w, image tokens differ per axis.
        Frequency slots are interleaved T,H,W,T,H,W,... as in the model (mrope sections)."""
        pos3 = np.asarray(pos3, dtype=np.float32).reshape(-1, 3)
        half = self.rot_dim // 2
        inv = 1.0 / (self.rope_theta ** (np.arange(0, self.rot_dim, 2, dtype=np.float32) / self.rot_dim))
        axis = np.zeros(half, dtype=np.int64)                 # which of (t, h, w) drives each frequency slot
        axis[1:self.mrope[1] * 3:3] = 1
        axis[2:self.mrope[2] * 3:3] = 2
        fr = torch.from_numpy(pos3[:, axis] * inv[None, :])  # float32, as the port computes its table
        z = {"i0": np.int64(i0)}
        for name, val in (("cos", fr.cos()), ("nsin", -fr.sin()), ("sin", fr.sin())):
            buf = torch.zeros(pos3.shape[0], self.rot_dim, dtype=torch.bfloat16)
            buf[:, :half] = val.to(torch.bfloat16)
            z[name] = buf.view(torch.int16).numpy()
        np.savez(ROPE_FILE, **z)
        self.command(SEG_ROPE, -1)

    def info(self):
        return {"model": self.name, "num_ctx": self.num_ctx, "cards": self.card_info, "segments_per_token": self.n_seg,
                "vision": self.vision_k >= 0, "vision_card": self.vision_k,
                "layers": self.L, "hidden": self.H, "load_s": self.load_s,
                "weights_mb": sum(c["params_mb"] for c in self.card_info), "details": MODELS[self.name][2]}

    def command(self, s, pos, want=1, limit_s=120.0):
        sh = self.sh
        sh.cmd[1] = s
        sh.cmd[2] = pos
        sh.cmd[3] = want
        self.seq += 1
        sh.cmd[0] = self.seq              # the bump is what the cards watch for
        t = time.perf_counter()
        n = 0
        while not (sh.done == self.seq).all():
            n += 1
            if n & 0x3FF == 0:
                if (sh.done == FAILED).any():
                    k = int(np.argmax(sh.done == FAILED))
                    m = self.pipes[k].recv() if self.pipes[k].poll(2.0) else ("error", "no detail")
                    raise RuntimeError("card %d: %s" % (k, m[1]))
                if time.perf_counter() - t > limit_s:
                    raise RuntimeError("cards %s did not respond" % np.nonzero(sh.done != self.seq)[0].tolist())

    def forward(self, vec_i16, pos, want, acc):
        """One position through all layers. want == 0: skip the LM head (prompt tokens whose
        prediction nobody reads). Returns the candidate list [(logit, token)] or None."""
        sh = self.sh
        sh.x[:] = vec_i16                 # the token's embedding row; the cards keep the residual from here on
        last = self.n_seg - 1
        for s in range(self.n_seg if want else last):
            t = time.perf_counter()
            self.command(s, pos if s == 0 else -1, want)
            if acc is not None:
                acc["seg_wall"][s] += (time.perf_counter() - t) * 1e3
                acc["hw"][:, s] += sh.meta[:, 0]
                acc["wall"][:, s] += sh.meta[:, 1]
        if not want:
            return None
        cands = []
        for k in range(self.P):
            for r in range(max(1, min(want, MAX_RANK))):
                cands.append((float(sh.meta[k, 3 + 2 * r]), self.voff[k] + int(sh.meta[k, 2 + 2 * r])))
        return cands

    def close(self):
        try:
            self.sh.cmd[1] = SEG_QUIT
            self.sh.cmd[0] = self.seq + 1
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.sh.close(unlink=True)


def pick(cands, opts, rng):
    """Choose the next token from the cards' candidates. Greedy unless temperature > 0; then
    sample among the global top-k (k <= 4: each card reports its four best, so the global top
    four are always among them), optionally cut by top_p."""
    cands = sorted(cands, reverse=True)
    temp = float(opts.get("temperature", 0.0) or 0.0)
    if temp <= 0.0:
        return cands[0][1], cands[0][0]
    k = max(1, min(int(opts.get("top_k", MAX_RANK) or MAX_RANK), MAX_RANK))
    top = cands[:k]
    z = np.array([c[0] for c in top], dtype=np.float64) / temp
    p = np.exp(z - z.max())
    p /= p.sum()
    top_p = float(opts.get("top_p", 1.0) or 1.0)
    if top_p < 1.0:
        keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
        p = p[:keep] / p[:keep].sum()
        top = top[:keep]
    i = int(rng.choice(len(top), p=p))
    return top[i][1], top[i][0]


def seg_report(m, acc, n):
    """Per-token timing, grouped by what the segment does."""
    n = max(n, 1)
    hw, wall, sw = acc["hw"] / n, acc["wall"] / n, acc["seg_wall"] / n
    S = m.n_seg
    groups = (("linear_attention", [s for s in range(S - 1) if (s // 2) % 4 != 3 and s % 2 == 0]),
              ("linear_mlp", [s for s in range(S - 1) if (s // 2) % 4 != 3 and s % 2 == 1]),
              ("full_attention", [s for s in range(S - 1) if (s // 2) % 4 == 3 and s % 2 == 0]),
              ("full_mlp", [s for s in range(S - 1) if (s // 2) % 4 == 3 and s % 2 == 1]),
              ("lm_head", [S - 1]))
    segs = []
    for name, sel in groups:
        segs.append({"name": name, "count": len(sel), "card_ms_avg": float(hw[:, sel].mean()),
                     "card_ms_slowest": float(hw[:, sel].max(axis=0).mean()),
                     "roundtrip_ms_slowest": float(wall[:, sel].max(axis=0).mean()),
                     "host_ms": float(sw[sel].mean()), "per_token_ms": float(sw[sel].sum())})
    return {"segments": segs, "host_ms_per_token": float(sw.sum()),
            "slowest_card_compute_ms_per_token": float(hw.max(axis=0).sum()),
            "avg_card_compute_ms_per_token": float(hw.mean(axis=0).sum()),
            "cards": [{"index": k, "dev": m.cards[k][1], "compute_ms_per_token": float(hw[k].sum()),
                       "roundtrip_ms_per_token": float(wall[k].sum())} for k in range(m.P)]}


class AptopState:
    """Writes the aptop-workload/v1 state file that the aptop monitor reads: model, phase and per-card
    memory. aptop cannot see our cards' use from process file handles (the runtime opens the xdma
    nodes only for each transfer). Written to $XDG_RUNTIME_DIR (what an interactive aptop reads) and to
    /tmp/aptop-$UID (its fallback); observer_pid lets aptop retire the state if this process dies."""

    def __init__(self):
        uid = os.getuid()
        self.paths = [os.path.join("/run/user/%d" % uid, "aptop-workload.json"),
                      os.path.join("/tmp/aptop-%d" % uid, "workload.json")]
        self.doc = None

    @staticmethod
    def _now():
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _write(self):
        data = json.dumps(self.doc).encode()
        for p in self.paths:
            d = os.path.dirname(p)
            try:
                if p.startswith("/tmp/"):
                    os.makedirs(d, mode=0o700, exist_ok=True)
                if not os.path.isdir(d):
                    continue
                tmp = p + ".tmp"
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, data)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(tmp, p)
            except OSError:
                pass

    def loading(self, name):
        self.doc = {"schema": "aptop-workload/v1", "active": True, "model": name, "display_name": "%s (OllApeX)" % name,
                    "phase": "loading", "observer_pid": os.getpid(), "started_at": self._now(),
                    "updated_at": self._now(), "cards": []}
        self._write()

    def loaded(self, m):
        mib = 1e6 / 1048576.0
        by_dev = {}
        for i in m.card_info:
            dev = i["dev"].split("/")[0]
            r = by_dev.setdefault(dev, {"w": 0.0, "kv": 0.0, "t": 0.0, "p": 0.0})
            r["w"] += i["params_mb"]; r["kv"] += i.get("kv_mb", 0.0)
            r["t"] += i.get("tensor_mb", 0.0); r["p"] += i["prog_mb"]
        cards = []
        for dev, r in sorted(by_dev.items()):
            w, kv, t, p = r["w"] * mib, r["kv"] * mib, r["t"] * mib, r["p"] * mib
            known = w + max(t, kv) + p
            cards.append({"device": dev, "layer_start": 0, "layer_end_exclusive": m.L, "phase": "idle",
                          "runtime_memory": {"capacity_mib": 4096, "weight_mib": round(w, 1), "kv_cache_mib": round(kv, 1),
                                             "tensor_scratch_mib": round(max(t - kv, 0.0), 1), "program_mib": round(p, 1),
                                             "unreported_or_reserved_mib": round(max(4096 - known, 0.0), 1),
                                             "known_mapped_mib": round(known, 1),
                                             "known_mapped_percent": round(100.0 * known / 4096, 2)}})
        self.doc.update({"display_name": "%s, %d cards x %d engines, ctx %d" % (m.name, len(cards), m.E, m.num_ctx),
                         "cards": cards})
        self.phase("idle")

    def phase(self, ph):
        if self.doc is None:
            return
        self.doc["phase"] = ph
        self.doc["updated_at"] = self._now()
        for c in self.doc["cards"]:
            c["phase"] = ph
        self._write()

    def unloaded(self):
        if self.doc is not None:
            self.doc.update({"active": False, "phase": "unloaded", "completed_at": self._now(), "updated_at": self._now()})
            self._write()


def generate(m, req, emit, abort):
    """req: ids (token ids), vecs ({position: int16 bytes} for positions fed a vector instead of a
    token embedding, e.g. image tokens), opts, stop_ids."""
    ids, opts = req["ids"], req.get("opts", {})
    vecs = req.get("vecs") or {}
    stop_ids = set(req.get("stop_ids") or [])
    keys = [(-1, hashlib.md5(vecs[i]).hexdigest()) if i in vecs else (t, "") for i, t in enumerate(ids)]
    if len(ids) >= m.num_ctx:
        raise ValueError("the prompt is %d tokens; the context limit (num_ctx) is %d" % (len(ids), m.num_ctx))
    num_predict = int(opts.get("num_predict", 1024) or 1024)
    if num_predict < 0:
        num_predict = m.num_ctx
    num_predict = max(1, min(num_predict, m.num_ctx - len(ids)))
    want = 1 if float(opts.get("temperature", 0.0) or 0.0) <= 0 else max(1, min(int(opts.get("top_k", MAX_RANK) or MAX_RANK), MAX_RANK))
    seed = opts.get("seed")
    rng = np.random.default_rng(None if seed in (None, -1) else int(seed))

    # Image tokens carry (t, h, w) positions; everything after an image is shifted accordingly.
    rope = req.get("rope")
    if rope:
        tail = [[i - rope["shift"]] * 3 for i in range(len(ids), m.num_ctx)]
        m.set_rope(rope["i0"], list(rope["pos"]) + tail)
        m.rope_from = rope["i0"] if m.rope_from is None else min(m.rope_from, rope["i0"])
    elif m.rope_from is not None:
        m.set_rope(m.rope_from, [[i] * 3 for i in range(m.rope_from, m.num_ctx)])
        m.rope_from = None

    # Reuse what the cards already hold when this prompt continues the previous exchange.
    reuse = 0
    if 0 < len(m.fed) < len(keys) and m.fed == keys[:len(m.fed)]:
        reuse = len(m.fed)
    else:
        m.command(SEG_RESET, -1)
        m.fed = []
    emit(("start", {"prompt_tokens": len(ids), "reused_tokens": reuse, "num_predict": num_predict, "num_ctx": m.num_ctx}))
    aptop = getattr(m, "aptop", None)
    if aptop:
        aptop.phase("prefill")

    def vec_at(i, token):
        return np.frombuffer(vecs[i], dtype=np.int16) if i in vecs else m.embed[token]

    t0 = time.perf_counter()
    cands = None
    for i in range(reuse, len(ids)):
        if abort.is_set():
            break
        lastp = i == len(ids) - 1
        cands = m.forward(vec_at(i, ids[i]), i, want if lastp else 0, None)
        m.fed.append(keys[i])
        if (i - reuse) % 8 == 7 and not lastp:
            emit(("prefill", {"done": i + 1 - reuse, "total": len(ids) - reuse}))
    t_prefill = time.perf_counter() - t0
    n_prefill = len(m.fed) - reuse

    acc = {"hw": np.zeros((m.P, m.n_seg)), "wall": np.zeros((m.P, m.n_seg)), "seg_wall": np.zeros(m.n_seg)}
    out, reason = [], "stop"
    t1 = time.perf_counter()
    ttft = None
    pos = len(ids)
    if abort.is_set() or cands is None:
        reason = "abort"
    else:
        while True:
            tok, logit = pick(cands, opts, rng)
            out.append(tok)
            if ttft is None:
                ttft = time.perf_counter() - t0
                if aptop:
                    aptop.phase("decode")
            emit(("token", tok, logit))
            if tok in stop_ids:
                reason = "stop"
                break
            if len(out) >= num_predict:
                reason = "length"
                break
            if abort.is_set():
                reason = "abort"
                break
            cands = m.forward(m.embed[tok], pos, want, acc)
            m.fed.append((tok, ""))
            pos += 1
    t_dec = time.perf_counter() - t1
    n_steps = max(len(out) - 1, 0)        # decode forwards measured (the first token came from the prefill)
    stats = {"prompt_tokens": len(ids), "reused_tokens": reuse, "prefill_tokens": n_prefill, "prefill_s": t_prefill,
             "prefill_tok_s": (n_prefill / t_prefill) if t_prefill > 0 and n_prefill else 0.0,
             "eval_tokens": len(out), "eval_s": t_dec, "decode_steps": n_steps,
             "eval_tok_s": (n_steps / t_dec) if t_dec > 0 and n_steps else 0.0,
             "ms_per_token": (1e3 * t_dec / n_steps) if n_steps else 0.0, "ttft_s": ttft or 0.0,
             "total_s": time.perf_counter() - t0, "done_reason": reason, "context_used": len(m.fed) + (1 if out else 0),
             "num_ctx": m.num_ctx, "sampling": {"temperature": float(opts.get("temperature", 0.0) or 0.0),
                                                "top_k": want, "top_p": float(opts.get("top_p", 1.0) or 1.0),
                                                "seed": seed, "candidates_per_card": want},
             "segments_per_token": m.n_seg}
    if n_steps:
        stats.update(m.seg_report(m, acc, n_steps) if hasattr(m, "seg_report") else seg_report(m, acc, n_steps))
    m.command(SEG_PARK, -1)
    if aptop:
        aptop.phase("idle")
    emit(("done", stats))


def driver_main(conn, abort):
    """Driver process: owns the loaded model. Commands in, events out, over one pipe."""
    os.sched_setaffinity(0, {cpu_plan(1)[0]})
    ctypes.CDLL(None).prctl(29, 1, 0, 0, 0)
    torch.set_num_threads(1)
    aptop = AptopState()
    m = None

    def emit(ev):
        conn.send(ev)

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        try:
            if msg[0] == "load":
                name, num_ctx = msg[1], int(msg[2])
                if m is not None:
                    m.close()
                    m = None
                aptop.loading(name)
                if is_gptoss(name):
                    import ollapex_gptoss
                    m = ollapex_gptoss.LoadedGptOss(name, num_ctx, emit, model_paths(name), MODELS[name][2],
                                                    MODEL_MAX_CTX.get(name, MAX_CTX))
                else:
                    m = Loaded(name, num_ctx, emit)
                m.aptop = aptop
                aptop.loaded(m)
                emit(("ready", m.info()))
            elif msg[0] == "generate":
                if m is None:
                    raise RuntimeError("no model loaded")
                abort.clear()
                generate(m, msg[1], emit, abort)
            elif msg[0] == "vision":
                if m is None:
                    raise RuntimeError("no model loaded")
                toks, meta = m.vision(msg[1], msg[2], msg[3])
                emit(("vision_done", toks.tobytes(), list(toks.shape), meta))
            elif msg[0] == "unload":
                if m is not None:
                    m.close()
                    m = None
                aptop.unloaded()
                emit(("unloaded",))
            elif msg[0] == "quit":
                break
        except Exception as e:
            if isinstance(e, ValueError):
                emit(("error", str(e), False))
            else:
                # the cards' state is unknown after a failure: drop the model so the next request reloads it
                detail = traceback.format_exc()[-1800:]
                try:
                    if m is not None:
                        m.close()
                except Exception:
                    pass
                m = None
                aptop.unloaded()
                emit(("error", "%s\n%s" % (e, detail), True))
    if m is not None:
        m.close()
    aptop.unloaded()
