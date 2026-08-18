"""LIBERO closed-loop evaluation for VeraPulse PulseVLA-LIBERO-0.5B -- SINGLE PROCESS.

Ported from models/pi05_libero/libero_eval.py, which is the proven harness for the same
class of model (VLA: vision + prefix + flow-matching action expert + action chunking) on
the same benchmark. Structure, cadence, checkpointing and failure handling are that
file's; only the model plumbing differs.

    reset -> settle -> observe -> infer a chunk -> execute --replan-steps of it -> re-plan

*** THE POLICY IS NOT VALIDATED YET. ***
As of this writing the action expert scores ~17 dB below its bf16 floor and the executed
actions sit at cos 0.97 against the CPU oracle. A success rate produced by this harness
today measures A MODEL WITH A KNOWN BUG, not the accelerator and not PulseVLA. The
harness exists so that the number can be taken the moment the expert bug is fixed. Every
run prints this warning and stamps `policy_validated: false` into the results JSON.

Run:
    # plumbing only -- no FPGA, no LIBERO, no simulator. Safe anywhere.
    python models/verapulse/libero_eval.py --dry-run --tasks 2 --trials 1 --max-steps 40

    # real: FPGA + LIBERO. MUJOCO_GL=egl is required for headless rendering.
    MUJOCO_GL=egl python models/verapulse/libero_eval.py \
        --task-suite libero_spatial --tasks 1 --trials 1 --max-steps 60   # smoke test
    MUJOCO_GL=egl python models/verapulse/libero_eval.py --trials 2       # 10x2 episodes

    # after a reboot (see below), resume where it stopped:
    MUJOCO_GL=egl python models/verapulse/libero_eval.py --trials 2 --resume

THREE CONSTRAINTS THIS FILE IS BUILT AROUND, all learned the expensive way:

1. COMPILE ONCE, EXECUTE MANY. The engine is constructed and weight_init/tensor_init are
   run EXACTLY ONCE for the whole eval; the three device programs are all compiled up
   front by precompile_all() during backend construction, and every inference -- the
   first included -- only re-EXECUTES them.
   Recompiling advances the program-DRAM allocator every inference and marches it at the
   4 GB ceiling: pi05 died after 3 inferences that way and survived 252 once fixed. Any
   change here that re-enters weight_init or constructs a second engine per episode
   silently reintroduces that bug.

2. NEVER CONSTRUCT A SECOND UnifiedEngine after weight_init. Every ctor DMA-writes ~16 KB
   of noise to a hardcoded 0x80000000, and this model's params_dram_base is 0x00000000 --
   i.e. exactly there. The second engine would shred the first weights stored. There is
   one engine in this process, built once, at the top.

3. THE BOX HARD-REBOOTS under sustained FPGA load, with no kernel log. So results are
   checkpointed to --results-out AFTER EVERY EPISODE, and --resume reads that file back
   and skips the (task_id, trial) pairs already recorded. A reboot costs one episode, not
   a night. The resume command is printed after every checkpoint.

Everything is single-process on purpose (pi05's rule): the FPGA driver and MuJoCo are not
forked alongside each other.
"""
import argparse
import collections
import gc
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))          # verapulse_test
sys.path.insert(0, str(_HERE.parent.parent))   # repo root (user_dma_core, engine libs)

# Headless rendering. MuJoCo picks its GL backend at IMPORT time, so this must be set
# before anything imports mujoco/robosuite -- setting it in the shell is the documented
# way (upstream's requirements.txt says so), and this is the belt-and-braces default for
# when someone forgets.
os.environ.setdefault("MUJOCO_GL", "egl")

LIBERO_ENV_RESOLUTION = 256      # what the sim renders
MODEL_IMAGE_SIZE = 512           # what the ViT consumes (config vision.image_size)
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Same per-suite step budgets pi05 uses; they come from the LIBERO paper's episode
# lengths, not from anything model-specific, so they carry over unchanged.
_SUITE_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_90": 400,
}

_POLICY_WARNING = (
    "  !! FPGA BACKEND ONLY: the action expert on the DEVICE is missing three known\n"
    "     model faults (prefix-K/V concat on even layers, (320,320) cross reprojection,\n"
    "     causal suffix mask). They are fixed in VeraPulseRef and NOT in the emitter, so\n"
    "     the device reads cos 0.64 vs upstream. --backend cpu and --backend oracle both\n"
    "     reproduce upstream exactly (114.9 dB / 1/1 on libero_object) and are sound.")


# ---------------------------------------------------------------------------
# observation -> model inputs, and model output -> robot action
# ---------------------------------------------------------------------------
def _resize_with_pad_UNUSED(img_u8, size):
    """DEAD -- kept only as a record of what NOT to do.

    This resamples with PIL bilinear on uint8 and centre-pads. Upstream resamples with
    F.interpolate bilinear on float [0,1] and pads left/top. Those are different images,
    not different paddings, and at LIBERO's 256->512 the resample is real. The live path
    calls upstream's own resize_with_pad from _VeraPulsePre. Delete once nothing in the
    tree references it.
    """
    """(H,W,3) uint8 -> (size,size,3) uint8, ASPECT PRESERVED with zero padding.

    LIBERO renders 256x256 (square) so the padding is inert today, but a
    plain stretch would silently distort the moment a non-square camera appears, and
    the training pipeline pads. Implemented locally with PIL rather than pulling in
    openpi's image_tools: this model has no openpi dependency and should not grow one.
    """
    from PIL import Image
    h, w = img_u8.shape[:2]
    if (h, w) == (size, size):
        return np.ascontiguousarray(img_u8)
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = np.asarray(Image.fromarray(img_u8).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    t, l = (size - nh) // 2, (size - nw) // 2
    out[t:t + nh, l:l + nw] = small
    return out


def _quat2axisangle(quat):
    """robosuite's eef quaternion -> axis-angle, the form the state vector uses."""
    q = np.array(quat, dtype=np.float32)
    q[3] = min(1.0, max(-1.0, q[3]))
    den = math.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


class _VeraPulsePre:
    """Raw LIBERO obs -> (images[2,512,512,3], token ids[48], mask[48], state[32]),
    and the model's normalized [10,7] chunk -> robot action space.

    Uses verapulse_test.tokenize() / load_norm_stats(), which is where those two stubs
    were implemented -- one source of truth shared with the _test.py entry point, so the
    eval and the bring-up runs can never tokenize or de-normalize differently.
    """

    def __init__(self, image_norm="pm1"):
        import verapulse_test as M
        self.M = M
        cfg = M._CFG
        # DELEGATE to the checkpoint's own preprocessing rather than reimplementing it.
        # Matching upstream by inspection is how we ended up with PIL-bilinear-on-uint8
        # standing in for F.interpolate-bilinear-on-float (a real resample difference at
        # 256->512, not the inert padding difference it was first written down as), plus
        # a lowercase() upstream does not do and a different normalization epsilon.
        # Importing their functions makes those classes of bug impossible.
        _b = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "verapulse_bin", "verapulse__pulsevla-libero-0.5b")
        if _b not in sys.path:
            sys.path.insert(0, _b)
        from smolvla.images import resize_with_pad, to_siglip_range
        from smolvla.normalize import normalize_meanstd, pad_to
        self._resize_with_pad = resize_with_pad
        self._to_siglip_range = to_siglip_range
        self._normalize_meanstd = normalize_meanstd
        self._pad_to = pad_to
        self.text_len = cfg["lm"]["tokenizer_max_length"]         # 48
        self.state_dim = cfg["action_head"]["state_dim"]          # 32 (padded)
        self.action_dim = cfg["action_head"]["action_dim"]        # 7
        self.n_exec = cfg["action_head"]["n_action_steps"]        # 10
        self.slots = cfg["vision"]["num_image_slots"]             # 2
        ns = M.load_norm_stats()
        self.s_mean, self.s_std = (ns["state"][0].numpy(), ns["state"][1].numpy())
        self.a_mean, self.a_std = (ns["action"][0].numpy(), ns["action"][1].numpy())
        self.image_norm = image_norm

    # -- images -------------------------------------------------------------------
    def images(self, base_u8, wrist_u8):
        """RAW (H,W,3) uint8 x2 -> [2,512,512,3] float32 in SigLIP range, via UPSTREAM.

        Slot 0 = agentview ('image'), slot 1 = wrist ('image2') -- the key order
        upstream's predict_example.py uses, and the order the oracle scored 1/1 with.

        Takes the images UNRESIZED: upstream's resize_with_pad is the resampler, so the
        caller must NOT pre-resize (doing so would apply a second, different resample).
        Their pipeline is exactly: [0,1] -> resize_with_pad(512,512) -> *2-1.

        Returns HWC because that is what the device's patchify consumes; upstream works
        in CHW, so the permute happens here and nowhere else.
        """
        import torch as _t
        out = []
        for a in (base_u8, wrist_u8):
            x = _t.as_tensor(np.ascontiguousarray(a), dtype=_t.float32)
            x = (x.permute(2, 0, 1) / 255.0).unsqueeze(0)          # (1,3,H,W) in [0,1]
            x = self._resize_with_pad(x, 512, 512, pad_value=0.0)
            if self.image_norm == "pm1":
                x = self._to_siglip_range(x)
            out.append(x[0].permute(1, 2, 0))                      # -> (512,512,3)
        return _t.stack(out).numpy().astype(np.float32)

    # -- language -----------------------------------------------------------------
    def tokens(self, language):
        """(ids[48], mask[48]). The mask is the whole point: the prefix's valid length is
        1 + 128 + n_real_text, NOT the constant 177 in the config. Feeding 177 would let
        every pad slot of a short instruction attend as if it were language."""
        return self.M.tokenize(language, self.text_len, return_mask=True)

    # -- state --------------------------------------------------------------------
    def state(self, state_raw):
        """robot state (8: eef pos 3 + axis-angle 3 + gripper 2) -> normalized, zero-padded
        to max_state_dim=32.

        Normalization is (x - mean) / std from norm_stats -- the lerobot family transform,
        NOT pi05's q01/q99 quantile scaling. Padding is ZERO, and that is safe here (unlike
        the 1e-6 the prefix/x_t rows need) because this vector goes through state_proj, a
        plain matmul whose K-padded columns are zero weights: zeros in, zeros contributed.
        The epsilon-free RMSNorm never sees this vector directly."""
        import torch as _t
        s = np.asarray(state_raw, dtype=np.float32).reshape(-1)
        n = self.s_mean.shape[0]
        assert s.shape[0] == n, (
            f"norm_stats expects a {n}-dim robot state, got {s.shape[0]} -- do not "
            f"silently truncate; the mean/std would apply to the wrong joints")
        # upstream's normalize_meanstd: (x-mean)/(std+1e-8). NOT max(std,1e-6) -- those
        # differ wherever a joint's std is small, which is exactly the gripper.
        x = self._normalize_meanstd(_t.as_tensor(s), _t.as_tensor(self.s_mean),
                                    _t.as_tensor(self.s_std))
        return self._pad_to(x, self.state_dim).numpy().astype(np.float32)

    # -- output -------------------------------------------------------------------
    def unnormalize(self, actions):
        """[n,7] normalized -> robot action space: a * std + mean (inverse of the
        training-time (a - mean)/std). A wrong inverse here does not crash -- it makes a
        correct policy look broken -- so it lives beside load_norm_stats() and nowhere
        else."""
        a = np.asarray(actions, dtype=np.float32)
        return a * self.a_std[None, : a.shape[1]] + self.a_mean[None, : a.shape[1]]


# ---------------------------------------------------------------------------
# backends: model inputs -> (10,7) NORMALIZED action chunk
# ---------------------------------------------------------------------------
class _FpgaBackend:
    """The UnifiedEngine on real hardware. ONE engine, built ONCE (see constraints 1+2
    in the module docstring)."""

    def __init__(self, weights="real", seed=0, use_run_inference=False, snr=False,
                 strict_gates=False, fused_silu=True, engines=1, vis_engines=None):
        import verapulse_test as M
        self.M = M
        self.use_run_inference = use_run_inference
        self.snr = snr
        self.strict_gates = strict_gates
        print("[eval] building FPGA engine (ctor + weight_init + tensor_init)...",
              flush=True)
        t0 = time.perf_counter()
        # THE ONLY UnifiedEngine construction in this process. Its ctor DMA-writes 16 KB
        # of noise to 0x80000000 == this model's params base, so a second one built later
        # would destroy the weights loaded below.
        self.ue = M.VeraPulse_UnifiedEngine()
        # Engine counts BEFORE weight_init: weight_init's first act is to build the
        # worker pool from them, and no UnifiedEngine may be constructed after the
        # weights are in DRAM (its ctor DMA-writes 16 KB of noise at a hardcoded
        # address). These are class attributes, so without this call every episode would
        # silently run single-engine no matter what the CLI said.
        M.configure_engines(self.ue, engines, vis=vis_engines, tag="eval")
        self.ue.PREFIX_FUSED_SILU = fused_silu
        self.ue.weight_init(dummy=(weights == "dummy"), seed=seed)
        self.ue.tensor_init()
        print(f"[eval] engine ready in {time.perf_counter() - t0:.1f}s", flush=True)
        # COMPILE PHASE, once, before any observation exists. All three programs
        # (encoder, prefix, denoise) are built here, so inference #0 is not structurally
        # different from inference #251: every rollout step is pure execute. The programs
        # are address-static and the per-observation quantities (attention biases, expert
        # RoPE base) are DRAM data restaged by run_prefix/run_denoise, so nothing about
        # this episode is baked into the bytes. precompile_all also freezes the program
        # set: if any stage tried to compile mid-episode it now raises instead of
        # stalling the control loop and marching the program-DRAM pointer.
        print("[eval] precompiling encoder + prefix + denoise (one-time)...", flush=True)
        t1 = time.perf_counter()
        self.ue.precompile_all()
        print(f"[eval] programs ready in {time.perf_counter() - t1:.1f}s -- "
              f"inferences are execute-only", flush=True)
        self._first = True

    def infer(self, images, ids, mask, state32, noise=None):
        ue = self.ue
        if self._first:
            print("[eval] first inference: execute-only (all programs were compiled "
                  "up front by precompile_all).", flush=True)
        if self.use_run_inference:
            # Single entry point, but it cannot carry the text mask (run_inference does
            # not take one and is off-limits to edit), so the prefix falls back to
            # counting ALL 48 text slots as valid. Conservative -- it never masks a real
            # token -- but padded text then attends. Kept as an opt-in so the eval can be
            # cross-checked against exactly what verapulse_test.py --stage all runs.
            out = ue.run_inference(images, ids, state32, noise=noise, snr=self.snr,
                                   strict_gates=self.strict_gates)
            self._first = False
            return np.asarray(out, dtype=np.float32)
        # DEFAULT: the same three stages run_inference runs, in the same order, with the
        # real text mask plumbed into run_prefix -> build_attn_bias. The per-stage SNR
        # gates are deliberately NOT run per inference: each one builds/uses the torch
        # oracle from a 2.23 GB checkpoint, which is minutes of CPU per step. Gate with
        # `verapulse_test.py --stage all`, then roll out here.
        vision_tokens = ue.run_vision(images)
        ue.run_prefix(vision_tokens, ids, state32, text_mask=mask)
        chunk = ue.run_denoise(noise=noise)          # [50, 7] normalized
        self._first = False
        n_exec = self.M._CFG["action_head"]["n_action_steps"]
        return np.asarray(chunk[:n_exec], dtype=np.float32)


class _CpuBackend:
    """THE CONTROL. Runs the pure-torch oracle (VeraPulseRef) in the closed loop, with
    no FPGA involved at all.

    This is the experiment that separates two completely different failure worlds:
      * CPU SUCCEEDS, FPGA fails  -> the accelerator's numerics are the problem.
      * CPU ALSO FAILS            -> the accelerator is exonerated and the fault is in
                                     the MODEL ASSEMBLY or preprocessing -- and note the
                                     assembly is still built on unverified OpenChoices
                                     (prefix_order, the RoPE position base, the
                                     cross-attn layer mapping). Each stage is verified
                                     in isolation (vision 112 dB vs transformers'
                                     SmolVLMVisionTransformer, prefix 122.8 dB vs
                                     LlamaModel, connector bit-exact) but the way they
                                     are wired together is not.

    Cheap enough to be the default first move: ~1 s per inference on CPU versus ~19 s
    per inference on hardware, so a full episode is about a minute instead of nine.
    """

    def __init__(self, weights="real", seed=0, tiny=False, hw_gelu=False):
        """hw_gelu=False (DEFAULT) = the model as published; the backend answers
        'is our implementation correct?'. hw_gelu=True substitutes the accelerator's
        quick_gelu, so the backend instead PREDICTS THE HARDWARE.

        These are different questions and the default used to be the second one, which
        made the CPU control unable to tell us whether our model was right -- it carried
        an 8.4 dB activation substitution by construction. Measured: hw_gelu=True reads
        cos 0.958 vs upstream on the action chunk; hw_gelu=False reads ~119 dB."""
        import torch
        import verapulse_test as M
        self.M, self.torch = M, torch
        print(f"[eval] building CPU oracle ({weights} weights, "
              f"{'quick_gelu (hardware-mimic)' if hw_gelu else 'exact gelu'})"
              f"{' [tiny]' if tiny else ''}...", flush=True)
        t0 = time.perf_counter()
        self.ref = (M.VeraPulseRef.from_checkpoint(hw_gelu=hw_gelu) if weights == "real"
                    else M.VeraPulseRef.from_fake(seed=seed, hw_gelu=hw_gelu))
        if tiny:
            self.ref.n_vis = self.ref.n_lm = self.ref.n_ae = 2
        cfg = M._CFG
        self.V, self.C = cfg["vision"], cfg["connector"]
        self.n_exec = cfg["action_head"]["n_action_steps"]
        self.adim = cfg["action_head"]["action_dim"]
        self.rng = np.random.RandomState(seed)
        print(f"[eval] oracle ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def infer(self, images, ids, mask, state32, noise=None):
        torch, M = self.torch, self.M
        V = self.V
        P, NPS, CH = V["patch_size"], V["num_patches_per_side"], V["num_channels"]
        imgs = torch.as_tensor(np.asarray(images), dtype=torch.float32)
        ids_t = torch.as_tensor(np.asarray(ids), dtype=torch.long)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        st = torch.as_tensor(np.asarray(state32), dtype=torch.float32)
        with torch.no_grad():
            vis = []
            for s in range(imgs.shape[0]):
                planes = imgs[s].permute(2, 0, 1).contiguous()
                patches = planes.reshape(CH, NPS, P, NPS, P).permute(1, 3, 0, 2, 4) \
                                .reshape(V["num_patches"], CH * P * P)
                vis.append(self.ref.forward_vision(patches))
            toks = torch.cat([self.ref.forward_connector(v) for v in vis], 0)
            # Only the REAL text tokens, mirroring what run_prefix does with the mask.
            x, valid, pos = self.ref.build_prefix(toks, ids_t[mask_t], st)
            _, kv = self.ref.forward_prefix(x, pos)
            acts = self.ref.denoise(kv, x.shape[0], noise=noise)
        return np.asarray(acts[:self.n_exec, :self.adim], dtype=np.float32)


class _OracleBackend:
    """UPSTREAM'S OWN reference implementation, driven by our env loop.

    This is the ground truth, not another reconstruction: the `smolvla/` package that
    ships with the checkpoint, loaded with a STRICT state_dict (it would throw on any
    architecture mismatch -- which is itself the proof that the expert's odd-layer
    k_proj really is (320,320) and reprojects the cached VLM K/V).

    It deliberately bypasses _VeraPulsePre and consumes RAW observations, so their
    processor does the resize/pad, the [-1,1] map, the tokenization and the state
    normalization. That takes OUR preprocessing out of the comparison too -- which
    matters, because that is exactly where the lowercase-tokenizer and centre-pad
    differences live. `takes_raw_obs` is the flag the infer() closure switches on.

    Needs only torch + safetensors + tokenizers. NOT lerobot and NOT libero: upstream's
    own eval_libero.py wants those, but we drive the simulator ourselves, so the model
    is the only thing we borrow. ~1-2 s per inference on CPU.
    """

    takes_raw_obs = True

    def __init__(self, bundle=None, device="cpu", n_exec=10, seed=0):
        bundle = bundle or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "verapulse_bin", "verapulse__pulsevla-libero-0.5b")
        if not os.path.isdir(bundle):
            raise FileNotFoundError(
                f"upstream bundle not found at {bundle} -- it ships with the checkpoint; "
                f"run the downloader (it must NOT filter the .py files out)")
        # Prepend so `import smolvla` resolves to the bundle, never to a stray install.
        sys.path.insert(0, bundle)
        from safetensors.torch import load_file
        from smolvla import (SmolVLA, SmolVLAProcessor, Tokenizer,
                             load_lerobot_norm_stats, load_smolvla_config)
        from smolvla.types import Obs

        self.Obs, self.device, self.n_exec = Obs, device, n_exec
        print(f"[eval] building UPSTREAM oracle from {bundle} on {device}...", flush=True)
        t0 = time.perf_counter()
        cfg = load_smolvla_config(os.path.join(bundle, "config.json"),
                                  n_action_steps=n_exec)
        self.model = SmolVLA(cfg).float().to(device).eval()
        self.model.load_state_dict(load_file(os.path.join(bundle, "model.safetensors")))
        stats = load_lerobot_norm_stats(os.path.join(bundle, "norm_stats.safetensors"))
        tok = Tokenizer(os.path.join(bundle, "tokenizer.json"),
                        max_length=cfg.tokenizer_max_length)
        self.proc = SmolVLAProcessor(cfg, tok, stats, device=device)
        torch.manual_seed(seed)          # the policy samples flow-matching noise
        n_par = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[eval] oracle ready in {time.perf_counter() - t0:.1f}s "
              f"({n_par:.1f}M params)", flush=True)

    def infer_raw(self, base_u8, wrist_u8, state_raw, language):
        """RAW obs -> (n_exec, 7) in ROBOT UNITS.

        Returns DE-NORMALIZED actions, unlike every other backend here, because their
        postprocess_action already applies a*std+mean. The caller must therefore skip
        pre.unnormalize -- denormalizing twice does not crash, it just makes a correct
        policy look broken, which is the exact failure mode we are trying to rule out.

        Images arrive already 180-flipped and resized to 512 by the loop; their
        resize_with_pad on a 512 square is a no-op, so there is no second resample.
        """
        def _img(a):
            t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
            return (t.permute(2, 0, 1) / 255.0).unsqueeze(0).to(self.device)

        obs = self.Obs(
            images={"image": _img(base_u8), "image2": _img(wrist_u8)},
            state=torch.as_tensor(
                np.asarray(state_raw, dtype=np.float32))[None].to(self.device),
            task=[str(language)],
        )
        model_input = self.proc.to_model_input(obs)
        with torch.inference_mode():
            chunk = self.model.predict_action_chunk(model_input)
        acts = self.proc.postprocess_action(chunk[:, : self.n_exec])
        return np.asarray(acts[0], dtype=np.float32)


class _FakeBackend:
    """--dry-run: exercises the WHOLE harness with no FPGA and no engine construction.

    Deliberately not a mock of the engine API -- it is a stand-in for the one thing the
    harness consumes, a (10,7) normalized chunk. Its actions are small and smooth so a
    dry run steps the loop plausibly rather than flailing; it is plumbing, not a policy."""

    def __init__(self, n_exec=10, action_dim=7, seed=0):
        self.n_exec, self.action_dim = n_exec, action_dim
        self.rng = np.random.RandomState(seed)
        self.calls = 0

    def infer(self, images, ids, mask, state32, noise=None):
        self.calls += 1
        assert images.shape[0] == 2, "fake backend still checks the 2-camera contract"
        assert ids.shape[-1] == mask.shape[-1], "ids/mask length mismatch"
        return (self.rng.randn(self.n_exec, self.action_dim) * 0.05).astype(np.float32)


# ---------------------------------------------------------------------------
# fake environment for --dry-run
# ---------------------------------------------------------------------------
class _FakeTask:
    def __init__(self, tid):
        self.language = f"fake task {tid}: pick up the black bowl and place it on the plate"
        self.problem_folder, self.bddl_file = "fake", f"task{tid}.bddl"


class _FakeEnv:
    """Minimal stand-in for OffScreenRenderEnv: same obs keys, same step/reset contract,
    and it declares success on a fixed step so the bookkeeping/checkpoint/resume paths are
    all exercised (successes AND failures) with zero hardware and zero simulator."""

    def __init__(self, seed=0, succeed_after=None):
        self.rng = np.random.RandomState(seed)
        self.succeed_after = succeed_after
        self.t = 0

    def _obs(self):
        r = LIBERO_ENV_RESOLUTION
        return {
            "agentview_image": self.rng.randint(0, 256, (r, r, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": self.rng.randint(0, 256, (r, r, 3), dtype=np.uint8),
            "robot0_eef_pos": self.rng.randn(3).astype(np.float32) * 0.1,
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": self.rng.randn(2).astype(np.float32) * 0.01,
        }

    def seed(self, s):
        self.rng = np.random.RandomState(s)

    def reset(self):
        self.t = 0
        return self._obs()

    def set_init_state(self, st):
        return self._obs()

    def step(self, action):
        assert len(action) == 7, f"env expects a 7-dof action, got {len(action)}"
        assert np.all(np.isfinite(action)), "non-finite action reached the env"
        self.t += 1
        done = self.succeed_after is not None and self.t >= self.succeed_after
        return self._obs(), 0.0, bool(done), {}

    def close(self):
        pass


# ---------------------------------------------------------------------------
# results file: checkpoint + resume
# ---------------------------------------------------------------------------
def _load_previous(path):
    """Read a results JSON back for --resume. Returns (episodes, done_keys).

    Tolerant on purpose: a hard reboot can truncate the file mid-write, and losing the
    whole history to a JSONDecodeError would defeat the point of checkpointing. A
    corrupt file is reported and treated as empty."""
    p = pathlib.Path(path)
    if not p.exists():
        return [], set()
    try:
        blob = json.loads(p.read_text())
    except Exception as e:
        print(f"[eval] --resume: {p} is unreadable ({e!r}); starting fresh "
              f"(the old file is NOT overwritten until the first new episode).")
        return [], set()
    eps = blob.get("episodes", [])
    return eps, {(int(e["task_id"]), int(e["trial"])) for e in eps}


def compare_trace(path):
    """CPU-ONLY replay: recompute every recorded inference with the torch oracle and
    report per-inference divergence.

    The episode summary tells you the policy failed; it cannot tell you WHETHER the
    hardware drifted uniformly or blew up at one observation. This does -- same idea as
    the per-layer KV curve and the per-Euler-step curve in verapulse_test.py, applied
    across the rollout.

    The oracle is fed the trace's PREPROCESSED images, so preprocessing differences
    cannot masquerade as model error. Touches no hardware.
    """
    import verapulse_test as M
    tr = np.load(path)
    n = tr["images"].shape[0]
    print(f"[compare] {n} inferences from {path}")
    print("[compare] loading the real checkpoint into the torch oracle "
          "(2.23 GB, CPU -- this is the slow part)...", flush=True)
    ref = M.VeraPulseRef.from_checkpoint(hw_gelu=True)
    cfg = M._CFG
    V, C, HEAD = cfg["vision"], cfg["connector"], cfg["action_head"]
    n_exec, adim = HEAD["n_action_steps"], HEAD["action_dim"]
    P, NPS, CH = V["patch_size"], V["num_patches_per_side"], V["num_channels"]

    print(f"\n{'inf':>4} {'cos':>10} {'snr(dB)':>9} {'rms hw/ref':>18}  {'|dev|max':>9}")
    rows = []
    for i in range(n):
        imgs = torch.as_tensor(tr["images"][i])            # [2,512,512,3] preprocessed
        ids = torch.as_tensor(tr["ids"][i])
        mask = torch.as_tensor(tr["mask"][i])
        state = torch.as_tensor(tr["state"][i])
        with torch.no_grad():
            vis = []
            for s in range(imgs.shape[0]):
                planes = imgs[s].permute(2, 0, 1).contiguous().float()
                patches = planes.reshape(CH, NPS, P, NPS, P).permute(1, 3, 0, 2, 4) \
                                .reshape(V["num_patches"], CH * P * P)
                vis.append(ref.forward_vision(patches))
            toks = torch.cat([ref.forward_connector(v) for v in vis], 0)
            x, valid, pos = ref.build_prefix(toks, ids[mask], state)
            _, kv = ref.forward_prefix(x, pos)
            acts = ref.denoise(kv, int(mask.sum()) + toks.shape[0] + 1, noise=None)
        r = acts[:n_exec, :adim]
        h = torch.as_tensor(tr["hw"][i])[:n_exec, :adim]
        m = torch.ones(h.shape[0], dtype=torch.bool)
        c, s_, dev = M.cos_sim(h, r, m), M.snr_db(h, r, m), float((h - r).abs().max())
        rows.append((i, c, s_, dev))
        print(f"{i:>4} {c:>10.6f} {s_:>9.2f} {M.rms(h, m):>8.4f}/{M.rms(r, m):<8.4f} {dev:>9.4f}")
    worst = min(rows, key=lambda t: t[1])
    print(f"\n  worst inference: #{worst[0]}  cos={worst[1]:.6f}  snr={worst[2]:.2f} dB")
    print(f"  mean cos: {sum(r[1] for r in rows) / len(rows):.6f}")
    print("  UNIFORM cos across inferences => a constant numerical gap;")
    print("  ONE bad inference => that observation drove the arm off-policy.")
    print("  NOTE: the oracle redraws its own noise, so some spread is expected;")
    print("  judge the SHAPE across inferences, not the absolute level.")


def main():
    ap = argparse.ArgumentParser(
        description="Closed-loop LIBERO evaluation for VeraPulse on the FPGA.")
    ap.add_argument("--task-suite", default="libero_spatial", choices=list(_SUITE_MAX_STEPS))
    ap.add_argument("--tasks", type=int, default=None,
                    help="cap #tasks (default: all 10 in the suite)")
    ap.add_argument("--task-start", type=int, default=0,
                    help="first task id. Lets a suite be split across processes.")
    ap.add_argument("--trials", type=int, default=2,
                    help="rollouts per task (2 -> 20 episodes, directional; 5 for a "
                         "citable number)")
    ap.add_argument("--max-steps", type=int, default=None, help="override the suite default")
    ap.add_argument("--replan-steps", type=int, default=10,
                    help="how many actions of each 10-step chunk to execute before "
                         "re-querying the model (default 10 = the whole chunk, i.e. the "
                         "fewest inferences; lower = tighter closed loop, more FPGA time)")
    ap.add_argument("--wait-steps", type=int, default=10,
                    help="dummy-action steps after reset so the objects settle")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--results-out", default=None,
                    help="per-episode results JSON (default: data/libero/"
                         "results_verapulse_<suite>.json). Written after EVERY episode.")
    ap.add_argument("--resume", action="store_true",
                    help="read --results-out back and SKIP the (task, trial) pairs it "
                         "already contains. This is the reboot recovery path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the whole harness with a FAKE env and FAKE backend: "
                         "no FPGA, no engine construction, no LIBERO import, no MuJoCo. "
                         "This is the plumbing test.")
    ap.add_argument("--dry-succeed-every", type=int, default=3,
                    help="--dry-run only: make every Nth episode 'succeed' so the "
                         "success/failure bookkeeping is exercised in both directions")
    ap.add_argument("--backend", default="fpga", choices=["fpga", "cpu", "oracle"],
                    help="fpga = the accelerator; cpu = OUR torch reconstruction in the "
                         "loop; oracle = UPSTREAM'S OWN implementation from the shipped "
                         "bundle. cpu is the control that exonerates the accelerator; "
                         "oracle is the control that tells you whether the harness and "
                         "the env are sound, because it is known-good by construction.")
    ap.add_argument("--cpu-hw-gelu", action="store_true",
                    help="--backend cpu only: build the reference with the "
                         "accelerator's quick_gelu so the CPU run PREDICTS THE HARDWARE "
                         "instead of testing the model. Default off: the CPU backend "
                         "should answer 'is our implementation right?', and carrying an "
                         "8.4 dB activation substitution by default made it unable to.")
    ap.add_argument("--oracle-device", default="cpu",
                    help="--backend oracle only: torch device for the reference model. "
                         "cpu is ~1-2 s/inference and touches no FPGA.")
    ap.add_argument("--weights", default="real", choices=["real", "dummy"],
                    help="which weights weight_init stores. dummy = shape-exact synthetic "
                         "(plumbing on hardware without the 2.23 GB checkpoint); it is NOT "
                         "a fidelity claim and the rollout means nothing.")
    ap.add_argument("--use-run-inference", action="store_true",
                    help="drive the model through run_inference() instead of the "
                         "vision/prefix/denoise sequence. Matches verapulse_test.py "
                         "--stage all exactly, but CANNOT pass the text mask, so the "
                         "prefix counts all 48 text slots as valid.")
    ap.add_argument("--snr", action="store_true",
                    help="--use-run-inference only: run the per-stage SNR gates on every "
                         "inference. Each gate runs the 2.23 GB torch oracle on CPU -- "
                         "minutes per step. For debugging one inference, not for a rollout.")
    ap.add_argument("--strict-gates", action="store_true")
    ap.add_argument("--engines", default="1", metavar="N|max",
                    help="--backend fpga only: shard every stage across N engines (or "
                         "'max' for each stage's own ceiling). 1 = the historical "
                         "single-engine programs, byte-for-byte.")
    ap.add_argument("--vis-8", dest="vis_8", action="store_true",
                    help="--backend fpga only: pin the vision encoder to 8 engines "
                         "regardless of --engines.")
    ap.add_argument("--no-fused-silu", dest="fused_silu", action="store_false",
                    help="prefix MLP: composed silu instead of the fused LALU path")
    ap.add_argument("--image-norm", default="pm1", choices=["pm1", "unit"],
                    help="pixel range fed to the ViT: pm1 = [-1,1] (SigLIP/SmolVLM2 "
                         "mean=std=0.5, the training preprocessing) or unit = [0,1]")
    ap.add_argument("--trace-out", default=None, metavar="PATH",
                    help="record EVERY inference (preprocessed images, token ids, mask, "
                         "state, and the hardware's normalized action chunk) to an .npz. "
                         "Replay it against the CPU oracle with --compare-trace to see "
                         "WHICH inference diverged, which the episode summary cannot.")
    ap.add_argument("--compare-trace", default=None, metavar="PATH",
                    help="offline, CPU-only: load a --trace-out file, recompute every "
                         "inference with the torch oracle and print the per-inference "
                         "divergence. Touches no hardware.")
    ap.add_argument("--video-out", default=str(_HERE / "data" / "libero" / "videos"))
    ap.add_argument("--no-video", action="store_true", help="skip mp4 writing")
    args = ap.parse_args()

    if args.compare_trace:
        compare_trace(args.compare_trace)
        return

    print("=" * 74)
    print("  VeraPulse PulseVLA-LIBERO-0.5B  |  LIBERO closed-loop eval")
    print(_POLICY_WARNING)
    print("=" * 74, flush=True)

    max_steps = args.max_steps or _SUITE_MAX_STEPS[args.task_suite]
    np.random.seed(args.seed)

    results_path = pathlib.Path(args.results_out) if args.results_out else (
        _HERE / "data" / "libero" / f"results_verapulse_{args.task_suite}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    episodes, done_keys = ([], set())
    if args.resume:
        episodes, done_keys = _load_previous(results_path)
        print(f"[eval] --resume: {len(done_keys)} episodes already in {results_path}; "
              f"they will be SKIPPED.", flush=True)

    # ---- environment + task list -------------------------------------------------
    # The LIBERO import is inside this branch so --dry-run never touches MuJoCo/robosuite.
    if args.dry_run:
        n_all_tasks = 10
        get_task = _FakeTask
        get_init_states = lambda tid: [None] * max(1, args.trials)
        make_env = lambda tid: _FakeEnv(
            seed=args.seed + tid,
            succeed_after=None)   # per-trial override below
    else:
        # LIBERO's benchmark.get_task_init_states() torch.load()s PICKLED NUMPY .init
        # files. torch >= 2.6 defaults weights_only=True and refuses them, so every task
        # dies before the sim starts. These files are plain data from the LIBERO clone;
        # restore the pre-2.6 default at this import site only (pi05's shim, same reason).
        import torch as _torch
        if not getattr(_torch.load, "_libero_weights_only_shim", False):
            _orig_load = _torch.load

            def _torch_load_compat(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig_load(*a, **kw)
            _torch_load_compat._libero_weights_only_shim = True
            _torch.load = _torch_load_compat

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.task_suite]()
        n_all_tasks = suite.n_tasks
        get_task = suite.get_task
        get_init_states = suite.get_task_init_states

        def make_env(tid):
            t = suite.get_task(tid)
            bddl = (pathlib.Path(get_libero_path("bddl_files"))
                    / t.problem_folder / t.bddl_file)
            e = OffScreenRenderEnv(bddl_file_name=str(bddl),
                                   camera_heights=LIBERO_ENV_RESOLUTION,
                                   camera_widths=LIBERO_ENV_RESOLUTION)
            e.seed(args.seed)
            return e

    n_tasks = min(args.tasks, n_all_tasks) if args.tasks else n_all_tasks

    # ---- backend + preprocessing, built ONCE -------------------------------------
    if args.dry_run:
        pre = None                       # verapulse_test import is FPGA-free, but the
        try:                             # checkpoint may not be present on a dev box;
            pre = _VeraPulsePre(args.image_norm)   # use it when it loads, fake it if not.
            print("[eval] dry-run: using the REAL tokenizer + norm_stats.", flush=True)
        except Exception as e:
            print(f"[eval] dry-run: real preprocessing unavailable ({e!r}); "
                  f"using synthetic tokens/stats. The loop is still exercised.", flush=True)
        backend = _FakeBackend(seed=args.seed)
    else:
        pre = _VeraPulsePre(args.image_norm)
        if args.backend == "oracle":
            if args.trace_out:
                ap.error("--trace-out is not supported with --backend oracle: it logs "
                         "the tensors OUR preprocessing produced, but the oracle runs "
                         "its own. The saved trace would look like a valid baseline and "
                         "would not be one.")
            backend = _OracleBackend(device=args.oracle_device, seed=args.seed,
                                     n_exec=pre.n_exec)
        elif args.backend == "cpu":
            backend = _CpuBackend(weights=args.weights, seed=args.seed,
                                  hw_gelu=args.cpu_hw_gelu)
        else:
            backend = _FpgaBackend(weights=args.weights, seed=args.seed,
                                   use_run_inference=args.use_run_inference,
                                   snr=args.snr, strict_gates=args.strict_gates,
                                   fused_silu=args.fused_silu,
                                   engines=args.engines,
                                   vis_engines=8 if args.vis_8 else None)
    print("[eval] backend ready.", flush=True)

    total_ep = sum(1 for _ in episodes)
    total_succ = sum(1 for e in episodes if e.get("success"))

    def _save():
        """Checkpoint AFTER EVERY EPISODE. The box hard-reboots under sustained FPGA
        load with no kernel log, so this file -- not the process -- is the run's memory.
        Written whole each time (tens of KB); a torn write is handled by _load_previous."""
        results_path.write_text(json.dumps({
            "model": "verapulse/pulsevla-libero-0.5b",
            "policy_validated": False,
            "warning": ("action expert ~17 dB below its bf16 floor, executed actions at "
                        "cos 0.97 vs the CPU oracle -- this success rate does NOT "
                        "characterize the model or the accelerator"),
            "backend": "dry-run" if args.dry_run else args.backend,
            "weights": args.weights, "task_suite": args.task_suite, "seed": args.seed,
            "task_start": args.task_start, "tasks": n_tasks, "trials": args.trials,
            "max_steps": max_steps, "replan_steps": args.replan_steps,
            "image_norm": args.image_norm,
            "entry": "run_inference" if args.use_run_inference else "vision+prefix+denoise",
            "successes": total_succ, "episodes_total": total_ep,
            "success_rate": total_succ / max(1, total_ep),
            "episodes": episodes,
        }, indent=2))

    def infer(base_u8, wrist_u8, state_raw, language):
        """One inference: preprocess -> backend -> de-normalize to robot units."""
        if getattr(backend, "takes_raw_obs", False):
            # The oracle owns its whole input pipeline AND returns robot units, so it
            # skips both _VeraPulsePre and pre.unnormalize. Tracing it against the
            # hardware would be comparing two different preprocessings, so --trace-out
            # is refused for this backend rather than silently logging a bad baseline.
            return backend.infer_raw(base_u8, wrist_u8, state_raw, language)
        if pre is not None:
            images = pre.images(base_u8, wrist_u8)
            ids, mask = pre.tokens(language)
            state32 = pre.state(state_raw)
            chunk = backend.infer(images, ids, mask, state32)
            if _trace is not None:
                # Store the PREPROCESSED images -- exactly what run_vision consumed --
                # so the oracle replays the same input rather than re-deriving it and
                # inheriting any preprocessing difference.
                _trace["images"].append(np.asarray(images, dtype=np.float32))
                _trace["ids"].append(np.asarray(ids, dtype=np.int64))
                _trace["mask"].append(np.asarray(mask, dtype=bool))
                _trace["state"].append(np.asarray(state32, dtype=np.float32))
                _trace["hw"].append(np.asarray(chunk, dtype=np.float32))
            return pre.unnormalize(chunk)
        # dry-run without a checkpoint: shapes only, same contract.
        images = np.stack([base_u8, wrist_u8]).astype(np.float32) / 127.5 - 1.0
        ids = np.zeros(48, dtype=np.int64)
        mask = np.zeros(48, dtype=bool)
        mask[:12] = True
        return backend.infer(images, ids, mask, np.zeros(32, dtype=np.float32))

    _trace = ({"images": [], "ids": [], "mask": [], "state": [], "hw": []}
              if args.trace_out else None)

    t_run0 = time.perf_counter()
    for task_id in range(args.task_start, args.task_start + n_tasks):
        task = get_task(task_id)
        init_states = get_init_states(task_id)

        # Skip the whole task (and its env construction, which is expensive and leaks an
        # EGL context) when --resume shows every trial already done.
        if all((task_id, tr) in done_keys for tr in range(args.trials)):
            print(f"[eval] task {task_id}: all {args.trials} trials already done "
                  f"(--resume) -- skipping.", flush=True)
            continue

        env = make_env(task_id)
        task_ep, task_succ = 0, 0
        for trial in range(args.trials):
            if (task_id, trial) in done_keys:
                print(f"[eval] task {task_id} trial {trial}: already done -- skipping.",
                      flush=True)
                continue
            if args.dry_run and args.dry_succeed_every:
                # Force a mix of successes and failures so the bookkeeping, the video
                # naming and the checkpoint file are all exercised in both directions.
                env.succeed_after = (args.wait_steps + 25
                                     if (total_ep + 1) % args.dry_succeed_every == 0
                                     else None)

            plan = collections.deque()
            replay, done, t, errored, n_infer = [], False, 0, False, 0
            t_ep = time.perf_counter()
            print(f"[eval] task {task_id} '{task.language}' trial {trial}", flush=True)
            # Isolate the episode: one bad rollout (sim glitch, a one-off device error)
            # is recorded as a failure and the run CONTINUES. Aborting the job would
            # throw away everything a long FPGA run has earned so far.
            try:
                env.reset()
                obs = env.set_init_state(init_states[trial])
                while t < max_steps + args.wait_steps:
                    if t < args.wait_steps:              # let the scene settle
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    # Rotate 180 to match the training-time camera convention, then
                    # resize 256 -> 512 (this model's ViT is 512, unlike pi05's 224).
                    # 180-degree rotation only. The RESIZE now happens inside
                    # _VeraPulsePre.images via upstream's own resize_with_pad -- doing it
                    # here as well would apply two different resamplers in series.
                    base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    if not args.no_video:
                        replay.append(base)
                    if not plan:
                        # ACTION CHUNKING: infer once, execute --replan-steps actions,
                        # then re-plan. This cadence (not per-step inference) is what
                        # makes a VLA tractable on hardware -- an inference is seconds.
                        state_raw = np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"]))
                        chunk = infer(base, wrist, state_raw, str(task.language))
                        n_infer += 1
                        plan.extend(np.asarray(chunk)[: args.replan_steps])
                    obs, _, done, _ = env.step(np.asarray(plan.popleft()).tolist())
                    if done:
                        break
                    t += 1
            except Exception as _e:
                import traceback
                errored, done = True, False
                print(f"[eval] !! task {task_id} trial {trial} ERRORED at step {t}: "
                      f"{_e!r} -- recording as failure, continuing", flush=True)
                traceback.print_exc()

            task_ep += 1
            total_ep += 1
            done = bool(done)
            if done:
                task_succ += 1
                total_succ += 1
            episodes.append({"task_id": task_id, "trial": trial,
                             "language": str(task.language), "steps": t,
                             "inferences": n_infer,
                             "seconds": round(time.perf_counter() - t_ep, 1),
                             "success": done, "errored": errored})
            _save()   # <- the reboot insurance. Never move this out of the trial loop.
            suffix = "error" if errored else ("success" if done else "failure")
            if not args.no_video and replay and not args.dry_run:
                try:
                    import imageio
                    vdir = pathlib.Path(args.video_out)
                    vdir.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(
                        vdir / f"verapulse_{args.task_suite}_t{task_id}_e{trial}_{suffix}.mp4",
                        replay, fps=10)
                except Exception as _ve:
                    # A missing codec must not cost an episode that already ran.
                    print(f"[eval]   (video write skipped: {_ve!r})", flush=True)
            print(f"[eval]   -> {suffix.upper()}  {n_infer} inferences  "
                  f"(running {total_succ}/{total_ep} = "
                  f"{100 * total_succ / max(1, total_ep):.1f}%)", flush=True)
            print(f"[eval]   checkpointed -> {results_path}")
            print(f"[eval]   if the box reboots, resume with:  MUJOCO_GL=egl python "
                  f"{pathlib.Path(__file__).resolve()} --task-suite {args.task_suite} "
                  f"--trials {args.trials} --results-out {results_path} --resume",
                  flush=True)
            del replay, plan
            gc.collect()
        print(f"[eval] task {task_id} success rate: {task_succ}/{task_ep}", flush=True)
        env.close()
        del env
        gc.collect()     # MuJoCo's EGL context is not released by close() alone

    _save()

    # ---- summary ------------------------------------------------------------------
    per_task = collections.defaultdict(lambda: [0, 0])
    for e in episodes:
        per_task[e["task_id"]][1] += 1
        per_task[e["task_id"]][0] += int(bool(e.get("success")))
    print("\n" + "=" * 74)
    print(f"  LIBERO {args.task_suite}  |  backend="
          f"{'dry-run' if args.dry_run else args.backend}  tasks={n_tasks}  "
          f"trials/task={args.trials}")
    for tid in sorted(per_task):
        s, n = per_task[tid]
        print(f"    task {tid:>2d}: {s}/{n} = {100 * s / max(1, n):5.1f}%")
    print(f"  OVERALL: {total_succ}/{total_ep} = "
          f"{100 * total_succ / max(1, total_ep):.1f}%   "
          f"({time.perf_counter() - t_run0:.0f}s)")
    print(f"  per-episode results -> {results_path}")
    print(_POLICY_WARNING)
    if _trace is not None and _trace["images"]:
        tp = pathlib.Path(args.trace_out)
        tp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(tp, **{k: np.stack(v) for k, v in _trace.items()})
        print(f"  inference trace ({len(_trace['images'])} inferences) -> {tp}")
        print(f"  replay on CPU:  python {pathlib.Path(__file__).name} "
              f"--compare-trace {tp}")
    print("=" * 74)


if __name__ == "__main__":
    main()
