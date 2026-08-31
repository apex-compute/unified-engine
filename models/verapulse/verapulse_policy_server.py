#!/usr/bin/env python3
"""LeRobot async-inference policy server backed by the VeraPulse FPGA accelerator.

    python models/verapulse/verapulse_policy_server.py --host=0.0.0.0 --port=8080 --engines=8

The robot_client on the Pi is UNMODIFIED. Only its --server_address changes.


WHAT THIS REPLACES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
lerobot's PolicyServer is ~440 lines of gRPC plumbing -- pickle framing, chunked
observation receive, the 1-deep observation queue, must_go/similarity filtering, timestep
dedup, FPS tracking, TimedAction stamping, the inference_latency floor. None of it cares
what runs the forward pass. Reimplementing any of it would be a chance to drift from the
client's expectations for no gain, so this file subclasses PolicyServer and overrides
exactly one hook.

The neural computation enters at exactly one place:

    SmolVLAPolicy.predict_action_chunk
      -> _get_action_chunk
        -> self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise)
        -> actions[:, :, :action_dim]     # unpad 32 -> 6

`sample_actions` IS the whole model. Everything on either side of it -- prepare_images
(resize-with-pad to 512, [0,1] -> [-1,1] for SigLIP), prepare_state (MEAN_STD + pad
6 -> 32), tokenization (the preprocessor puts lang_tokens/lang_masks in the batch; the
policy does NOT tokenize), the unpad to 6 DoF, and the postprocessor's unnormalization --
is lerobot's own code operating on the checkpoint's own buffers.

So we swap ONE attribute:

    policy.model.sample_actions = <this file's FPGA implementation>

and inherit every unit convention from the checkpoint. That matters more here than
elsewhere: the README is explicit that without the shipped normalizer buffers "the policy
emits garbage in real units", and a units error does not crash -- it produces a policy
that moves smoothly and wrongly. Reusing lerobot's pre/post pipeline makes that class of
bug structurally impossible rather than merely tested-for.


THE PIXEL RANGE, WHICH IS NOT WHAT verapulse_test.main() USES
-------------------------------------------------------------
verapulse_test.main() feeds the tower uint8/255 -> [0, 1]. The real model does NOT:
modeling_smolvla.prepare_images ends with `img = img * 2.0 - 1.0`, so SigLIP sees
[-1, 1]. Our run_vision applies no normalization of its own -- whatever it is handed is
what the tower sees.

The SNR gate cannot see this: it feeds the SAME pixels to hardware and to the torch
oracle, so any range gates clean. This server therefore takes prepare_images' output
VERBATIM and does not rescale it. Do not "fix" this to match main().


CAMERAS: TWO, NOT THE THREE config.json DECLARES
------------------------------------------------
The checkpoint's config.json input_features lists camera1/2/3, but the model only ever
saw two. dataset_meta/info.json has exactly two ('high', 'wrist' -> camera1/camera2,
44 episodes / 25949 frames), and prepare_images only synthesizes a zero-filled substitute
while `num_empty_cameras < config.empty_cameras` -- empty_cameras is 0, so that loop
breaks on iteration 0 and camera3 is never materialized. The prefix is genuinely
1 + 2*64 + 48 = 177 -> 192, not 241 -> 256. The engine is compiled for 2 slots and this
file asserts the batch agrees, because a silent 3rd slot would be a different model.
"""

import argparse
import os
import sys
import threading

# ---- variant selection MUST precede the verapulse_test import ------------------------
# Every dim on VeraPulse_UnifiedEngine (V_SLOTS, E_HIDDEN, NUM_LAYERS, ...) is a CLASS
# attribute read from the config when the class STATEMENT executes, so setting this after
# the import would change nothing while appearing to work.
os.environ.setdefault("VERAPULSE_VARIANT", "smolvla")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import verapulse_test as vp  # noqa: E402

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402


# ======================================================================================
# the FPGA forward pass
# ======================================================================================

class FPGABackend:
    """Owns the accelerator. Built ONCE, before the gRPC server starts serving.

    ENGINE LIFETIME IS A HARDWARE CONSTRAINT, not a performance choice. Every
    UnifiedEngine constructor -- and software_reset() -- DMA-writes 16 KB of noise to a
    hardcoded 0x80000000, which is this model's first stored weight. So the engine is
    constructed and weight_init'd once at startup and reused for the life of the process.
    A per-request engine would shred its own weights on request 2.
    """

    def __init__(self, engines=8, vis=None, prefix=None, denoise=None):
        self.lock = threading.Lock()   # one FPGA, and PolicyServer runs a 4-thread pool

        print(f"[fpga] variant={vp.VARIANT}  engines={engines}")
        self.ue = vp.VeraPulse_UnifiedEngine()
        # BEFORE weight_init: it builds the worker pool from these counts, and no engine
        # may be constructed afterwards (see the class docstring).
        vp.configure_engines(self.ue, engines, vis=vis, prefix=prefix, denoise=denoise,
                             tag="fpga")
        self.ue.weight_init()
        self.ue.tensor_init()

        # COMPILE EVERYTHING NOW. freeze=True makes a later lazy compile raise instead of
        # silently allocating program DRAM mid-rollout: compilation is ~7 s, which at
        # 13 Hz is ~90 dropped control ticks, and it would land on the first GetActions
        # rather than at startup where it is free.
        self.ue.precompile_all(stages=("vision", "prefix", "denoise"))

        self.slots = self.ue.V_SLOTS
        self.img = self.ue._cfg["vision"]["image_size"]
        self.chunk = self.ue.CHUNK
        self.pad_dim = self.ue.ACTION_DIM_PAD
        print(f"[fpga] ready: {self.slots} camera slot(s), prefix "
              f"{self.ue.PREFIX_LEN}->{self.ue.PREFILL_MAX_SEQ_LEN}, "
              f"{self.ue.N_STEPS} denoise steps, chunk {self.chunk}")

    # ---------------------------------------------------------------- sample_actions --
    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                       noise=None, **kwargs):
        """Drop-in for VLAFlowMatching.sample_actions.

        Returns (B, chunk_size, max_action_dim) NORMALIZED, exactly as upstream does --
        the caller unpads to 6 DoF and the postprocessor unnormalizes.
        """
        if kwargs.get("inference_delay") is not None or kwargs:
            # RTC (real-time chunking) reshapes the denoise loop; our compiled program is
            # the plain 10-step Euler schedule. rtc_config is null in this checkpoint, so
            # this should never fire -- but silently ignoring it would run a different
            # algorithm than the caller asked for.
            unexpected = {k: v for k, v in kwargs.items() if v is not None}
            if unexpected:
                raise NotImplementedError(
                    f"FPGA sample_actions got unsupported kwargs {sorted(unexpected)}; "
                    f"the compiled program is the fixed 10-step Euler schedule (no RTC)")

        bsize = state.shape[0]
        if bsize != 1:
            raise NotImplementedError(
                f"FPGA path is batch-1 (the robot client sends one observation); got "
                f"B={bsize}")

        # ---- images -> [slots, 512, 512, 3] ----------------------------------------
        # VERBATIM from prepare_images: already resize-with-padded to 512 and mapped to
        # [-1, 1] for SigLIP. No rescaling here (see module docstring).
        if len(images) != self.slots:
            raise ValueError(
                f"engine is compiled for {self.slots} camera slot(s) but the batch has "
                f"{len(images)}. This is a DIFFERENT MODEL, not a padding difference: "
                f"the prefix length and every RoPE position change with slot count. "
                f"Check the client's --robot.cameras and the checkpoint's empty_cameras.")
        for i, m in enumerate(img_masks):
            # A false mask means upstream substituted a blank image, which our compiled
            # attention bias does not model. Never silently accept one.
            if not bool(torch.as_tensor(m).all()):
                raise NotImplementedError(
                    f"camera slot {i} arrived masked-out; the compiled prefix bias "
                    f"assumes every slot is a real image")
        imgs = torch.stack([im[0].detach().float().cpu().permute(1, 2, 0)
                            for im in images], dim=0)          # [slots,H,W,C]
        if tuple(imgs.shape) != (self.slots, self.img, self.img, 3):
            raise ValueError(
                f"expected [{self.slots},{self.img},{self.img},3] images after "
                f"prepare_images, got {tuple(imgs.shape)}")

        # ---- language --------------------------------------------------------------
        # THE MASK IS NOT OPTIONAL. pad_language_to='max_length' pads every prompt to 48
        # tokens, so the real valid prefix is 1 + 2*64 + n_real_text and varies per
        # prompt. Dropping the mask would let the pad tokens attend as if they were
        # language AND -- because positions are cumsum(mask) -- shift every RoPE position
        # after them. Both are finite, plausible and wrong.
        tok = lang_tokens[0].detach().cpu().to(torch.long)
        mask = lang_masks[0].detach().cpu().to(torch.bool)

        # ---- state -----------------------------------------------------------------
        # prepare_state already normalized (MEAN_STD) and padded 6 -> 32.
        st = state[0].detach().float().cpu()

        nz = None if noise is None else noise[0].detach().float().cpu()

        with self.lock:
            vis = self.ue.run_vision(imgs)
            self.ue.run_prefix(vis, tok, st, text_mask=mask)
            self.ue.run_denoise(nz)
            padded = self.ue._last_denoise["x_t_padded"]        # [64, 64]

        out = padded[: self.chunk, : self.pad_dim].to(torch.float32)
        return out.unsqueeze(0).to(state.device)                # (1, 50, 32)


# ======================================================================================
# the server
# ======================================================================================

class FPGAPolicyServer(PolicyServer):
    """Stock PolicyServer with the model's forward pass rebound to the accelerator."""

    def __init__(self, config, backend):
        super().__init__(config)
        self.backend = backend      # None => --cuda: stock torch, no FPGA

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Let upstream load the policy + processors, then rebind the forward pass.

        Deliberately calls super() first rather than reimplementing the load: that is
        where from_pretrained and make_pre_post_processors run, and those carry the
        normalization buffers, the baked-in rename map and the tokenizer. We only take
        over afterwards.
        """
        out = super().SendPolicyInstructions(request, context)

        policy = getattr(self, "policy", None)
        if policy is None:
            return out       # upstream declined (server not running); nothing to bind

        if self.backend is None:
            # --cuda: leave the policy exactly as lerobot built it. This is the stock
            # upstream server, useful as an A/B reference for the FPGA path and as a
            # fallback. The device is whatever the CLIENT asked for via --policy_device,
            # because the pre/post processors were already constructed against it --
            # moving the model here without moving them would put the model and its
            # normalization buffers on different devices.
            dev = getattr(self, "device", "?")
            self.logger.info(f"--cuda: serving with stock torch on {dev} (FPGA disabled)")
            if str(dev) == "cpu":
                self.logger.warning(
                    "client requested policy_device=cpu, so this will run on CPU despite "
                    "--cuda. Pass --policy_device=cuda on the robot_client to use the GPU.")
            return out

        model = getattr(policy, "model", None)
        if model is None or not hasattr(model, "sample_actions"):
            raise RuntimeError(
                f"policy {type(policy).__name__} has no .model.sample_actions -- this "
                f"server only knows how to back SmolVLA's VLAFlowMatching")

        # Sanity-check the checkpoint against what the engine was COMPILED for. A
        # mismatch here means the client pointed us at a different checkpoint, and every
        # compiled program is wrong for it.
        cfg = policy.config
        want = {
            "chunk_size": (cfg.chunk_size, self.backend.chunk),
            "num_steps": (cfg.num_steps, self.backend.ue.N_STEPS),
        }
        # CAMERA COUNT IS DELIBERATELY NOT CHECKED HERE. cfg.image_features is what the
        # checkpoint DECLARES (3: camera1/2/3), not what the model ever sees. Only the
        # cameras the client actually sends are used: prepare_images synthesizes a
        # substitute for a missing key only while num_empty_cameras < cfg.empty_cameras,
        # and empty_cameras is 0, so camera3 is never materialized. The count that
        # matters is len(images) in the batch, which sample_actions checks per call.
        declared = len(cfg.image_features)
        if declared != self.backend.slots:
            self.logger.info(
                f"checkpoint declares {declared} camera(s), engine compiled for "
                f"{self.backend.slots}; empty_cameras={getattr(cfg, 'empty_cameras', 0)} "
                f"so only supplied cameras are used -- enforced per-batch in "
                f"sample_actions")
        bad = {k: v for k, v in want.items() if v[0] != v[1]}
        if bad:
            raise RuntimeError(
                f"checkpoint/engine mismatch {bad} (checkpoint, engine) -- the compiled "
                f"programs do not describe this checkpoint. Recompile the engine for it "
                f"or point the client at the matching checkpoint.")

        model.sample_actions = self.backend.sample_actions
        self.logger.info("forward pass rebound to the VeraPulse FPGA accelerator")
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0, never 127.0.0.1, or the Pi cannot reach the server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--inference_latency", type=float, default=0.033,
                    help="a FLOOR, not an addition; the FPGA pass is far above it")
    ap.add_argument("--obs_queue_timeout", type=float, default=2.0)
    ap.add_argument("--cuda", action="store_true",
                    help="serve with stock torch instead of the FPGA (no accelerator is "
                         "touched, nothing is compiled). The A/B reference for the FPGA "
                         "path. Pair with --policy_device=cuda on the robot_client, since "
                         "the client's device is what the pre/post processors are built "
                         "against.")
    ap.add_argument("--engines", default="8", metavar="N|max")
    ap.add_argument("--vis-engines", type=int, default=None)
    ap.add_argument("--prefix-engines", type=int, default=None)
    ap.add_argument("--denoise-engines", type=int, default=None)
    args = ap.parse_args()

    # Hardware FIRST: bring the accelerator up and compile everything before the port is
    # open. A client that connects to a server still compiling would time out its first
    # GetActions and, worse, would have already started the arm.
    if args.cuda:
        import torch as _t
        print(f"[server] --cuda: FPGA DISABLED, stock torch "
              f"(cuda available: {_t.cuda.is_available()})")
        backend = None
    else:
        backend = FPGABackend(engines=args.engines, vis=args.vis_engines,
                              prefix=args.prefix_engines, denoise=args.denoise_engines)

    cfg = PolicyServerConfig(host=args.host, port=args.port, fps=args.fps,
                             inference_latency=args.inference_latency,
                             obs_queue_timeout=args.obs_queue_timeout)

    from concurrent import futures
    import grpc
    from lerobot.transport import services_pb2_grpc

    server_impl = FPGAPolicyServer(cfg, backend)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server_impl, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    print(f"[server] VeraPulse FPGA policy server listening on {cfg.host}:{cfg.port}")
    print(f"[server] point the robot_client at --server_address=<this-host>:{cfg.port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[server] stopping")
        server.stop(0)


if __name__ == "__main__":
    main()
