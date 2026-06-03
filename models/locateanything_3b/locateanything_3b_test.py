#!/usr/bin/env python3
"""LocateAnything-3B — Stage 1 hybrid: GPU(MoonViT + connector) -> FPGA decoder.

The vision encoder + MLP connector run on GPU (our verified own-PyTorch impl in
locateanything_3b_cpu_test.py). Their [N, 2048] output is DMA'd into the FPGA's
VIS_ENCODER_OUT_DRAM, and the Qwen2.5-3B decoder runs prefill+decode on the
board to emit grounding boxes. This is the first step of the staged migration:
later the connector, then MoonViT, move onto the FPGA at the same DRAM seam.

The decoder reuses the qwen2.5_vl_3b engine verbatim (dimensionally identical LM):
  * config redirected to locateanything_3b_fpga_config.json (vocab 152681, 1D rope)
  * LM weights from locateanything_3b_lm_if4.bin (see _gen_lm_bin.py)
  * vision-weight load skipped (vision is on GPU)
  * image-token id remapped to the placeholder run_prefill scans for

Run (apex-compute env, board attached):
  python locateanything_3b_test.py --query car
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import torch
from PIL import Image

LA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(LA_DIR))
QWEN_PATH = os.path.join(REPO_ROOT, "models", "qwen2.5_vl_3b", "qwen2.5_vl_3b_test.py")
FPGA_CFG = os.path.join(LA_DIR, "locateanything_3b_fpga_config.json")
ENGINE_IMG_PLACEHOLDER = 151655   # what run_prefill() scans for; we remap onto it
LA_IMAGE_TOKEN = 151665           # <IMG_CONTEXT> in LocateAnything


def _import_qwen_engine():
    """Import the dotted-filename qwen engine module via importlib."""
    spec = importlib.util.spec_from_file_location("qwen_engine", QWEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


def _gpu_vision(query, prompt_kind, image_path, device="cuda", compute_vit=True, keep_model=False):
    """Build the MoonViT model, process the image, tokenize. Optionally compute the
    GPU vit/connector reference. Returns a dict; the FPGA-vision path reuses the model
    + pixel_values to run MoonViT on the board instead.

    Keys: vit ([N,4608] bf16 cpu or None), conn_ref ([N,2048] or None), prefill_seq,
    n_img, wh, model (or None), pixel_values, grid_hw, cfg."""
    import locateanything_3b_cpu_test as la
    cfg = json.load(open(os.path.join(LA_DIR, "locateanything_3b_config.json")))
    model = la.LocateAnything(cfg).eval()
    la.load_weights(model, la.MODEL_DIR, device, torch.bfloat16)

    improc = la.ImageProcessor(merge_kernel_size=tuple(cfg["vision_config"]["merge_kernel_size"]))
    img = Image.open(image_path).convert("RGB")
    pv, grid_hw = improc(img)
    mk = cfg["vision_config"]["merge_kernel_size"]
    n_img = (grid_hw[0] * grid_hw[1]) // (mk[0] * mk[1])

    vit = conn_ref = None
    if compute_vit:
        with torch.no_grad():
            vit = model.vision_features(pv.to(device, dtype=torch.bfloat16), grid_hw)  # [N,4608]
            conn_ref = model.mlp1(vit).to(torch.bfloat16).cpu()                        # [N,2048]
            vit = vit.to(torch.bfloat16).cpu()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(la.MODEL_DIR, trust_remote_code=True)
    text = la.build_prompt_text(la.PROMPTS[prompt_kind].format(q=query), n_img)
    ids = tok([text], return_tensors="pt").input_ids[0].tolist()
    # remap our <IMG_CONTEXT> onto the placeholder run_prefill replaces with vision
    ids = [ENGINE_IMG_PLACEHOLDER if t == LA_IMAGE_TOKEN else t for t in ids]

    if not keep_model:
        del model
        model = None
        if device == "cuda":
            torch.cuda.empty_cache()
    return dict(vit=vit, conn_ref=conn_ref, prefill_seq=tuple(ids), n_img=n_img,
                wh=(img.width, img.height), model=model, pixel_values=pv,
                grid_hw=grid_hw, cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(REPO_ROOT, "test_samples", "vette.jpg"))
    ap.add_argument("--prompt-kind", default="detect",
                    choices=["detect", "ground_multi", "ground_one", "point"])
    ap.add_argument("--query", default="car")
    ap.add_argument("--check-connector", action="store_true",
                    help="diff FPGA connector output vs the bit-exact GPU connector")
    ap.add_argument("--fpga-vision", action="store_true",
                    help="run MoonViT on the FPGA (stitched op flow) instead of the GPU")
    ap.add_argument("--check-vision", action="store_true",
                    help="with --fpga-vision: parity-check FPGA vit[N,4608] vs the GPU reference")
    ap.add_argument("--vision-precision", default="if4", choices=["if4", "bf16"],
                    help="MoonViT encoder weight precision (if4 fits the params window; "
                         "bf16 ~0.98GB may overflow)")
    ap.add_argument("--vision-debug", action="store_true",
                    help="run MoonViT op-by-op (separate execute per op) with a NaN/Inf "
                         "check after each stage; halts at the first bad stage")
    ap.add_argument("--stop-after", default=None,
                    help="with --vision-debug: force a halt after this stage label "
                         "(e.g. L0.flash, L0.fc0_gelu, final_norm)")
    ap.add_argument("--vision-no-mask", action="store_true",
                    help="with --vision-debug: run flash WITHOUT the pad-key mask bias "
                         "(isolation: output will be wrong but should be NaN-free)")
    args = ap.parse_args()

    # ---- 1. Vision prep (build model, process image, tokenize) ----
    fpga_vision = args.fpga_vision
    want_gpu_vit = (not fpga_vision) or args.check_vision
    mode = "FPGA MoonViT" if fpga_vision else "GPU MoonViT"
    print(f"=== Vision: {mode} -> FPGA connector + decoder ===")
    t0 = time.time()
    prep = _gpu_vision(args.query, args.prompt_kind, args.image,
                       compute_vit=want_gpu_vit, keep_model=fpga_vision)
    vit = prep["vit"]; conn_ref = prep["conn_ref"]; prefill_seq = prep["prefill_seq"]
    n_img = prep["n_img"]; (W, H) = prep["wh"]; vis_cfg = prep["cfg"]
    print(f"  image_tokens={n_img}  prefill_len={len(prefill_seq)}  ({time.time()-t0:.1f}s)")

    # ---- 2. Build FPGA decoder ----
    print("\n=== Building FPGA decoder (loads LM bin to DRAM) ===")
    lm_bin = os.path.join(LA_DIR, "locateanything_3b_bin", "locateanything_3b_lm_if4.bin")
    if not os.path.exists(lm_bin):
        print("  LM weight bin missing -> generating (one-time, ~45s) ...")
        import _gen_lm_bin
        _gen_lm_bin.main()
    # dummy vision bin so the engine's weight_init existence check passes (vision is on GPU)
    open(os.path.join(LA_DIR, "locateanything_3b_bin", "locateanything_3b_vision_UNUSED.bin"), "a").close()
    qe = _import_qwen_engine()

    def _la_load_config(script_dir):
        cfg = json.load(open(FPGA_CFG))
        wd = {"LAYER_WEIGHT_SIZE": cfg["file_info"]["layer_size"]}
        for k, r in cfg.get("regions", {}).items():
            wd[k] = qe._parse_offset(r["offset"]); wd[f"{k}_SIZE"] = r["size"]
        for k, r in cfg.get("non_layer_regions", {}).items():
            wd[k] = qe._parse_offset(r["offset"]); wd[f"{k}_SIZE"] = r["size"]
        cfg["_weight_defs"] = wd
        return cfg
    qe._load_config = _la_load_config

    class LADecoder(qe.Qwen25VL3B_UnifiedEngine):
        def _load_vision_weights(self, *a, **k):
            print("  [stage2] MoonViT on GPU -> skipping FPGA vision-weight load")

        def load_connector(self):
            """Load the connector (mlp1) bf16 weights into params DRAM."""
            conn_bin = os.path.join(LA_DIR, "locateanything_3b_bin", "locateanything_3b_connector.bin")
            man_path = conn_bin.rsplit(".", 1)[0] + ".json"
            if not os.path.exists(conn_bin):
                print("  connector bin missing -> generating ...")
                import _gen_connector_bin
                _gen_connector_bin.main()
            man = json.load(open(man_path))
            blob = open(conn_bin, "rb").read()

            def _ld(key):
                m = man[key]
                import numpy as np
                arr = np.frombuffer(blob[m["offset"]:m["offset"] + m["size"]], dtype=np.uint16).copy()
                t = torch.from_numpy(arr).view(torch.bfloat16).reshape(m["shape"])
                return qe.store_weight(self, t)

            self.conn_ln_g = _ld("mlp1.0.weight")   # LayerNorm gamma [4608]
            self.conn_ln_b = _ld("mlp1.0.bias")     # LayerNorm beta  [4608]
            self.conn_w0 = _ld("mlp1.1.weight")     # Linear0 [2048,4608]
            self.conn_b0 = _ld("mlp1.1.bias")       # Linear0 bias [2048]
            self.conn_w2 = _ld("mlp1.3.weight")     # Linear1 [2048,2048]
            self.conn_b2 = _ld("mlp1.3.bias")       # Linear1 bias [2048]
            self._conn_in, self._conn_out = man["mlp1.1.weight"]["shape"][1], man["mlp1.3.weight"]["shape"][0]
            self._conn_hidden = man["mlp1.1.weight"]["shape"][0]
            print(f"  connector loaded: LayerNorm({self._conn_in}) -> "
                  f"Linear({self._conn_in},{self._conn_hidden}) -> GELU -> "
                  f"Linear({self._conn_hidden},{self._conn_out})")

        def run_connector(self, vit):
            """FPGA connector: vit[N,4608] -> VIS_ENCODER_OUT_DRAM[N,2048].
            LayerNorm(g,b) -> matmul+bias+GELU -> matmul+bias. Reuses the stub
            vision scratch buffers (ping-pong); no new DRAM.

            The compute cores only EMIT instructions into the capture buffer; they
            run on hardware solely via start_capture -> write_to_dram ->
            start_execute_from_dram -> wait_queue (same scaffold as compile_encoder).
            Without this wrapper the matmuls never execute and the output is stale."""
            N, Cin = vit.shape
            assert Cin == self._conn_in, f"connector in {Cin} != {self._conn_in}"
            IN = self.VIS_MERGED_DRAM          # holds [N,4608] in, later [N,2048]
            LN = self.VIS_MERGER_INTER_DRAM    # holds LayerNorm output [N,4608]

            # input embeddings -> DRAM (immediate host->device DMA, fine pre-capture)
            self.dma_to_accelerator_memory(IN, vit.contiguous().flatten())

            # capture the 3-core connector program
            self.clear_inst_id()
            self.start_capture()
            # 0) LayerNorm(4608) with gamma+beta
            self.layer_norm_core_dram(M=N, N=Cin, A_DRAM_ADDR=IN, OUTPUT_DRAM_ADDR=LN,
                                      GAMMA_DRAM_ADDR=self.conn_ln_g, BETA_DRAM_ADDR=self.conn_ln_b)
            # 1+2) Linear(4608->2048) + bias + GELU  (engine GELU = sigmoid-approx)
            self.matmat_mul_core(M=N, K=Cin, N=self._conn_hidden,
                                 A_DRAM_ADDR=LN, B_DRAM_ADDR=self.conn_w0,
                                 OUTPUT_DRAM_ADDR=IN,
                                 C_DRAM_ADDR=self.conn_b0, bias_mode="broadcast_N",
                                 gelu_enable=True)
            # 3) Linear(2048->2048) + bias -> VIS_ENCODER_OUT_DRAM
            self.matmat_mul_core(M=N, K=self._conn_hidden, N=self._conn_out,
                                 A_DRAM_ADDR=IN, B_DRAM_ADDR=self.conn_w2,
                                 OUTPUT_DRAM_ADDR=self.VIS_ENCODER_OUT_DRAM,
                                 C_DRAM_ADDR=self.conn_b2, bias_mode="broadcast_N")
            self.stop_capture()
            self.generate_instruction_halt()

            # transient program: run it before prefill compiles (which reuses this region)
            prog_addr = self.get_program_dram_addr()
            self.write_captured_instructions_to_dram(prog_addr)
            self.clear_capture_buffer()
            self.start_execute_from_dram(prog_addr)
            self.wait_queue(120.0)
            self._vis_num_tokens = N

    ue = LADecoder(script_dir=LA_DIR)
    ue.load_connector()

    # ---- 2b. FPGA MoonViT encoder (optional): produces vit[N_merged,4608] in DRAM ----
    if fpga_vision:
        import moonvit_encoder as mv
        print("\n=== FPGA MoonViT encoder (stitched op flow) ===")
        model = prep["model"]
        N = prep["grid_hw"][0] * prep["grid_hw"][1]
        mv.moonvit_load_weights(ue, model, vis_cfg, precision=args.vision_precision)
        mv.moonvit_setup_dram(ue, N, vis_cfg)
        mv.moonvit_prepare_input(ue, model, prep["pixel_values"], prep["grid_hw"], vis_cfg)
        if args.vision_debug:
            # op-by-op execute + NaN check; halts at the first bad stage (or --stop-after)
            mv.moonvit_run_staged(ue, vis_cfg, prep["grid_hw"], stop_after=args.stop_after,
                                  mask_pad=not args.vision_no_mask)
        else:
            prog = mv.moonvit_compile_encoder(ue, vis_cfg, prep["grid_hw"])
            mv.moonvit_run_encoder(ue, prog)
        # read vit[N_merged,4608] back from DRAM (also feeds the connector + optional parity)
        vit = ue.dma_from_accelerator_memory(ue.VIS_MERGED_DRAM, (ue.MV_N_MERGED, 4608)).cpu()
        if args.check_vision and prep["vit"] is not None:
            ref = prep["vit"].float()
            f = vit.float()
            diff = (f - ref).abs()
            denom = ref.abs().mean().clamp_min(1e-6)
            cos = torch.nn.functional.cosine_similarity(f.flatten(), ref.flatten(), dim=0)
            print(f"  [check-vision] FPGA vit vs GPU: max|Δ|={diff.max():.4f} "
                  f"mean|Δ|={diff.mean():.5f} rel={diff.mean()/denom:.4f} cos={cos:.5f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 3. FPGA connector: vit[N,4608] -> VIS_ENCODER_OUT_DRAM[N,2048] ----
    ue.run_connector(vit)
    print(f"  FPGA connector -> {n_img} vision tokens @ VIS_ENCODER_OUT_DRAM 0x{ue.VIS_ENCODER_OUT_DRAM:X}")

    if args.check_connector:
        conn_fpga = ue.dma_from_accelerator_memory(
            ue.VIS_ENCODER_OUT_DRAM, (n_img, ue._conn_out)).cpu().float()
        ref = conn_ref.float()
        diff = (conn_fpga - ref).abs()
        denom = ref.abs().mean().clamp_min(1e-6)
        print(f"  [check] connector FPGA vs GPU: max|Δ|={diff.max():.4f}  "
              f"mean|Δ|={diff.mean():.5f}  rel={diff.mean()/denom:.4f}")

    # ---- 4. Prefill + decode on FPGA ----
    print("\n=== Prefill ===")
    pa, pre, gflops = ue.compile_prefill()
    ue.run_prefill(pa, pre, prefill_seq=prefill_seq, gflops=gflops, has_image=True)

    print("\n=== Decode ===")
    da, gpt = ue.compile_decoder()
    ue._rope_offset = 0  # plain 1D rope, sequential positions (no mRoPE)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ue.run_decoder(da, token_id=prefill_seq[-1], gflops=gpt, repetition_penalty=1.0)
    answer = buf.getvalue()
    print("OUTPUT:", answer.strip())

    # ---- 5. Draw overlay (reuse cpu_test renderer) ----
    import re
    import locateanything_3b_cpu_test as la
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(LA_DIR, f"{stem}_boxes_fpga.jpg")
    path, nb, npt = la.draw_overlay(args.image, answer, out_path)
    boxes = re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
    print(f"\n{len(boxes)} boxes: {boxes}")
    print(f"overlay saved -> {path}  ({nb} boxes, {npt} points)")


if __name__ == "__main__":
    main()
