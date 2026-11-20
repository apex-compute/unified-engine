"""OllApeX vision: the model's image encoder, run on one FPGA card.

The encoder (patch embedding, 27 transformer blocks, patch merger) runs as one card program
on a card that also holds a slice of the language model. It only uses card memory the language
model does not: its weights go after the LM weights, its buffers after the LM's highest
activation address, its program after the LM program. So running it never disturbs a
conversation in progress.

Host side this module only prepares inputs: it resizes and cuts the image into patches, and
looks up the position tables (interpolated learned positions, rotary angles) for the grid.

Weights come from vision_prep.py (padded layout, see there). Follows the structure of the
vendor's Qwen3.5-2B vision encoder, generalized to padded head and MLP widths.
"""
import time

import numpy as np
import torch

BF16 = 2
PX = 128                      # image sides are multiples of this, so the patch count is a multiple of 64
DETAIL = {"low": 256, "medium": 384, "high": 512}
N_MAX = (max(DETAIL.values()) // 16) ** 2      # patches of the largest image (patch size 16)


def plan_size(w, h, detail="medium"):
    """(W, H) the image is resized to: long side ~ the detail level, both sides multiples of 128."""
    long_side = DETAIL.get(detail, DETAIL["medium"])
    s = long_side / float(max(w, h))
    W = max(PX, int(round(w * s / PX)) * PX)
    H = max(PX, int(round(h * s / PX)) * PX)
    return W, H


def tokens_for(W, H, patch=16, merge=2):
    return (W // (patch * merge)) * (H // (patch * merge))


def preprocess(image_path, W, H, dims):
    """Image -> patch rows [N_P, C*tps*ps*ps] in the model's order (spatial-merge-block major)."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((W, H), Image.BICUBIC)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5          # mean 0.5, std 0.5
    x = np.transpose(x, (2, 0, 1))                                          # [C, H, W]
    ps, tps, m = dims["patch"], dims["tps"], int(round(dims["MERGE"] ** 0.5))
    gh, gw = H // ps, W // ps
    x = np.broadcast_to(x[None], (tps,) + x.shape)                          # a still image fills every frame
    x = x.reshape(1, tps, 3, gh // m, m, ps, gw // m, m, ps)
    x = np.transpose(x, (0, 3, 6, 4, 7, 2, 1, 5, 8)).reshape(gh * gw, 3 * tps * ps * ps)
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16), gh, gw


def position_tables(gh, gw, pos_table, dims):
    """Learned positions resampled to the grid, and the rotary table in the padded head layout."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids
    m = int(round(dims["MERGE"] ** 0.5))
    grid = torch.tensor([[1, gh, gw]])
    idx, wts = get_vision_interpolation_indices_and_weights(grid, num_grid_per_side=dims["grid"], mode="bilinear",
                                                            align_corners=True, spatial_merge_size=m)
    pos = (pos_table.float()[idx] * wts[:, :, None].float()).sum(1).to(torch.bfloat16)      # [N_P, HID]
    pid = get_vision_position_ids(grid, m).float()                                             # [N_P, 2]
    half = dims["HD"] // 2
    inv_freq = 1.0 / (dims["theta"] ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    rot = (pid.unsqueeze(-1) * inv_freq).flatten(1)                                           # [N_P, half]
    n_p, hdp = rot.shape[0], dims["HDP"]
    cos = torch.ones(n_p, hdp // 2)
    sin = torch.zeros(n_p, hdp // 2)
    cos[:, :half] = rot.cos()
    sin[:, :half] = rot.sin()
    table = torch.empty(n_p, 2, hdp)
    table[:, 0, :hdp // 2] = cos;  table[:, 0, hdp // 2:] = cos
    table[:, 1, :hdp // 2] = -sin; table[:, 1, hdp // 2:] = sin
    rope = table.to(torch.bfloat16).repeat(dims["HEADS"], 1, 1).flatten()
    return pos, rope


class VisionCard:
    """Card-side encoder. `T` is the port module, `ue` the card's engine (shared with the LM)."""

    def __init__(self, T, ue, weights_path, tensor_base, tensor_limit):
        self.T, self.ue = T, ue
        self.weights_path = weights_path
        self.cursor = tensor_base            # next free byte of the region the LM never touches
        self.limit = tensor_limit
        self.A = None
        self.dims = None
        self.pos_table = None
        self.progs = {}
        self.B = None

    def load(self):
        if self.A is not None:
            return 0.0
        T, ue = self.T, self.ue
        t0 = time.time()
        W = torch.load(self.weights_path, weights_only=False, map_location="cpu")
        self.dims = d = W["dims"]
        self.pos_table = W["pos_table"]
        up = lambda t: T._upload_bf16(ue, t, t.numel())
        A = {"patch_w": up(W["patch_w"]), "patch_b": up(W["patch_b"]),
             "inv_n": up(torch.full((d["HID"],), 1.0 / d["HID"], dtype=torch.bfloat16)), "layers": []}
        for L in W["layers"]:
            a = {k: up(L[k]) for k in ("ln1_w", "ln1_b", "ln2_w", "ln2_b", "qkv_b", "proj_b", "fc1_b", "fc2_b")}
            for k in ("qkv", "proj", "fc1", "fc2"):
                a[k] = T._upload_fp4_pair(ue, L[k])              # (scale address, data address)
            A["layers"].append(a)
        mg = W["merger"]
        A["merger"] = {k: up(mg[k]) for k in ("ln_w", "ln_b", "fc1_b", "fc2_b")}
        for k in ("fc1", "fc2"):
            A["merger"][k] = T._upload_fp4_pair(ue, mg[k])
        self.A = A
        return time.time() - t0

    def _alloc(self, nbytes):
        ue = self.ue
        saved = ue._tensor_dram_addr
        ue._tensor_dram_addr = self.cursor
        addr = ue.allocate_tensor_dram(nbytes)
        self.cursor = ue._tensor_dram_addr
        ue._tensor_dram_addr = saved
        if self.cursor > self.limit:
            raise RuntimeError("no card memory left for the image encoder (%d MB over)" % ((self.cursor - self.limit) >> 20))
        return addr

    def program(self, N_P):
        if N_P in self.progs:
            return self.progs[N_P]
        T, ue, d, A = self.T, self.ue, self.dims, self.A
        assert N_P % 64 == 0 and N_P % d["MERGE"] == 0
        HID, HEADS, HD, HDP, QD = d["HID"], d["HEADS"], d["HD"], d["HDP"], d["QD"]
        QKV, MLPP, K_PIXEL, K_FC, OUT, DEPTH = 3 * QD, d["MLPP"], d["K_PIXEL"], d["K_FC"], d["OUT"], d["DEPTH"]
        N_M = N_P // d["MERGE"]
        hs = N_P * HDP * BF16                                    # one head's block in the head-major buffers
        if self.B is None:
            # One set of buffers, sized for the largest image, shared by the programs of all image sizes
            # (a program for a smaller image just uses the front of each buffer).
            n = N_MAX
            assert N_P <= n
            al = self._alloc
            self.B = {"PIXEL": al(n * K_PIXEL * BF16), "POS": al(n * HID * BF16), "PATCH_OUT": al(n * HID * BF16),
                      "BLOCK": al(n * HID * BF16), "LN1": al(n * HID * BF16), "QKV": al(n * QKV * BF16),
                      "LN2": al(n * HID * BF16), "Q": al(HEADS * n * HDP * BF16), "K": al(HEADS * n * HDP * BF16),
                      "V": al(HEADS * n * HDP * BF16), "ATTN": al(HEADS * n * HDP * BF16), "CONCAT": al(n * QD * BF16),
                      "PROJ": al(n * HID * BF16), "POST": al(n * HID * BF16), "FC1": al(n * MLPP * BF16),
                      "FC2": al(n * HID * BF16), "SCRATCH": al((HDP + n) * n * BF16 + n * HDP * BF16),
                      "BIAS": al(n * n * BF16), "EYE": al(64 * 64 * BF16), "ZEROS": al(HID * BF16),
                      "ROPE": al(HEADS * n * 2 * HDP * BF16), "MG_LN": al(n * HID * BF16),
                      "MG_FC1": al(n // d["MERGE"] * K_FC * BF16), "MG_OUT": al(n // d["MERGE"] * OUT * BF16)}
            H2C = T.DMA_DEVICE_H2C
            ue.dma_write(H2C, self.B["EYE"], torch.eye(64, dtype=torch.bfloat16).contiguous(), 64 * 64 * BF16)
            ue.dma_write(H2C, self.B["BIAS"], torch.zeros(n * n, dtype=torch.bfloat16), n * n * BF16)
            ue.dma_write(H2C, self.B["ZEROS"], torch.zeros(HID, dtype=torch.bfloat16), HID * BF16)
        B = self.B
        assert N_P <= N_MAX

        MT, US, U0 = T.MEMCPY_TYPE.URAM.value, T.URAM_SECTION.URAM_A.value, T.URAM_START_ADDR
        TYPE = T.TYPE
        rows_max = 65536 // HDP                                  # rows of one head that fit one on-chip memory load

        def residual(a, b, out):
            total, done = N_P * HID, 0
            while done < total:
                c = min(65536, total - done)
                off = done * BF16
                ue.accelerator_memory_to_sram(a + off, 0x00000, c)
                ue.accelerator_memory_to_sram(b + off, 0x80000, c)
                ue.eltwise_add_core(0x00000, 0x80000, 0x00000, c)
                ue.sram_to_accelerator_memory(0x00000, out + off, c)
                done += c

        def qmat(M, K, N, a, w, out, bias, reg, gelu=False):
            ue.matmat_mul_core(M=M, K=K, N=N, A_DRAM_ADDR=a, B_DRAM_ADDR=w[1], OUTPUT_DRAM_ADDR=out,
                               C_DRAM_ADDR=bias, bias_mode="broadcast_N", is_B_quantized=True, data_type=TYPE.IF4,
                               SCALE_DRAM_ADDR=w[0], gelu_enable=gelu, gpr_M_reg=reg)

        saved_regs = ue._isa_reg_counter
        ue._isa_reg_counter = 8                                  # keep clear of the registers the LM programs use
        vis_M = ue.alloc_isa_reg()
        ue.clear_capture_buffer()
        T.UnifiedEngine.start_capture(ue)
        ue.generate_instruction_add_set(vis_M, N_P)
        ue.matmat_mul_core(M=N_P, K=K_PIXEL, N=HID, A_DRAM_ADDR=B["PIXEL"], B_DRAM_ADDR=A["patch_w"],
                           OUTPUT_DRAM_ADDR=B["PATCH_OUT"], C_DRAM_ADDR=A["patch_b"], bias_mode="broadcast_N",
                           is_B_quantized=False, gpr_M_reg=vis_M)
        residual(B["PATCH_OUT"], B["POS"], B["BLOCK"])
        for li in range(DEPTH):
            L = A["layers"][li]
            ue.layer_norm_core_dram(M=N_P, N=HID, A_DRAM_ADDR=B["BLOCK"], OUTPUT_DRAM_ADDR=B["LN1"],
                                    GAMMA_DRAM_ADDR=L["ln1_w"], BETA_DRAM_ADDR=L["ln1_b"], gpr_M_reg=vis_M,
                                    ZEROS_DRAM_ADDR=B["ZEROS"], INV_N_DRAM_ADDR=A["inv_n"])
            qmat(N_P, HID, QKV, B["LN1"], L["qkv"], B["QKV"], L["qkv_b"], vis_M)
            # [N_P, 3*QD] token-major -> Q, K, V head-major [HEADS, N_P, HDP]
            for h in range(HEADS):
                for part, dst in ((0, B["Q"]), (QD, B["K"]), (2 * QD, B["V"])):
                    r0 = 0
                    while r0 < N_P:
                        r = min(rows_max, N_P - r0)
                        ue.ue_memcpy_from_dram(B["QKV"] + (r0 * QKV + part + h * HDP) * BF16, r * HDP * BF16, MT, U0, US,
                                               stride_bytes_per_chunk=HDP * BF16, stride_jump_bytes=QKV * BF16)
                        ue.ue_memcpy_to_dram(MT, US, U0, dst + h * hs + r0 * HDP * BF16, r * HDP * BF16)
                        r0 += r
            ue.generate_instruction_add_set(vis_M, HEADS * N_P)
            for buf in (B["Q"], B["K"]):
                ue.rope_hf_core_dram(M=HEADS * N_P, N=HDP, input_dram_addr=buf, output_dram_addr=buf,
                                     cos_dram_addr=B["ROPE"], sin_dram_addr=B["ROPE"] + HDP * BF16, gpr_M_reg=vis_M)
            ue.generate_instruction_add_set(vis_M, N_P)
            for h in range(HEADS):
                ue.unified_attention_core(batch=N_P, aligned_seq_len=N_P, head_dim=HDP,
                                          Q_DRAM_ADDR=B["Q"] + h * hs, K_DRAM_ADDR=B["K"] + h * hs,
                                          V_DRAM_ADDR=B["V"] + h * hs, BIAS_DRAM_ADDR=B["BIAS"],
                                          OUTPUT_DRAM_ADDR=B["ATTN"] + h * hs, SCRATCH_DRAM_ADDR=B["SCRATCH"],
                                          IDENTITY_DRAM_ADDR=B["EYE"], q_scale=HD ** -0.5)   # the real head width
            for h in range(HEADS):
                r0 = 0
                while r0 < N_P:
                    r = min(rows_max, N_P - r0)
                    ue.ue_memcpy_from_dram(B["ATTN"] + h * hs + r0 * HDP * BF16, r * HDP * BF16, MT, U0, US)
                    ue.ue_memcpy_to_dram(MT, US, U0, B["CONCAT"] + (r0 * QD + h * HDP) * BF16, r * HDP * BF16,
                                         stride_bytes_per_chunk=HDP * BF16, stride_jump_bytes=QD * BF16)
                    r0 += r
            qmat(N_P, QD, HID, B["CONCAT"], L["proj"], B["PROJ"], L["proj_b"], vis_M)
            residual(B["BLOCK"], B["PROJ"], B["POST"])
            ue.layer_norm_core_dram(M=N_P, N=HID, A_DRAM_ADDR=B["POST"], OUTPUT_DRAM_ADDR=B["LN2"],
                                    GAMMA_DRAM_ADDR=L["ln2_w"], BETA_DRAM_ADDR=L["ln2_b"], gpr_M_reg=vis_M,
                                    ZEROS_DRAM_ADDR=B["ZEROS"], INV_N_DRAM_ADDR=A["inv_n"])
            qmat(N_P, HID, MLPP, B["LN2"], L["fc1"], B["FC1"], L["fc1_b"], vis_M, gelu=True)
            qmat(N_P, MLPP, HID, B["FC1"], L["fc2"], B["FC2"], L["fc2_b"], vis_M)
            residual(B["POST"], B["FC2"], B["BLOCK"])
        mg = A["merger"]
        ue.layer_norm_core_dram(M=N_P, N=HID, A_DRAM_ADDR=B["BLOCK"], OUTPUT_DRAM_ADDR=B["MG_LN"],
                                GAMMA_DRAM_ADDR=mg["ln_w"], BETA_DRAM_ADDR=mg["ln_b"], gpr_M_reg=vis_M,
                                ZEROS_DRAM_ADDR=B["ZEROS"], INV_N_DRAM_ADDR=A["inv_n"])
        ue.generate_instruction_add_set(vis_M, N_M)
        # the 2x2 neighbours of a merge block are consecutive rows, so merging is just a reshape
        qmat(N_M, K_FC, K_FC, B["MG_LN"], mg["fc1"], B["MG_FC1"], mg["fc1_b"], vis_M, gelu=True)
        qmat(N_M, K_FC, OUT, B["MG_FC1"], mg["fc2"], B["MG_OUT"], mg["fc2_b"], vis_M)
        ue.generate_instruction_halt()
        T.UnifiedEngine.stop_capture(ue)
        n_inst = ue.capture_count
        size = ue.get_capture_instruction_size_bytes()
        addr = ue.get_program_dram_addr()
        ue.write_captured_instructions_to_dram(addr)
        ue.allocate_program_dram(size)
        ue.clear_capture_buffer()
        ue._isa_reg_counter = saved_regs
        self.progs[N_P] = {"B": B, "addr": addr, "instructions": n_inst, "N_M": N_M}
        return self.progs[N_P]

    def encode(self, image_path, W, H):
        """Returns (tokens int16 [N_M, OUT] as bf16 bits, info)."""
        T, ue = self.T, self.ue
        info = {"width": W, "height": H}
        info["weights_upload_s"] = self.load()
        d = self.dims
        t0 = time.time()
        pixel, gh, gw = preprocess(image_path, W, H, d)
        pos, rope = position_tables(gh, gw, self.pos_table, d)
        N_P = pixel.shape[0]
        info["host_prep_s"] = time.time() - t0
        t0 = time.time()
        new = N_P not in self.progs
        p = self.program(N_P)
        info["compile_s"] = (time.time() - t0) if new else 0.0
        B, H2C = p["B"], T.DMA_DEVICE_H2C
        ue.dma_write(H2C, B["PIXEL"], pixel.contiguous(), pixel.numel() * BF16)
        ue.dma_write(H2C, B["POS"], pos.contiguous(), pos.numel() * BF16)
        ue.dma_write(H2C, B["ROPE"], rope.contiguous(), rope.numel() * BF16)
        t0 = time.time()
        ue.start_execute_from_dram(p["addr"])
        while ue.is_queue_busy():
            time.sleep(0.005)
            if time.time() - t0 > 600:
                raise RuntimeError("the image encoder did not finish within 600 s")
        info["card_s"] = ue.report_latency_in_us() / 1e6
        info["wall_s"] = time.time() - t0
        out = torch.zeros(p["N_M"] * d["OUT"], dtype=torch.bfloat16)
        ue.dma_read(T.DMA_DEVICE_C2H, B["MG_OUT"], out, out.numel() * BF16)
        info.update({"patches": N_P, "tokens": p["N_M"], "grid": [gh, gw], "instructions": p["instructions"],
                     "tensor_mb_used": (self.cursor - (self.limit - (512 << 20))) / 1048576.0,
                     "finite": bool(torch.isfinite(out.float()).all()), "abs_mean": float(out.float().abs().mean())})
        return out.view(torch.int16).numpy().reshape(p["N_M"], d["OUT"]), info


def expand_image_tokens(ids, image_pad_id, images):
    """The chat template leaves ONE placeholder token per image; replace it by as many as the
    image has tokens. images: [{"tokens": int16 [n, H], "gh": rows, "gw": cols}] in prompt order.
    Returns (ids, vecs {index: bytes}, blocks [(start index, rows, cols)])."""
    out, vecs, blocks, it = [], {}, [], iter(images)
    for t in ids:
        if t != image_pad_id:
            out.append(t)
            continue
        im = next(it)
        n = im["gh"] * im["gw"]
        assert im["tokens"].shape[0] == n
        blocks.append((len(out), im["gh"], im["gw"]))
        for j in range(n):
            vecs[len(out)] = im["tokens"][j].tobytes()
            out.append(image_pad_id)
    return out, vecs, blocks


def rope_positions(n_ids, blocks):
    """Positions as the model defines them: text counts up by one; an image's tokens share one
    time position and take their row / column as height / width position; the text after the
    image continues max(rows, cols) later. Returns None without images, else
    {"i0": first non-default index, "pos": [[t, h, w], ...] from i0, "shift": index - position after the prompt}."""
    if not blocks:
        return None
    pos, cur, i, starts = [], 0, 0, {b[0]: b for b in blocks}
    while i < n_ids:
        if i in starts:
            _, gh, gw = starts[i]
            for r in range(gh):
                for c in range(gw):
                    pos.append([cur, cur + r, cur + c])
            i += gh * gw
            cur += max(gh, gw)
        else:
            pos.append([cur] * 3)
            cur += 1
            i += 1
    i0 = blocks[0][0]
    return {"i0": i0, "pos": pos[i0:], "shift": n_ids - cur}
