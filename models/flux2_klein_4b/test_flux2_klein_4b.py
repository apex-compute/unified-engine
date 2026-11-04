#!/usr/bin/env python3
"""Host-only tests for the FLUX.2 [klein] 4B port (no FPGA needed).

    python -m unittest discover -s models/flux2_klein_4b -p 'test_*.py'

Fast tests always run; the ones that need the checkpoint / diffusers / the
exported bins skip themselves when those are absent.
"""
import json
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import flux2_klein_4b_ref as ref  # noqa: E402
import flux2_klein_4b_weights as W  # noqa: E402

BIN = os.path.join(HERE, "flux2_klein_4b_bin")


def _snap():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(ref.HF_REPO, local_files_only=True,
                                 allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*",
                                                 "tokenizer/*", "transformer/*", "vae/*"])
    except Exception:
        return None


def snr(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(10 * torch.log10(a.pow(2).sum() / (a - b).pow(2).sum().clamp(min=1e-30)))


class TestGeometry(unittest.TestCase):
    def test_token_counts(self):
        g = ref.Flux2Geometry(512, 512, 512, 4)
        self.assertEqual((g.latent_h, g.latent_w, g.token_h, g.token_w, g.n_img, g.n_joint), (64, 64, 32, 32, 1024, 1536))
        g = ref.Flux2Geometry(256, 256, 128, 4)
        self.assertEqual((g.n_img, g.n_joint), (256, 384))

    def test_pack_roundtrip(self):
        x = torch.randn(1, 32, 64, 64)
        p = ref.patchify_latents(x)
        self.assertEqual(tuple(p.shape), (1, 128, 32, 32))
        # channel = c*4 + py*2 + px
        self.assertTrue(torch.equal(p[0, 4 * 3 + 1 * 2 + 1], x[0, 3, 1::2, 1::2]))
        self.assertTrue(torch.equal(ref.unpatchify_latents(p), x))
        t = ref.pack_latents(p)
        self.assertEqual(tuple(t.shape), (1, 1024, 128))
        self.assertTrue(torch.equal(ref.unpack_latents(t, 32, 32), p))

    def test_schedule(self):
        s = ref.flow_match_sigmas(1024, 4)
        self.assertEqual(len(s), 5)
        self.assertAlmostEqual(float(s[0]), 1.0, places=6)
        self.assertEqual(float(s[-1]), 0.0)
        self.assertTrue(all(s[i] > s[i + 1] for i in range(4)))
        # the shift is monotone in the token count: more tokens -> larger sigmas
        s2 = ref.flow_match_sigmas(4096, 4)
        self.assertTrue(all(s2[i] >= s[i] for i in range(1, 4)))


class TestRope(unittest.TestCase):
    def test_rotate_half_equivalence(self):
        """Interleaved-pair RoPE on x == rotate-half RoPE (engine table) on the
        row-permuted x; q.k dot products are preserved."""
        torch.manual_seed(0)
        g = ref.Flux2Geometry(256, 256, 64, 4)
        ids = torch.cat([ref.text_position_ids(g.n_txt), ref.image_position_ids(g)], 0)
        cos, sin = ref.rope_cos_sin(ids, g.axes_dims_rope, g.rope_theta)
        S, D = cos.shape
        H = 3
        q = torch.randn(1, S, H, D)
        k = torch.randn(1, S, H, D)
        q_ref = ref.apply_rope_interleaved(q, cos, sin)
        k_ref = ref.apply_rope_interleaved(k, cos, sin)
        perm = ref.rotate_half_permutation(D)
        table = ref.rope_table_rotate_half(cos, sin).float()        # [c|c|-s|s]
        c_tab, s_tab = table[:, :D], table[:, D:]

        def rot_half(x):
            xp = x[..., perm]
            h = D // 2
            a = xp * c_tab[None, :, None, :]
            b = torch.cat([xp[..., h:] * s_tab[None, :, None, :h], xp[..., :h] * s_tab[None, :, None, h:]], -1)
            return a + b
        q_hw, k_hw = rot_half(q), rot_half(k)
        # same values up to the permutation (bf16 table rounding)
        self.assertGreater(snr(q_ref[..., perm], q_hw), 40)
        scores_ref = torch.einsum("bshd,bthd->bhst", q_ref, k_ref)
        scores_hw = torch.einsum("bshd,bthd->bhst", q_hw, k_hw)
        self.assertGreater(snr(scores_ref, scores_hw), 40)

    def test_permutation_is_involution_safe(self):
        perm = ref.rotate_half_permutation(128)
        self.assertEqual(sorted(perm.tolist()), list(range(128)))
        self.assertEqual(perm[:3].tolist(), [0, 2, 4])
        self.assertEqual(perm[64:67].tolist(), [1, 3, 5])

    def test_qk_weight_permutation_preserves_attention(self):
        torch.manual_seed(1)
        D, hd, Hh = 256, 64, 4
        perm = ref.rotate_half_permutation(hd)
        wq = torch.randn(D, D)
        gamma = torch.rand(hd)
        x = torch.randn(5, D)
        q = (x @ wq.T).view(5, Hh, hd)
        q_n = ref._rms_norm(q, gamma, 1e-6)
        wq_p = W.permute_qk_rows(wq, hd, perm)
        q_p = (x @ wq_p.T).view(5, Hh, hd)
        q_pn = ref._rms_norm(q_p, W.permute_norm_gamma(gamma, perm), 1e-6)
        self.assertTrue(torch.allclose(q_pn, q_n[..., perm], atol=1e-5))


class TestPacking(unittest.TestCase):
    def test_layout_planner_matches_builder(self):
        cfg = W.load_config()
        plan = W.plan_te_layout(cfg)
        # the builder must land each region at the planned offset
        bb = W._BinBuilder()
        for name, kind, shape in W.te_layer_spec(cfg):
            if kind == "bf16":
                bb.add_bf16(name, torch.zeros(*shape, dtype=torch.bfloat16))
            else:
                bb.add_quant(name, torch.randn(*shape), cfg["quantization"]["precision"], cfg["quantization"]["block_size"])
        _, regions = bb.finish()
        for k, v in regions.items():
            self.assertEqual((v["offset"], v["size"]), (plan["layer_regions"][k]["offset"], plan["layer_regions"][k]["size"]), k)

    def test_if4_roundtrip_is_close(self):
        torch.manual_seed(0)
        w = torch.randn(128, 256) * 0.02
        wq = ref.fake_quantize_tensor(w, "if4", 64)
        self.assertGreater(snr(w, wq), 15)

    def test_conv_if8_roundtrip(self):
        torch.manual_seed(0)
        w = torch.randn(64, 96, 3, 3)
        codes, scales, rel = W.quantize_conv_if8(w)
        self.assertEqual(tuple(codes.shape), (64, 96, 3, 3))
        self.assertEqual(tuple(scales.shape), (64, 9 * 2))
        self.assertTrue((scales < 0).all())
        self.assertGreater(snr(w, W.dequantize_conv_if8(codes, scales)), 35)

    @unittest.skipUnless(os.path.exists(os.path.join(BIN, "params_dit.json")), "bins not exported")
    def test_exported_manifests_match_plan(self):
        cfg = W.load_config()
        with open(os.path.join(BIN, "params_te.json")) as f:
            W.check_meta_matches_plan(json.load(f), W.plan_te_layout(cfg), "te")
        with open(os.path.join(BIN, "params_dit.json")) as f:
            W.check_meta_matches_plan(json.load(f), W.plan_dit_layout(cfg), "dit")


class TestCompile(unittest.TestCase):
    def test_offline_compile_small(self):
        """Emit both programs for a 128x128 / 64-token geometry without a board."""
        import flux2_klein_4b_test as T
        cfg = T.load_config()
        geo = ref.Flux2Geometry(128, 128, 64, 2)
        ue = T.Flux2KleinEngine(cfg, geo, offline=True)
        tmp = os.path.join(BIN, "test_programs")
        os.makedirs(tmp, exist_ok=True)
        meta = ue.compile_programs(os.path.join(tmp, "programs.bin"), os.path.join(tmp, "programs.json"))
        self.assertGreater(meta["te_program_size"], 0)
        self.assertGreater(meta["dit_program_size"], 0)
        total = sum(im["size"] for im in meta["images"])
        self.assertEqual(total % 32, 0)
        self.assertEqual(os.path.getsize(os.path.join(tmp, "programs.bin")), total)
        self.assertEqual(len(meta["sigmas"]), 3)
        # all three phases' DRAM stays below 4 GiB
        self.assertLess(ue._program_dram_base + meta["images"][0]["size"], 1 << 32)
        self.assertLess(ue.VAE_TENSOR_BASE, ue._program_dram_base)

    def test_offline_compile_eight_engines(self):
        """Sharded streams: one program image per engine and phase, every image inside its slot."""
        import flux2_klein_4b_test as T
        cfg = T.load_config()
        geo = ref.Flux2Geometry(128, 128, 64, 2)
        ue = T.Flux2KleinEngine(cfg, geo, num_engines=8, offline=True)
        tmp = os.path.join(BIN, "test_programs_8")
        os.makedirs(tmp, exist_ok=True)
        meta = ue.compile_programs(os.path.join(tmp, "programs.bin"), os.path.join(tmp, "programs.json"))
        self.assertEqual(meta["engines"], 8)
        self.assertEqual(len(meta["te_worker_prog_addrs"]), 7)
        self.assertEqual(len(meta["dit_worker_prog_addrs"]), 7)
        self.assertEqual(len([im for im in meta["images"] if im["phase"] in ("primary", "te", "dit")]), 1 + 2 * 7)
        self.assertEqual(len(meta["flag_clear_addrs"]), 8)
        for im in meta["images"]:
            slot = ue._program_dram_base + im["engine"] * ue.PROGRAM_SLOT
            self.assertGreaterEqual(im["addr"], slot)
            self.assertLessEqual(im["addr"] + im["size"], slot + ue.PROGRAM_SLOT)
        self.assertEqual(os.path.getsize(os.path.join(tmp, "programs.bin")), sum(im["size"] for im in meta["images"]))
        self.assertEqual(len(meta["vae_worker_prog_addrs"]), 7)
        self.assertEqual(len([im for im in meta["images"] if im["phase"] == "vae"]), 7)

    def test_vae_phase_layout(self):
        """The VAE phase covers the decoder, every buffer is inside the params arena, the
        plan is reproducible, and the constant image builder fills exactly the planned bytes."""
        import flux2_klein_4b_test as T
        import flux2_klein_4b_vae as V
        cfg = T.load_config()
        geo = ref.Flux2Geometry(128, 128, 64, 2)
        ue = T.Flux2KleinEngine(cfg, geo, num_engines=2, offline=True)
        tmp = os.path.join(BIN, "test_programs_vae")
        os.makedirs(tmp, exist_ok=True)
        meta = ue.compile_programs(os.path.join(tmp, "programs.bin"), os.path.join(tmp, "programs.json"))
        vm = meta["vae"]
        self.assertEqual(vm["latent"], [32, 16, 16])
        self.assertEqual(vm["out_shape"], [3, 128, 128])
        kinds = {}
        for st in vm["steps"]:
            kinds[st["kind"]] = kinds.get(st["kind"], 0) + 1
        self.assertEqual(kinds, {"zero_pairs": 1, "conv": 34, "conv1x1": 6, "gn": 30, "silu": 29, "add": 15,
                                 "attention": 1, "upsample": 3})
        self.assertLessEqual(vm["const_base"] + vm["const_size"], vm["arena_end"])
        self.assertLessEqual(vm["arena_end"], ue._tensor_dram_base)
        self.assertGreater(meta["vae_program_size"], 0)
        self.assertEqual(len(meta["vae_worker_prog_addrs"]), 1)
        # the plan is deterministic (the run-time side re-derives every address from it)
        again = V.Flux2VaeOneShot(ue, cfg).plan(geo.latent_h, geo.latent_w)
        self.assertEqual(again, vm)
        if not os.path.exists(os.path.join(BIN, "vae_weights.pt")):
            self.skipTest("vae weights not exported (constant image needs them)")
        w = torch.load(os.path.join(BIN, "vae_weights.pt"), map_location="cpu")
        img = ue.vae.const_image(w)
        self.assertEqual(len(img), vm["const_size"])
        lat = ue.vae.pack_latent(torch.zeros(32, 16, 16, dtype=torch.bfloat16))
        self.assertEqual(lat.numel() * 2, vm["input_bytes"])


@unittest.skipUnless(_snap(), "checkpoint not downloaded")
class TestAgainstDiffusers(unittest.TestCase):
    def setUp(self):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            self.skipTest("diffusers not installed")
        self.snap = _snap()

    def test_vae_decoder_matches_diffusers_fp32(self):
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(os.path.join(self.snap, "vae"), torch_dtype=torch.float32).eval()
        mine = ref.Flux2VaeDecoder(ref.load_safetensors_dir(os.path.join(self.snap, "vae")),
                                   ref.load_json(os.path.join(self.snap, "vae", "config.json")))
        torch.manual_seed(0)
        z = torch.randn(1, 32, 16, 16)
        with torch.no_grad():
            a = vae.decode(z, return_dict=False)[0]
        b = mine.decode(z)
        self.assertGreater(snr(a, b), 80)

    def test_vae_if8_reference_close_to_fp32(self):
        if not os.path.exists(os.path.join(BIN, "vae_weights.pt")):
            self.skipTest("vae weights not exported")
        import flux2_klein_4b_vae as V
        cfg = W.load_config()
        w = torch.load(os.path.join(BIN, "vae_weights.pt"), map_location="cpu")
        mine = ref.Flux2VaeDecoder(ref.load_safetensors_dir(os.path.join(self.snap, "vae")),
                                   ref.load_json(os.path.join(self.snap, "vae", "config.json")))
        torch.manual_seed(0)
        z = torch.randn(32, 16, 16)
        a = mine.decode(z[None])[0]
        b = V.reference_decode(w, cfg, z)
        self.assertGreater(snr(a, b), 25)

    def test_dit_matches_diffusers_fp32_tiny(self):
        """One fp32 forward of the real 4B transformer on CPU (~16 GB RAM, ~10 s)."""
        if os.environ.get("FLUX2_SKIP_SLOW"):
            self.skipTest("FLUX2_SKIP_SLOW set")
        from diffusers import Flux2Transformer2DModel
        geo = ref.Flux2Geometry(128, 128, 64, 4)
        m = Flux2Transformer2DModel.from_pretrained(os.path.join(self.snap, "transformer"), torch_dtype=torch.float32).eval()
        mine = ref.Flux2Transformer(dict(m.state_dict()), geo)
        torch.manual_seed(0)
        lat = torch.randn(1, geo.n_img, 128)
        emb = torch.randn(1, geo.n_txt, 7680)
        ids = torch.cat([ref.text_position_ids(geo.n_txt), ref.image_position_ids(geo)], 0)
        cos, sin = ref.rope_cos_sin(ids, geo.axes_dims_rope, geo.rope_theta)
        sigma = 0.9
        with torch.no_grad():
            t = torch.tensor([sigma * 1000.0]) / 1000.0
            a = m(hidden_states=lat, encoder_hidden_states=emb, timestep=t, img_ids=ids[geo.n_txt:].float(),
                  txt_ids=ids[:geo.n_txt].float(), guidance=None, return_dict=False)[0]
        # ref.temb rounds the timestep through bf16 like the bf16 pipeline does; in fp32 feed the
        # same bf16-rounded sinusoid both ways by comparing with a tolerance that absorbs it.
        b = mine.forward(lat, emb, sigma, cos, sin)
        self.assertGreater(snr(a, b), 40)


if __name__ == "__main__":
    unittest.main()
