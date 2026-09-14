#!/usr/bin/env python3
"""Offline structural checks for Omni's one-round decode attention."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest

import torch


HERE = os.path.dirname(os.path.abspath(__file__))


def _load_omni_lm():
    name = "qwen2_5_omni_7b_lm_group_attention_test"
    path = os.path.join(HERE, "qwen2.5_omni_7b_lm.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Omni LM from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OMNI = _load_omni_lm()
VL = OMNI._vl_lm


class _Scheduler:
    num_engines = 8

    def __init__(self):
        self.engines = [f"engine-{idx}" for idx in range(self.num_engines)]
        self.events = []

    def worker_indices(self):
        return list(range(1, self.num_engines))

    def release(self):
        self.events.append(("release", 0))

    def begin_worker_round(self, engine):
        self.events.append(("begin", engine))

    def end_worker_round(self, engine):
        self.events.append(("end", engine))

    def join(self):
        self.events.append(("join", 0))

    @staticmethod
    def split_cols(width):
        cols = width // 8
        return [(engine * cols, cols) for engine in range(8)]


class _Emitter:
    LM_ATTN_SCRATCH_PER_ENGINE = tuple(
        0x100000 + idx * 0x10000 for idx in range(8)
    )
    _decode_attn_worker_regs = tuple(
        {"aligned": 20 + idx} for idx in range(7)
    )

    def __init__(self):
        self.calls = []

    def _emit_one_round_group_attention(self, ue, **kwargs):
        self.calls.append((ue, kwargs))
        return 10, 9


class GroupAttentionStructureTest(unittest.TestCase):
    def test_omni_policy_is_opt_in(self):
        self.assertFalse(
            VL.Qwen25VLLMMixin._decode_use_one_round_group_attention(None)
        )
        model = object.__new__(OMNI.Qwen25OmniLMMixin)
        model.multi_core = 8
        model._lm_dims = lambda: {
            "KVH": 4, "G": 7, "AHD": 128, "QH": 28,
        }
        self.assertTrue(model._decode_use_one_round_group_attention())

    def test_layout_and_all_eight_participants(self):
        group_size, head_dim, bpe = 7, 128, 2
        groups = [
            (kv, kv * group_size * head_dim * bpe, group_size)
            for kv in range(4)
        ]
        assignments = VL.Qwen25VLLMMixin._one_round_group_assignments(
            groups, 8
        )
        self.assertEqual(tuple(groups), assignments[:4])
        self.assertEqual((None, None, None, None), assignments[4:])
        self.assertEqual(
            [g[1] for g in groups],
            [0, 7 * 128 * 2, 14 * 128 * 2, 21 * 128 * 2],
        )

        model = _Emitter()
        scheduler = _Scheduler()
        total, scaled = (
            VL.Qwen25VLLMMixin._emit_one_round_group_attention_round(
                model,
                scheduler,
                assignments,
                aligned_kv=2048,
                aligned_kv_reg=4,
                k_base=0x200000,
                v_base=0x300000,
            )
        )
        self.assertEqual((total, scaled), (40, 36))
        self.assertEqual(
            [event for event in scheduler.events if event[0] == "begin"],
            [("begin", idx) for idx in range(1, 8)],
        )
        self.assertEqual(
            [event for event in scheduler.events if event[0] == "end"],
            [("end", idx) for idx in range(1, 8)],
        )
        self.assertEqual(scheduler.events[0], ("release", 0))
        self.assertEqual(scheduler.events[-1], ("join", 0))
        self.assertEqual(
            [call[0] for call in model.calls],
            [model, *scheduler.engines[1:4]],
        )
        self.assertEqual(
            [call[1]["group"] for call in model.calls], groups
        )

    def test_worker_scratch_covers_max_context(self):
        aligned_context, head_dim, group_size = 2048, 128, 7
        needed = (
            head_dim * aligned_context
            + group_size * aligned_context
            + group_size * head_dim
        )
        # Omni executes/allocates prefill at 8*64 rows in its entrypoint.
        prefill_rows = 512
        allocated = (
            (head_dim + prefill_rows) * prefill_rows
            + prefill_rows * head_dim
        )
        self.assertGreaterEqual(allocated, needed)

    def test_bf16_o_stripes_fit_and_cover_every_column(self):
        model = object.__new__(OMNI.Qwen25OmniLMMixin)
        model._multi_core_schedulers = {"decode": _Scheduler()}
        model.mc_arena = type("Arena", (), {"stride": 0x20000000})()
        model.bytes_per_element = 2
        model.PARAMS_BASE = 0x100000000
        model.PARAMS_LIMIT = 0x1E8000000
        model.TENSOR_BASE = 0x1E8000000

        stripes, layers = model._plan_decode_o_stripes(28, 3584)
        self.assertEqual(len(stripes), 8)
        self.assertEqual(
            [(s["col_offset"], s["cols"]) for s in stripes],
            [(engine * 448, 448) for engine in range(8)],
        )
        self.assertTrue(all(s["layer_bytes"] == 3_211_264 for s in stripes))
        self.assertTrue(all(s["end"] - s["base"] == 89_915_392 for s in stripes))
        self.assertEqual(stripes[-1]["end"], 0x1E55C0000)
        self.assertEqual(len(layers), 28)
        for layer in range(28):
            weight = layers[("o", layer)]
            self.assertEqual(weight.K, 3584)
            self.assertEqual(weight.N, 3584)
            self.assertEqual(sum(shard.cols for shard in weight.shards), 3584)


class DecodeNumericEquivalenceTest(unittest.TestCase):
    def test_m1_layout_aliases_are_byte_exact(self):
        torch.manual_seed(25)
        q_token_major = torch.randn(1, 28, 128, dtype=torch.bfloat16)
        q_head_major = q_token_major.reshape(28, 1, 128)
        self.assertTrue(
            torch.equal(q_token_major.reshape(-1), q_head_major.reshape(-1))
        )

        attention_head_major = torch.randn(28, 1, 128, dtype=torch.bfloat16)
        attention_token_major = attention_head_major.reshape(1, 28, 128)
        self.assertTrue(
            torch.equal(
                attention_head_major.reshape(-1),
                attention_token_major.reshape(-1),
            )
        )

    def test_grouped_rope_matches_per_head(self):
        torch.manual_seed(25)
        heads, width = 28, 128
        x = torch.randn(heads, width, dtype=torch.bfloat16)
        position = torch.tensor(37.0)
        half = width // 2
        inv = 1.0 / (
            1_000_000.0
            ** (torch.arange(half, dtype=torch.float32) / half)
        )
        phase = position * inv
        cos = torch.cat([phase.cos(), phase.cos()])
        signed_sin = torch.cat([-phase.sin(), phase.sin()])

        def rotate(rows):
            swapped = torch.cat([rows[..., half:], rows[..., :half]], dim=-1)
            return (rows.float() * cos + swapped.float() * signed_sin).to(
                torch.bfloat16
            )

        grouped = rotate(x)
        per_head = torch.stack([rotate(x[head]) for head in range(heads)])
        self.assertTrue(torch.equal(grouped, per_head))

    def test_one_round_attention_matches_per_head(self):
        torch.set_num_threads(1)
        torch.manual_seed(25)
        kv_heads, group_size, width = 4, 7, 128

        def attention(q_rows, k_rows, v_rows, bias):
            scaled_q = (
                q_rows.float() * (1.0 / math.sqrt(width))
            ).to(torch.bfloat16)
            score = (scaled_q.float() @ k_rows.float().T).to(torch.bfloat16)
            probability = torch.softmax(
                score.float() + bias.float(), dim=-1
            ).to(torch.bfloat16)
            return (probability.float() @ v_rows.float()).to(torch.bfloat16)

        for aligned in (64, 2048):
            q = torch.randn(
                kv_heads, group_size, width, dtype=torch.bfloat16
            )
            k = torch.randn(kv_heads, aligned, width, dtype=torch.bfloat16)
            v = torch.randn(kv_heads, aligned, width, dtype=torch.bfloat16)
            bias = torch.zeros(group_size, aligned, dtype=torch.bfloat16)
            bias[:, -13:] = float("-inf")
            grouped = torch.stack(
                [attention(q[g], k[g], v[g], bias) for g in range(kv_heads)]
            )
            per_head = torch.stack(
                [
                    torch.stack(
                        [
                            attention(
                                q[g, h:h + 1],
                                k[g],
                                v[g],
                                bias[h:h + 1],
                            )[0]
                            for h in range(group_size)
                        ]
                    )
                    for g in range(kv_heads)
                ]
            )
            torch.testing.assert_close(grouped, per_head, rtol=0, atol=2e-5)
            flat = grouped.reshape(kv_heads * group_size, width)
            for group in range(kv_heads):
                self.assertTrue(
                    torch.equal(
                        flat[group * group_size:(group + 1) * group_size],
                        grouped[group],
                    )
                )


if __name__ == "__main__":
    unittest.main()
