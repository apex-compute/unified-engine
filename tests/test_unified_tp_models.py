"""Contracts between the unified server, the multi-card runners, and their tokenizers."""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SERVER = Path(__file__).resolve().parents[1] / "models/unified_server"
sys.path.insert(0, str(SERVER))
import tp_models
import unified_server


class Tokenizer:
    pieces = {1: "user", 2: "assistant", 3: "answer", 4: "<|image_pad|>", 5: "<|image|>",
              6: "<|image>", 7: "<image|>", 8: "<|channel|>", 9: "<|message|>", 10: "<|end|>",
              11: "analysis", 12: "final", 13: "hidden", 14: "<|channel>", 15: "<channel|>"}

    def __init__(self, image_marker=5):
        self.marker = image_marker

    def convert_ids_to_tokens(self, token):
        return self.pieces[token]

    def convert_tokens_to_ids(self, piece):
        return next(i for i, p in self.pieces.items() if p == piece)

    def apply_chat_template(self, turns, **opts):
        ids = []
        for turn in turns:
            if isinstance(turn["content"], list):
                ids.append(self.marker)
            ids.append(1 if turn["role"] == "user" else 2)
        return ids + [2]

    def encode(self, template, **opts):
        return template

    def decode(self, ids, **opts):
        return "".join(self.pieces[i] for i in ids)


def load_qwen_vision():
    spec = importlib.util.spec_from_file_location("qwen_vision", tp_models.ROOT / "qwen_vision.py")
    vision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vision)
    return vision


class TestResidentChat(unittest.TestCase):
    def backend(self, model, **opts):
        self.step = Mock(return_value=3)
        self.park = Mock()
        self.reset = Mock()
        self.stop = threading.Event()
        self.tokens = []
        return tp_models.Chat(model, self.stop, Mock(), tokenizer=Tokenizer(4 if model.startswith("q3") else 5),
                              step=self.step, reset=self.reset, park=self.park, stops={99}, devices=["xdma0"],
                              cards=[{"params_mb": 1}], load_s=2, **opts)

    def generate(self, backend, turns, **opts):
        return backend.generate({"messages": turns, "conversation": "a", "max_new_tokens": 2, **opts},
                                lambda token, text: self.tokens.append((token, text)))

    def test_stream_and_reset_each_turn(self):
        backend = self.backend("q35_9b")
        self.step.side_effect = [3, 3, 99]
        msgs = [{"role": "user", "content": "hello"}]
        result = self.generate(backend, msgs)
        self.assertEqual(self.tokens, [(3, "answer"), (99, "")])
        self.assertEqual(result["generated_tokens"], 2)
        self.assertGreater(result["decode_wall_tok_s"], 0)
        self.step.side_effect = None
        result = self.generate(backend, backend.history + msgs, max_new_tokens=1)
        self.assertEqual(result["prompt_tokens"], 4)
        self.assertEqual(self.reset.call_count, 2)
        self.assertEqual(self.park.call_count, 3)

    def test_image_reuse_and_truncation(self):
        for model in ("q38_27b", "gemma4_26b"):
            with self.subTest(model=model):
                qwen = model.startswith("q3")
                encode = Mock(return_value=([20, 21], (1, 2)) if qwen else [20, 21])
                block, positions = Mock(), Mock() if qwen else None
                backend = self.backend(model, encode_image=encode, block_step=block, positions=positions)
                image_only = [{"role": "user", "content": ""}]
                user = [{"role": "user", "content": "Tell me more"}]
                first = self.generate(backend, image_only, image="uploaded.png")
                self.assertEqual(first["image_tokens"], 2)
                block.assert_called_once_with([20, 21], 0 if qwen else 1)
                if qwen:
                    positions.assert_called_once_with(4, [(0, 1, 2)])
                history = image_only + [{"role": "assistant", "content": first["text"]}]
                second = self.generate(backend, history + user)
                self.assertEqual(second["image_tokens"], 2)
                encode.assert_called_once_with("uploaded.png")
                backend.limit = 4
                trimmed = self.generate(backend, backend.history + user)
                self.assertEqual(trimmed["history_dropped"], 2)
                self.assertEqual(trimmed["image_tokens"], 0)
                if qwen:
                    positions.assert_called_with(2, [])
                self.assertEqual(self.generate(backend, user, conversation="another")["history_dropped"], 0)

    def test_stop_and_prompt_overflow(self):
        backend = self.backend("q35_9b")
        self.stop.set()
        user = [{"role": "user", "content": "hello"}]
        result = self.generate(backend, user)
        self.assertEqual(result["generated_tokens"], 0)
        self.step.assert_not_called()
        backend.limit = 2
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.generate(backend, user)
        with self.assertRaisesRegex(ValueError, "vision weights"):
            self.generate(backend, user, image="file")
        self.assertEqual(self.park.call_count, 4)

    def test_qwen_rotary_tables_upload(self):
        import numpy as np
        import torch
        vision = load_qwen_vision()
        positions = vision.rope_positions(6, [(1, 2, 2)])
        self.assertEqual(positions["pos"], [[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [3, 3, 3]])
        port, engine = Mock(DMA_DEVICE_H2C="dma"), Mock()
        engine._fa_cos_pad_dram, engine._fa_neg_sin_pad_dram, engine._fa_sin_hi_pad_dram = 0, 1000, 2000
        engine.dma_write.side_effect = lambda device, addr, rows, size: size
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "rope.npz"
            vision.save_rope(dest, 1, positions["pos"], 6, 10000., [1, 1, 1])
            with np.load(dest) as tables:
                vision.upload_rope(port, engine, tables)
        cos, nsin, sin = [call.args for call in engine.dma_write.call_args_list]
        self.assertEqual([(c[1], c[3]) for c in (cos, nsin, sin)], [(12, 60), (1012, 60), (2012, 60)])
        self.assertTrue(torch.equal(nsin[2], -sin[2]))
        self.assertEqual(cos[2][0, 0], torch.tensor(1.).cos().bfloat16())
        self.assertTrue(torch.equal(cos[2][:, 3:], torch.zeros((5, 3))))

    def test_vision_shares_only_unused_card_memory(self):
        from user_dma_core import UnifiedEngine

        def decoder(base):
            ue = object.__new__(UnifiedEngine)
            ue._next_params_dram_addr, ue._params_limit = base + 134, base + 260
            ue._tensor_dram_base = ue._tensor_dram_addr = ue._tensor_dram_peak = base + 640
            ue._program_dram_base, ue._next_program_dram_addr = base + 832, base + 960
            ue.allocate_tensor_dram(65)
            ue.reset_tensor_dram_addr()
            return ue

        vision = load_qwen_vision()
        for count in (1, 2):
            with self.subTest(engines=count):
                engines = [decoder(i * 1024) for i in range(count)]
                card = vision.VisionCard(None, engines, "", 1024)
                for bank in reversed(range(count)):
                    self.assertEqual(card._alloc(65), bank * 1024 + 192)
                    self.assertEqual(card._alloc(64), bank * 1024 + 768)
                if count == 2:
                    self.assertEqual(card._alloc(64), 960)
                with self.assertRaisesRegex(MemoryError, "contiguous"):
                    card._alloc(1)
                self.assertEqual(engines[-1].get_program_dram_addr(), (count - 1) * 1024 + 960)

    def test_visible_channels(self):
        for model, sequence in [("gptoss20b", [8, 11, 9, 13, 10, 8, 12, 9, 3]),
                                ("gemma4_26b", [14, 13, 15, 3])]:
            with self.subTest(model=model):
                answer = tp_models.Answer(Tokenizer(), model)
                self.assertEqual("".join(answer.push(token) for token in sequence), "answer")
                self.assertEqual(answer.answer, "answer")


class TestServerDispatch(unittest.TestCase):
    def test_catalog_and_runner_arguments(self):
        cards = ["xdma6", "xdma2", "xdma5", "xdma3", "xdma0", "xdma1"]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"APEX_TP_WORK": tmp}), \
                patch.object(tp_models.importlib, "import_module") as module, patch.object(sys, "path", sys.path.copy()):
            for model, (_, runner, slices, limit) in tp_models.MODELS.items():
                with self.subTest(model=model):
                    path = tp_models.plan_file(model)
                    path.parent.mkdir(parents=True)
                    path.write_text("[]")
                    self.assertFalse(tp_models.static(model)["ready"])
                    path.write_text(json.dumps({"P": 10}))
                    self.assertEqual(unified_server.chat_static(model)["max_context"], limit)
                    module.return_value.cards_in_pci_order.return_value = cards
                    callback = Mock()
                    tp_models.run(model, "", callback)
                    module.assert_called_with(runner)
                    module.return_value.main.assert_called_with(
                        ["--tp", slices, "--engines", "2", "--max-context", str(limit), "--devs", ",".join(cards[:5])],
                        serve=callback)
                    with self.assertRaisesRegex(ValueError, "five distinct"):
                        tp_models.run(model, ",".join(cards[:4] + cards[:1]), callback)

    def test_detection_result_and_device_release(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(unified_server, "OUT_DIR", tmp), \
                patch.object(tp_models, "ROOT", Path(tmp)), patch.dict(os.environ, {"YOLO26_WEIGHTS": __file__}):
            board = unified_server.Board(sys.executable, "xdma2", 8)
            job = unified_server.DetectionJob("sample", "upload.png", "sample.png")
            detections = {"detections": [{"class": "ship", "confidence": 0.9, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
                          "card_s": 0.1}
            def detector(cmd, **kwargs):
                self.assertEqual(cmd[cmd.index("--dev") + 1], "xdma2")
                self.assertEqual(cmd[cmd.index("--image") + 1], job.source)
                Path(cmd[cmd.index("--result-json") + 1]).write_text(json.dumps(detections))
                Path(cmd[cmd.index("--out") + 1]).touch()
                return Mock(returncode=0)
            with patch.object(board, "stop_all") as release, patch.object(unified_server.subprocess, "run", detector):
                board._run_detection(job)
                release.assert_called_once()
            job.state = "done"
            self.assertEqual(job.public()["result"], detections)
            self.assertEqual(job.public()["image"], "/api/jobs/sample/image.png")
