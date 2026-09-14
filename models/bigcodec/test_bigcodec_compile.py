"""Whole-model graph semantics and liveness planning without hardware access."""

from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from bigcodec_common import DEFAULT_CHECKPOINT, load_models
from bigcodec_compile import Arena, build_graph, compiled_sample_count, plan_memory, convolution_reuse_pixels, _prepare_operation, emit_operation, shared
from bigcodec_vq.module import ResidualUnit


def tiny_models():
    encoder = SimpleNamespace(block=torch.nn.Sequential(
        torch.nn.Conv1d(1, 64, 1), ResidualUnit(64),
        torch.nn.Conv1d(64, 1024, kernel_size=200, stride=200)))
    decoder = SimpleNamespace(quantizer=torch.nn.Identity(), model=torch.nn.Sequential(
        torch.nn.ConvTranspose1d(1024, 1, kernel_size=200, stride=200), torch.nn.Tanh()))
    return encoder, decoder


class BigCodecGraphTest(unittest.TestCase):
    def test_official_padding_dimensions(self):
        for samples, expected in ((1, 200), (199, 200), (200, 400), (201, 400), (3200, 3400)):
            self.assertEqual(compiled_sample_count(samples), expected)
        for samples in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                compiled_sample_count(samples)

    def test_free_blocks_are_merged_and_reused(self):
        arena = Arena(0x1000, 0x2000)
        a, b = arena.allocate(128), arena.allocate(256)
        arena.release(a, 128)
        arena.release(b, 256)
        self.assertEqual(arena.allocate(384), 0x1000)
        self.assertEqual(arena.cursor, 0x1180)
        with self.assertRaises(ValueError):
            arena.allocate(0x1000)

    def test_adaptive_reuse_bounds_large_weights_and_uses_small_gather(self):
        for inputs, kernel, expected in ((1, 7, 32), (48, 7, 9), (96, 7, 4), (192, 7, 3), (768, 7, 1)):
            module = torch.nn.Conv1d(inputs, 64, kernel)
            self.assertEqual(convolution_reuse_pixels(module), expected)
        module = torch.nn.ConvTranspose1d(96, 48, kernel_size=4, stride=2)
        # Two-tap phases pack 96 channels into three profitable gather blocks.
        self.assertEqual(convolution_reuse_pixels(module), 21)

    def test_bf16_dispatch_has_own_workspace_and_same_audio_contract(self):
        backend = SimpleNamespace(conv_scratch_bytes=Mock(return_value=128 * 1024),
            prepare_conv1d=Mock(return_value="bf16-conv"),
            prepare_conv_transpose1d=Mock(return_value="bf16-transpose"), emit_conv=Mock())
        with patch("bigcodec_compile._bf16_convolutions", return_value=backend):
            graph = plan_memory(build_graph(*tiny_models(), 400, conv_precision="bf16"))
            self.assertEqual(graph.conv_precision, "bf16")
            self.assertEqual(graph.tokens_address, shared.TENSOR_BASE + 400 * 128)
            self.assertEqual(graph.output_bytes, 402 * 128)
            for operation in graph.operations:
                if operation.op not in ("conv1d", "conv_transpose1d"):
                    continue
                self.assertEqual(operation.scratch_bytes, 128 * 1024)
                self.assertEqual(operation.weight_reuse_pixels, 1)
                _prepare_operation(graph, object(), 0x90000000, 0x90002000, operation)
                with patch("bigcodec_compile.zero"):
                    emit_operation(object(), graph, operation, 0x90000000, 0x90002000)
                self.assertEqual(backend.emit_conv.call_args.args[1], operation.plan)
            self.assertTrue(backend.prepare_conv1d.called)
            self.assertTrue(backend.prepare_conv_transpose1d.called)
            self.assertTrue(any(call.kwargs.get("transpose") for call in backend.conv_scratch_bytes.call_args_list))
        with self.assertRaises(ValueError):
            build_graph(*tiny_models(), 400, conv_precision="fp16")

    def test_live_tensors_never_alias_and_outputs_are_contiguous(self):
        graph = plan_memory(build_graph(*tiny_models(), 400))
        self.assertEqual(graph.tokens_offset, 400 * 64 * 2)
        self.assertEqual(graph.tokens_address, shared.TENSOR_BASE + graph.tokens_offset)
        self.assertEqual(graph.output_bytes, (400 + 2) * 64 * 2)
        self.assertEqual(graph.tensor_end, graph.scratch_address + graph.scratch_bytes)
        self.assertTrue(all(t.address % 128 == 0 for t in graph.tensors.values()))
        for index in range(len(graph.operations)):
            live = [t for t in graph.tensors.values() if t.first <= index <= t.last]
            for left_index, left in enumerate(live):
                for right in live[left_index + 1:]:
                    self.assertFalse(left.address < right.address + right.size_bytes
                        and right.address < left.address + left.size_bytes,
                        (index, left.name, right.name))
        allocated = [t.address for name, t in graph.tensors.items() if name not in ("input", graph.output)]
        self.assertLess(len(set(allocated)), len(allocated))

    def test_unknown_graph_operations_and_arena_overflow_fail(self):
        encoder, decoder = tiny_models()
        encoder.block.insert(0, torch.nn.ReLU())
        with self.assertRaisesRegex(ValueError, "unsupported"):
            build_graph(encoder, decoder, 400)
        encoder, decoder = tiny_models()
        with self.assertRaisesRegex(ValueError, "input arena"):
            plan_memory(build_graph(encoder, decoder, 4_000_000))

    @unittest.skipUnless(DEFAULT_CHECKPOINT.exists(), "official checkpoint not downloaded")
    def test_official_graph_shapes_match_actual_torch_forward(self):
        torch.set_num_threads(1)
        encoder, decoder = load_models(remove_weight_norm=True)
        graph = plan_memory(build_graph(encoder, decoder, 400))
        counts = Counter(operation.op for operation in graph.operations)
        self.assertEqual(counts, {"conv1d": 69, "conv_transpose1d": 5,
            "activation": 72, "lstm": 2, "add": 30, "quantizer": 1, "tanh": 1})
        observed, hooks = {}, []
        for operation in graph.operations:
            if operation.module is None:
                continue
            def hook(_module, _inputs, output, name=operation.output):
                if isinstance(output, tuple):
                    output = output[0]
                observed[name] = (int(output.shape[-1]), int(output.shape[1]))
            hooks.append(operation.module.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                encoded = encoder(torch.zeros(1, 1, 400))
                quantized, _, _ = decoder(encoded, vq=True)
                output = decoder(quantized, vq=False)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(tuple(output.shape), (1, 1, 400))
        self.assertEqual(len(observed), len(graph.operations) - counts["add"])
        for name, shape in observed.items():
            self.assertEqual(shape, graph.tensors[name].shape, name)


if __name__ == "__main__":
    unittest.main()
