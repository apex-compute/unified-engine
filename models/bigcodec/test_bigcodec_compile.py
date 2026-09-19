"""Whole-model graph semantics and liveness planning without hardware access."""

from collections import Counter
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from bigcodec_common import DEFAULT_CHECKPOINT, load_models
from bigcodec_compile import Arena, build_graph, compiled_sample_count, ensure_checkpoint, plan_memory, convolution_reuse_pixels, _prepare_operation, emit_operation, _lstm_numerics, shared
from bigcodec_layout import LEGACY_LAYOUT, EXTENDED_LAYOUT, LARGE_PROGRAM_LAYOUT, layout_for_hardware
from bigcodec_vq.activations import SnakeBeta
from bigcodec_vq.alias_free_torch import Activation1d
from bigcodec_vq.module import ResidualUnit, ResLSTM


def tiny_models():
    encoder = SimpleNamespace(block=torch.nn.Sequential(
        torch.nn.Conv1d(1, 64, 1), ResidualUnit(64),
        torch.nn.Conv1d(64, 1024, kernel_size=200, stride=200)))
    decoder = SimpleNamespace(quantizer=torch.nn.Identity(), model=torch.nn.Sequential(
        torch.nn.ConvTranspose1d(1024, 1, kernel_size=200, stride=200), torch.nn.Tanh()))
    return encoder, decoder


def projection_models(code_dimensions=8):
    """Small temporal graph with the official 1024-channel VQ boundary."""
    encoder = torch.nn.Module()
    encoder.block = torch.nn.Sequential(
        torch.nn.Conv1d(1, 2, 200, stride=200), torch.nn.Tanh(),
        torch.nn.Conv1d(2, 1024, 3, padding=1))
    decoder = torch.nn.Module()
    decoder.quantizer = torch.nn.Module()
    decoder.quantizer.in_proj = torch.nn.Linear(1024, code_dimensions)
    decoder.model = torch.nn.Sequential(
        torch.nn.ConvTranspose1d(1024, 1, 200, stride=200), torch.nn.Tanh())
    return encoder.eval(), decoder.eval()


class BigCodecGraphTest(unittest.TestCase):
    def test_missing_default_checkpoint_is_fetched_but_custom_path_is_not(self):
        import bigcodec_compile as compiler
        with tempfile.TemporaryDirectory() as root:
            default = Path(root) / 'bigcodec.pt'
            with patch.object(compiler, 'DEFAULT_CHECKPOINT', default):
                with patch('bigcodec_fetch.fetch', return_value=default) as fetch:
                    self.assertEqual(ensure_checkpoint(default), default)
                    fetch.assert_called_once_with(default)
                with self.assertRaisesRegex(FileNotFoundError, 'Provide an existing --checkpoint'):
                    ensure_checkpoint(Path(root) / 'custom-missing.pt')

    def test_legacy_addresses_remain_default_and_shared_bounds_are_unchanged(self):
        constants = (shared.MODEL_BASE, shared.MODEL_LIMIT, shared.TENSOR_BASE, shared.TENSOR_LIMIT)
        models = tiny_models()
        automatic = plan_memory(build_graph(*models, 400))
        explicit = plan_memory(build_graph(*models, 400), layout=LEGACY_LAYOUT)
        self.assertEqual(automatic.layout, LEGACY_LAYOUT)
        self.assertEqual(vars(automatic), vars(explicit))
        self.assertEqual(constants, (shared.MODEL_BASE, shared.MODEL_LIMIT,
                                    shared.TENSOR_BASE, shared.TENSOR_LIMIT))

    def test_model_capacity_replans_each_supported_arena_and_then_fails(self):
        import bigcodec_compile as compiler
        result = object()
        failure = compiler._ModelCapacityError('model arena full', LEGACY_LAYOUT)
        with patch.object(compiler, '_compile_models', side_effect=(failure, result)) as compile_graph:
            self.assertIs(compiler.compile_models('encoder', 'decoder', samples=1000), result)
            self.assertEqual(compile_graph.call_args.kwargs['memory_layout'], EXTENDED_LAYOUT)
        failures = (failure, compiler._ModelCapacityError('model arena full', EXTENDED_LAYOUT), result)
        with patch.object(compiler, '_compile_models', side_effect=failures) as compile_graph:
            self.assertIs(compiler.compile_models('encoder', 'decoder', samples=1000), result)
            self.assertEqual(compile_graph.call_args.kwargs['memory_layout'], LARGE_PROGRAM_LAYOUT)
            self.assertEqual(compile_graph.call_count, 3)
        with patch.object(compiler, '_compile_models', side_effect=compiler._ModelCapacityError(
                'model arena full', LARGE_PROGRAM_LAYOUT)) as compile_graph:
            with self.assertRaisesRegex(ValueError, 'model arena full'):
                compiler.compile_models('encoder', 'decoder', samples=1000)
            self.assertEqual(compile_graph.call_count, 1)
        with patch.object(compiler, '_compile_models', side_effect=ValueError('invalid weights')) as compile_graph:
            with self.assertRaisesRegex(ValueError, 'invalid weights'):
                compiler.compile_models('encoder', 'decoder', samples=1000)
            self.assertEqual(compile_graph.call_count, 1)
        with patch.object(compiler, '_compile_models', side_effect=failure) as compile_graph:
            with self.assertRaisesRegex(ValueError, 'model arena full'):
                compiler.compile_models('encoder', 'decoder', samples=1000, memory_layout=LEGACY_LAYOUT)
            self.assertEqual(compile_graph.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            compiler.compile_models('encoder', 'decoder', samples=1000, memory_layout='unknown')

    def test_large_program_arena_cannot_overlap_packed_audio(self):
        graph = build_graph(*tiny_models(), 524400)
        with self.assertRaisesRegex(ValueError, 'input arena'):
            plan_memory(graph, layout=LARGE_PROGRAM_LAYOUT)

    def test_offline_capture_limit_covers_large_program_and_restores_on_failure(self):
        import bigcodec_compile as compiler
        models = tiny_models()
        for model in models:
            model.training = False
            model.named_parameters = lambda: iter(())
        old_width = compiler.udc.UE_AXI_DATA_WIDTH_BITS
        def fail(*args):
            self.assertEqual(compiler.udc.UE_AXI_DATA_WIDTH_BITS, 256)
            self.assertGreaterEqual(compiler.udc.MAX_DECODER_INSTRUCTIONS, 2 * 1024**3 // 32)
            raise RuntimeError("stop offline capture")
        with patch.object(compiler.udc, "MAX_DECODER_INSTRUCTIONS", 1024):
            with patch.object(compiler, "prepare_operations", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "stop offline capture"):
                    compiler._compile_models(*models, samples=399, memory_layout=LARGE_PROGRAM_LAYOUT)
            self.assertEqual(compiler.udc.MAX_DECODER_INSTRUCTIONS, 1024)
            self.assertEqual(compiler.udc.UE_AXI_DATA_WIDTH_BITS, old_width)

    @unittest.skipUnless(DEFAULT_CHECKPOINT.exists(), 'official checkpoint not downloaded')
    def test_longest_noisy_file_fits_extended_layout_without_live_aliases(self):
        torch.set_num_threads(1)
        encoder, decoder = load_models(remove_weight_norm=True)
        graph = build_graph(encoder, decoder, 379000, conv_precision='bf16')
        with self.assertRaisesRegex(ValueError, 'tensors/workspace'):
            plan_memory(graph, layout=LEGACY_LAYOUT)
        graph = plan_memory(graph)
        self.assertEqual(graph.layout, EXTENDED_LAYOUT)
        self.assertEqual(graph.tensor_end - graph.layout.tensor_base, 679410560)
        self.assertLessEqual(graph.tensor_end, 0x100000000)
        self.assertEqual(graph.output_address, 0xD0000000)
        self.assertEqual(graph.tokens_address, graph.output_address + 379000 * 128)
        for index in range(len(graph.operations)):
            live = [tensor for tensor in graph.tensors.values() if tensor.first <= index <= tensor.last]
            for left_index, left in enumerate(live):
                for right in live[left_index + 1:]:
                    self.assertFalse(left.address < right.address + right.size_bytes
                        and right.address < left.address + left.size_bytes)
        self.assertEqual(layout_for_hardware({key: getattr(graph.layout, key) for key in
            ('model_base', 'model_limit', 'tensor_base', 'tensor_limit')}), EXTENDED_LAYOUT)
        graph = plan_memory(graph, layout=LARGE_PROGRAM_LAYOUT)
        self.assertEqual(graph.tensor_end - graph.layout.tensor_base, 679410560)
        self.assertLessEqual(graph.tensors['input'].address + graph.tensors['input'].size_bytes,
                             graph.layout.model_base)
        self.assertEqual(graph.layout.model_limit - graph.layout.model_base, 1216 * 1024 * 1024)
        self.assertEqual(layout_for_hardware({key: getattr(graph.layout, key) for key in
            ('model_base', 'model_limit', 'tensor_base', 'tensor_limit')}), LARGE_PROGRAM_LAYOUT)

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

    def test_compensated_cell_reserves_residual_and_reaches_lstm_plan(self):
        models = tiny_models()
        models[0].block.insert(1, ResLSTM(64))
        legacy = plan_memory(build_graph(*models, 400))
        compensated = plan_memory(build_graph(*models, 400, lstm_cell_precision="compensated"))
        old = next(op for op in legacy.operations if op.op == "lstm")
        new = next(op for op in compensated.operations if op.op == "lstm")
        self.assertEqual(new.scratch_bytes - old.scratch_bytes, 64 * 2)
        with patch("bigcodec_compile.prepare_lstm") as prepare:
            _prepare_operation(compensated, object(), 0x90000000, 0x90002000, new)
            self.assertTrue(prepare.call_args.kwargs["compensated_cell"])
            _prepare_operation(legacy, object(), 0x90000000, 0x90002000, old)
            self.assertFalse(prepare.call_args.kwargs["compensated_cell"])
        for graph in (legacy, compensated):
            self.assertLessEqual(graph.tensor_end, graph.layout.tensor_limit)
        with self.assertRaisesRegex(ValueError, "cell precision"):
            build_graph(*models, 400, lstm_cell_precision="fp32")

    def test_selected_stack_gets_fused_and_compensated_arithmetic(self):
        models = tiny_models()
        models[0].block.insert(1, ResLSTM(64))
        models[1].model.insert(1, ResLSTM(1))
        for scope in ("encoder", "decoder", "both"):
            graph = plan_memory(build_graph(*models, 400, lstm_cell_precision="products",
                lstm_tanh_precision="compensated", lstm_fused_gates=True, lstm_math_scope=scope))
            for op in (op for op in graph.operations if op.op == "lstm"):
                with patch("bigcodec_compile.prepare_lstm") as prepare:
                    _prepare_operation(graph, object(), 0x90000000, 0x90002000, op)
                    selected = scope == "both" or op.name.startswith(scope + ".")
                    for flag in ("compensated_cell", "compensated_tanh", "fused_projection"):
                        self.assertEqual(prepare.call_args.kwargs[flag], selected)
                    self.assertFalse(prepare.call_args.kwargs["preserve_cell_residual"])
        for kwargs in (dict(lstm_tanh_precision="fp32"), dict(lstm_math_scope="neither"),
                       dict(lstm_fused_gates=1)):
            with self.assertRaises(ValueError):
                build_graph(*models, 400, **kwargs)

    def test_paired_sigmoid_is_limited_to_corrected_bf16_decoder(self):
        graph = SimpleNamespace(lstm_math_scope="decoder", lstm_precision="bf16",
            lstm_cell_precision="compensated", lstm_tanh_precision="compensated",
            lstm_fused_gates=True, tensors={"source": SimpleNamespace(shape=(317, 1536))})
        decoder = SimpleNamespace(name="decoder.model.1", inputs=("source",))
        encoder = SimpleNamespace(name="encoder.block.6", inputs=("source",))
        self.assertTrue(_lstm_numerics(graph, decoder)["paired_sigmoid"])
        self.assertFalse(_lstm_numerics(graph, encoder)["paired_sigmoid"])
        for scope in ("both", "decoder"):
            for precision in ("bf16", "encoder-if8"):
                variant = SimpleNamespace(**(vars(graph) | dict(lstm_math_scope=scope, lstm_precision=precision)))
                self.assertTrue(_lstm_numerics(variant, decoder)["paired_sigmoid"])
                self.assertFalse(_lstm_numerics(variant, encoder)["paired_sigmoid"])
        for override in (dict(lstm_math_scope="encoder"), dict(lstm_precision="if8"),
                dict(lstm_cell_precision="products"), dict(lstm_cell_precision="bf16"),
                dict(lstm_tanh_precision="bf16"), dict(lstm_fused_gates=False),
                dict(tensors={"source": SimpleNamespace(shape=(317, 64))})):
            with self.subTest(override=override):
                variant = SimpleNamespace(**(vars(graph) | override))
                self.assertFalse(_lstm_numerics(variant, decoder)["paired_sigmoid"])

    def test_quantizer_precision_flags_reach_packing_without_changing_memory_shape(self):
        plain = plan_memory(build_graph(*tiny_models(), 400))
        precise = plan_memory(build_graph(*tiny_models(), 400, center_quantizer_scores=True,
                                         compensated_codebook=True))
        self.assertEqual(plain.scratch_bytes, precise.scratch_bytes)
        self.assertEqual(plain.output_bytes, precise.output_bytes)
        for graph, enabled in ((plain, False), (precise, True)):
            op = next(op for op in graph.operations if op.op == "quantizer")
            with patch("bigcodec_compile.prepare_quantizer") as prepare:
                _prepare_operation(graph, object(), 0x90000000, 0x90002000, op)
                self.assertEqual(prepare.call_args.kwargs["center_scores"], enabled)
                self.assertEqual(prepare.call_args.kwargs["compensated_codebook"], enabled)
        for flag in ("center_quantizer_scores", "compensated_codebook"):
            with self.assertRaises(ValueError):
                build_graph(*tiny_models(), 400, **{flag: 1})

    def test_filter_scope_selects_activation_plans_without_changing_buffers(self):
        models = projection_models()
        models[0].block[1] = Activation1d(activation=SnakeBeta(2, alpha_logscale=True))
        models[1].model.insert(1, Activation1d(activation=SnakeBeta(1, alpha_logscale=True)))
        default = plan_memory(build_graph(*models, 400, conv_precision='bf16'))
        explicit = plan_memory(build_graph(*models, 400, conv_precision='bf16',
            filter_accumulation='sorted', filter_math_scope='encoder'))
        self.assertEqual(vars(default), vars(explicit))
        cases = (
            ('serial', 'both', {'encoder': 'serial', 'decoder': 'serial'}),
            ('sorted', 'encoder', {'encoder': 'sorted', 'decoder': 'serial'}),
            ('sorted', 'decoder', {'encoder': 'serial', 'decoder': 'sorted'}),
            ('sorted', 'both', {'encoder': 'sorted', 'decoder': 'sorted'}),
        )
        for accumulation, scope, expected in cases:
            with self.subTest(accumulation=accumulation, scope=scope):
                graph = plan_memory(build_graph(*models, 400, conv_precision='bf16',
                    filter_accumulation=accumulation, filter_math_scope=scope,
                    lstm_math_scope='decoder' if scope == 'encoder' else 'encoder'))
                self.assertEqual(graph.tensors, default.tensors)
                self.assertEqual((graph.scratch_bytes, graph.tensor_end, graph.output_bytes),
                                 (default.scratch_bytes, default.tensor_end, default.output_bytes))
                observed = {}
                for operation in (op for op in graph.operations if op.op == 'activation'):
                    plan = object()
                    with patch('bigcodec_compile.prepare_activation', return_value=plan) as prepare:
                        _prepare_operation(graph, object(), 0x90000000, 0x90002000, operation)
                    self.assertIs(operation.plan, plan)
                    stack = operation.name.split('.', 1)[0]
                    observed[stack] = prepare.call_args.kwargs['filter_accumulation']
                    self.assertEqual(prepare.call_args.kwargs['input_shape'],
                                     graph.tensors[operation.inputs[0]].shape)
                self.assertEqual(observed, expected)

    def test_filter_configuration_rejects_unknown_orders_and_scopes(self):
        models = projection_models()
        for value in ('fp32', 'pairwise', '', None, True):
            with self.subTest(order=value), self.assertRaisesRegex(ValueError, 'Filter accumulation'):
                build_graph(*models, 400, filter_accumulation=value)
        for value in ('neither', 'encoder.block', '', None, True):
            with self.subTest(scope=value), self.assertRaisesRegex(ValueError, 'Filter math scope'):
                build_graph(*models, 400, filter_math_scope=value)

    def test_filter_configuration_survives_compile_arena_retry(self):
        import bigcodec_compile as compiler
        failure = compiler._ModelCapacityError('model arena full', LEGACY_LAYOUT)
        result = object()
        with patch.object(compiler, '_compile_models', side_effect=(failure, result)) as compile_graph:
            self.assertIs(compiler.compile_models(*projection_models(), samples=399,
                filter_accumulation='sorted', filter_math_scope='encoder'), result)
            self.assertEqual(compile_graph.call_count, 2)
            for call in compile_graph.call_args_list:
                self.assertEqual(call.kwargs['filter_accumulation'], 'sorted')
                self.assertEqual(call.kwargs['filter_math_scope'], 'encoder')

    def test_matrix_filter_scope_reserves_exact_workspace_and_reaches_packer(self):
        from bigcodec_activation import activation_scratch_bytes
        encoder, decoder = projection_models()
        encoder.block.insert(1, Activation1d(SnakeBeta(2, alpha_logscale=True)))
        decoder.model.insert(0, Activation1d(SnakeBeta(1024, alpha_logscale=True)))
        for scope in ('encoder', 'decoder', 'both'):
            for stage in ('up', 'down', 'both'):
                graph = plan_memory(build_graph(encoder, decoder, 400, conv_precision='bf16',
                    filter_accumulation='matrix', filter_math_scope=scope, filter_stage=stage))
                for operation in (op for op in graph.operations if op.op == 'activation'):
                    stack = operation.name.split('.', 1)[0]
                    mode = 'matrix' if scope in (stack, 'both') else 'serial'
                    shape = graph.tensors[operation.inputs[0]].shape
                    self.assertEqual(operation.scratch_bytes, activation_scratch_bytes(shape,
                        filter_accumulation=mode, filter_stage=stage))
                    self.assertLessEqual(operation.scratch_bytes, graph.scratch_bytes)
                    with patch('bigcodec_compile.prepare_activation') as prepare:
                        _prepare_operation(graph, object(), 0x90000000, 0x90002000, operation)
                    self.assertEqual(prepare.call_args.kwargs['filter_accumulation'], mode)
                    self.assertEqual(prepare.call_args.kwargs['filter_stage'], stage)
                    self.assertEqual(prepare.call_args.kwargs['workspace_bytes'], graph.scratch_bytes)
                for tensor in graph.tensors.values():
                    self.assertLessEqual(tensor.address + tensor.size_bytes, graph.scratch_address)

    def test_matrix_stage_validation_and_compile_retry(self):
        import bigcodec_compile as compiler
        for stage in ('neither', '', None, True):
            with self.assertRaisesRegex(ValueError, 'Filter stage'):
                build_graph(*projection_models(), 400, filter_stage=stage)
        failure = compiler._ModelCapacityError('model arena full', LEGACY_LAYOUT)
        result = object()
        with patch.object(compiler, '_compile_models', side_effect=(failure, result)) as compile_graph:
            self.assertIs(compiler.compile_models(*projection_models(), samples=399,
                filter_accumulation='matrix', filter_math_scope='encoder', filter_stage='down'), result)
            for call in compile_graph.call_args_list:
                self.assertEqual(call.kwargs['filter_accumulation'], 'matrix')
                self.assertEqual(call.kwargs['filter_math_scope'], 'encoder')
                self.assertEqual(call.kwargs['filter_stage'], 'down')

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
