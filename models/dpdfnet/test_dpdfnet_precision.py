"""Encoded weight selection and named CLI guards without hardware access."""

import contextlib
import hashlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parent / "dpdfnet8khz", HERE.parents[1]):
    sys.path.insert(0, str(path))

import user_dma_core as udc
from dpdfnet_precision import validate_weight_precision
import dpdfnet_run_from_bin as runner16
import dpdfnet8khz_run_from_bin as runner8
import dpdfnet_audio as audio16
import dpdfnet8khz_audio as audio8


def descriptor(mode=0, data_type=0, *, kind=udc.INSTRUCTION_UE_OP):
    # ISA positions, encoded directly into the complete little-endian
    # descriptor rather than using the precision helper's NumPy word view.
    value = (int(kind) << 8) | (int(mode) << 172) | (int(data_type) << 181)
    return value.to_bytes(32, "little")


def payload(instructions, *, metadata="BF16"):
    """Supply the image/program subset returned by a mocked artifact validator."""
    # Parameter bytes can resemble instructions and must never affect the
    # selected program's precision. The active program ends in one HALT.
    prefix = descriptor(udc.UE_MODE.CONV2D, udc.TYPE.IF4) + descriptor(udc.UE_MODE.QUANTIZE)
    program = b"".join(instructions) + descriptor(kind=udc.INSTRUCTION_HALT)
    return {"hardware": {
        "model_image": torch.tensor(list(prefix + program), dtype=torch.uint8),
        "program_offset": len(prefix), "program_size": len(program),
        "precision": metadata, "dense_convolution_precision": metadata,
    }}


def model_payload(precision, *, kind=udc.INSTRUCTION_UE_OP, metadata="stale label"):
    instructions = [descriptor(udc.UE_MODE.BF16_DOT_PRODUCT, kind=kind)]
    for data_type in {"bf16": (), "if8": (udc.TYPE.IF8,),
                      "if4_if8": (udc.TYPE.IF4, udc.TYPE.IF8)}[precision]:
        instructions.append(descriptor(udc.UE_MODE.CONV2D, data_type, kind=kind))
    return payload(instructions, metadata=metadata)


class WeightPrecisionTest(unittest.TestCase):
    def test_encoded_precision_overrides_stale_or_missing_metadata(self):
        for kind in (udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_UE_PBI):
            for expected, label in (("bf16", "BF16"), ("if8", "IF8"), ("if4_if8", "IF4/IF8")):
                for metadata in ("BF16", "BF16/IF8-INT", "IF4/IF8", None):
                    with self.subTest(kind=kind, expected=expected, metadata=metadata):
                        artifact = model_payload(expected, kind=kind, metadata=metadata)
                        if metadata is None:
                            artifact["hardware"].pop("precision")
                            artifact["hardware"].pop("dense_convolution_precision")
                        self.assertEqual(validate_weight_precision(artifact), label)
                        self.assertEqual(validate_weight_precision(artifact, expected), label)

    def test_only_ue_instruction_modes_affect_precision(self):
        noise = []
        for kind in (udc.INSTRUCTION_CONFIG, udc.INSTRUCTION_PBI_SET, udc.INSTRUCTION_NOP):
            for mode in (udc.UE_MODE.CONV2D, udc.UE_MODE.DOT_PRODUCT,
                         udc.UE_MODE.QUANTIZE, udc.UE_MODE.DEQUANTIZE,
                         udc.UE_MODE.BF16_DOT_PRODUCT):
                noise.append(descriptor(mode, udc.TYPE.IF4, kind=kind))
        artifact = payload([descriptor(udc.UE_MODE.BF16_DOT_PRODUCT), *noise])
        self.assertEqual(validate_weight_precision(artifact, "bf16"), "BF16")
        # Unused dtype bits on a real elementwise instruction are harmless.
        artifact = payload([descriptor(udc.UE_MODE.BF16_DOT_PRODUCT),
                            descriptor(udc.UE_MODE.ELTWISE_ADD, udc.TYPE.IF4)])
        self.assertEqual(validate_weight_precision(artifact), "BF16")
        with self.assertRaisesRegex(ValueError, "BF16 matrix"):
            validate_weight_precision(payload([
                descriptor(udc.UE_MODE.BF16_DOT_PRODUCT, kind=udc.INSTRUCTION_NOP),
                descriptor(udc.UE_MODE.CONV2D, udc.TYPE.IF8)]))

    def test_rejects_unsupported_quantized_arithmetic_and_convolution_types(self):
        bf16 = descriptor(udc.UE_MODE.BF16_DOT_PRODUCT)
        for kind in (udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_UE_PBI):
            for mode in (udc.UE_MODE.DOT_PRODUCT, udc.UE_MODE.QUANTIZE, udc.UE_MODE.DEQUANTIZE):
                with self.subTest(kind=kind, mode=mode):
                    artifact = payload([bf16, descriptor(mode, udc.TYPE.IF8, kind=kind)])
                    with self.assertRaisesRegex(ValueError, "outside dense CONV"):
                        validate_weight_precision(artifact)
            for types in ((0,), (udc.TYPE.TQ4,), (udc.TYPE.IF4,), (udc.TYPE.IF8, udc.TYPE.TQ4)):
                with self.subTest(kind=kind, types=types):
                    artifact = payload([bf16, *[descriptor(udc.UE_MODE.CONV2D, value, kind=kind)
                                               for value in types]])
                    with self.assertRaisesRegex(ValueError, "Unsupported dense convolution"):
                        validate_weight_precision(artifact)

    def test_rejects_missing_or_invalid_bf16_arithmetic(self):
        for kind in (udc.INSTRUCTION_UE_OP, udc.INSTRUCTION_UE_PBI):
            for data_type in (udc.TYPE.IF8, udc.TYPE.TQ4, udc.TYPE.IF4):
                with self.subTest(kind=kind, data_type=data_type):
                    artifact = payload([descriptor(udc.UE_MODE.BF16_DOT_PRODUCT, data_type, kind=kind)])
                    with self.assertRaisesRegex(ValueError, "BF16 matrix"):
                        validate_weight_precision(artifact)
        for instructions in ([], [descriptor(udc.UE_MODE.CONV2D, udc.TYPE.IF8)]):
            with self.assertRaisesRegex(ValueError, "BF16 matrix"):
                validate_weight_precision(payload(instructions))

    def test_rejects_invalid_program_bounds_and_requested_precision(self):
        for offset, size in ((-32, 64), (64, 0), (64, 33), (128, 64)):
            with self.subTest(offset=offset, size=size):
                artifact = model_payload("bf16")
                artifact["hardware"].update(program_offset=offset, program_size=size)
                with self.assertRaisesRegex(ValueError, "program bounds"):
                    validate_weight_precision(artifact)
        for expected in ("if4", "fp32", "BF16"):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "Unsupported requested"):
                validate_weight_precision(model_payload("bf16"), expected)

    def test_mismatch_identifies_both_requested_and_encoded_weights(self):
        labels = {"bf16": "BF16", "if8": "IF8", "if4_if8": "IF4/IF8"}
        for actual, actual_label in labels.items():
            for expected, expected_label in labels.items():
                if actual == expected:
                    continue
                with self.subTest(actual=actual, expected=expected):
                    with self.assertRaises(ValueError) as raised:
                        validate_weight_precision(model_payload(actual, metadata=expected_label), expected)
                    self.assertIn(f"requires {expected_label}", str(raised.exception))
                    self.assertIn(f"contains {actual_label}", str(raised.exception))
                    self.assertIn("matching --bin", str(raised.exception))


class NamedRunnerTest(unittest.TestCase):
    def test_all_named_entrypoints_forward_arguments_bin_and_precision(self):
        cases = (
            ("dpdfnet8khz_bf16_weights_test", "dpdfnet8khz", "dpdfnet8khz_bin/dpdfnet2_8khz-bf16-andromeda.bin", "bf16"),
            ("dpdfnet8khz_if8_weights_test", "dpdfnet8khz", "dpdfnet8khz_bin/dpdfnet2_8khz-andromeda.bin", "if8"),
            ("dpdfnet16khz_bf16_weights_test", "dpdfnet", "dpdfnet_bin/dpdfnet2-bf16-andromeda.bin", "bf16"),
            ("dpdfnet16khz_if4_if8_weights_test", "dpdfnet", "dpdfnet_bin/dpdfnet2-andromeda.bin", "if4_if8"),
        )
        for name, folder, filename, precision in cases:
            module = importlib.import_module(name)
            for argv in (None, ["--input", "noisy.wav", "--output", "enhanced.wav"]):
                with self.subTest(name=name, argv=argv), mock.patch.object(module, "run_main") as run:
                    module.main(argv)
                    run.assert_called_once_with(argv, default_bin=HERE.parent / folder / filename,
                                                expected_precision=precision)

    def test_wrong_bin_fails_before_input_read_or_hardware_setup_for_both_rates(self):
        cases = ((runner8, audio8, "bf16", "if8"), (runner8, audio8, "if8", "bf16"),
                 (runner16, audio16, "bf16", "if4_if8"), (runner16, audio16, "if4_if8", "bf16"))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            selected = folder / "wrong.bin"
            selected.write_bytes(b"synthetic artifact validated by the mocked loader")
            for runner, audio, expected, actual in cases:
                for extension in (".wav", ".npy"):
                    with self.subTest(runner=runner.__name__, expected=expected, input=extension):
                        source = folder / ("unread-input" + extension)
                        source.write_bytes(b"intentionally invalid audio/spectra: must not be read")
                        output = folder / ("enhanced" + extension)
                        original = selected.read_bytes(), source.read_bytes()
                        stderr = io.StringIO()
                        with contextlib.ExitStack() as stack:
                            loaded = stack.enter_context(mock.patch.object(
                                runner, "load_artifact", return_value=model_payload(actual, metadata=expected)))
                            digest = stack.enter_context(mock.patch.object(
                                runner, "sha256", return_value=hashlib.sha256(original[0]).hexdigest()))
                            untouched = [stack.enter_context(mock.patch.object(owner, name,
                                side_effect=AssertionError(f"{name} must not run before precision rejection")))
                                for owner, name in ((audio, "read_audio"), (runner, "_load_frames"),
                                                    (runner, "configure_hardware_runtime"),
                                                    (runner, "StreamingEngine"), (runner, "WholeGraphBackend"))]
                            stack.enter_context(contextlib.redirect_stderr(stderr))
                            with self.assertRaises(SystemExit) as raised:
                                runner.main(["--bin", str(selected), "--input", str(source),
                                             "--output", str(output)],
                                            default_bin=folder / "named-default.bin", expected_precision=expected)
                            self.assertEqual(raised.exception.code, 2)
                        loaded.assert_called_once_with(selected)
                        digest.assert_called_once_with(selected)
                        for action in untouched:
                            action.assert_not_called()
                        self.assertIn("requires", stderr.getvalue())
                        self.assertIn("matching --bin", stderr.getvalue())
                        self.assertFalse(output.exists())
                        self.assertFalse(output.with_suffix(".metrics.json").exists())
                        self.assertEqual((selected.read_bytes(), source.read_bytes()), original)


if __name__ == "__main__":
    unittest.main()
