"""CPU-only diagnostic used for p257_018; does not access the FPGA or alter models.

Run from the repository root using my_torch_env Python with
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
This is the in-memory counterfactual script used for the accompanying evidence.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, "models/dpdfnet")
from dpdfnet_audio import read_audio, synthesize_audio
from dpdfnet_common import initial_state

evidence = json.loads(Path(__file__).with_suffix(".json").read_text())
for source_name in ("noisy_wav", "cpu_reference_wav", "fpga_wav", "onnx_model",
                    "audio_wrapper"):
    source = evidence["sources"][source_name]
    path = Path(source["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; see the accompanying README prerequisites")
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise RuntimeError(f"Source SHA256 differs from the archived evidence: {path}")

base = Path("models/dpdfnet/dpdfnet_bin/noisy_eval_20260913/results/p257_018")
audio = read_audio(base / "noisy.wav")
ref, _ = sf.read(base / "cpu.wav", dtype="float32")
hw, _ = sf.read(base / "fpga.wav", dtype="float32")
model_original = onnx.load("models/dpdfnet/dpdfnet_bin/dpdfnet2.onnx")
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1

for mode in ("original", "epsilon_zero", "state_bf16", "frontend_state_bf16",
             "constants_bf16_state_bf16"):
    model = copy.deepcopy(model_original)
    if mode == "epsilon_zero":
        for node in model.graph.node:
            if node.op_type == "LayerNormalization":
                for attribute in node.attribute:
                    if attribute.name == "epsilon":
                        attribute.f = 0
    if mode == "constants_bf16_state_bf16":
        for index, initializer in enumerate(model.graph.initializer):
            value = onnx.numpy_helper.to_array(initializer)
            if value.dtype == np.float32:
                quantized = torch.from_numpy(value.copy()).bfloat16().float().numpy()
                model.graph.initializer[index].CopyFrom(
                    onnx.numpy_helper.from_array(quantized, initializer.name))
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    state = initial_state(session.get_modelmeta().custom_metadata_map)
    outputs = []
    for frame in audio.frames:
        if mode in ("state_bf16", "constants_bf16_state_bf16"):
            state = torch.from_numpy(state).bfloat16().float().numpy()
        elif mode == "frontend_state_bf16":
            state[:128] = torch.from_numpy(state[:128]).bfloat16().float().numpy()
        spectrum, state = session.run(["spec_e", "state_out"], {"spec": frame, "state_in": state})
        outputs.append(spectrum)
    value = synthesize_audio(np.stack(outputs), audio)
    if mode == "original" and not np.array_equal(value.view(np.uint32), ref.view(np.uint32)):
        raise RuntimeError("Original CPU rerun differs from the archived reference; comparison aborted")
    delta = value.astype(np.float64) - ref
    hardware_delta = hw.astype(np.float64) - ref
    first, stop = int(0.593 * 16000), int(0.793 * 16000)
    print(json.dumps({
        "mode": mode,
        "bit_equal_cpu": bool(np.array_equal(value, ref)),
        "relative_l2_vs_cpu": float(np.linalg.norm(delta) / np.linalg.norm(ref.astype(np.float64))),
        "relative_l2_vs_fpga": float(np.linalg.norm(value - hw) / np.linalg.norm(hw)),
        "onset_error_energy_vs_fpga_error_energy": float(
            np.dot(delta[first:stop], delta[first:stop])
            / np.dot(hardware_delta[first:stop], hardware_delta[first:stop])),
        "error_alignment_with_fpga": float(
            np.dot(delta, hardware_delta) / (np.linalg.norm(delta) * np.linalg.norm(hardware_delta)))
            if np.any(delta) else None,
        "onset_gain_vs_cpu": float(
            np.dot(value[first:stop], ref[first:stop]) / np.dot(ref[first:stop], ref[first:stop])),
    }), flush=True)
