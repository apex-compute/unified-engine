"""Pinned BigCodec reference model and full-utterance audio/token helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODEL_DIR / "bigcodec_bin" / "bigcodec.pt"
SAMPLE_RATE = 16000
HOP_LENGTH = 200
CODEBOOK_SIZE = 8192
CHECKPOINT_SHA256 = "1fba3806e87cc01c1a65bea22fa1becefbbf46881e4219593c4d9f3cf56206b9"
TOKEN_FORMAT = "bigcodec-tokens-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(checkpoint: str | Path = DEFAULT_CHECKPOINT,
                remove_weight_norm: bool = False):
    """Load the official encoder/decoder on CPU, with strict state validation."""
    import torch
    try:
        from .bigcodec_vq.codec_encoder import CodecEncoder
        from .bigcodec_vq.codec_decoder import CodecDecoder
    except ImportError:
        from bigcodec_vq.codec_encoder import CodecEncoder
        from bigcodec_vq.codec_decoder import CodecDecoder

    checkpoint = Path(checkpoint)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {actual_sha}; expected {CHECKPOINT_SHA256}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder, decoder = CodecEncoder(), CodecDecoder()
    encoder.load_state_dict(state["CodecEnc"], strict=True)
    decoder.load_state_dict(state["generator"], strict=True)
    encoder.eval()
    decoder.eval()
    if remove_weight_norm:
        encoder.remove_weight_norm()
        decoder.remove_weight_norm()
    return encoder, decoder


def pad_audio(audio: np.ndarray) -> np.ndarray:
    """Match official inference.py: always append 1..200 zero samples."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be a nonempty, finite mono vector")
    return np.pad(audio, (0, HOP_LENGTH - audio.size % HOP_LENGTH))


def read_audio(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read float audio, average channels, and use upstream's soxr_hq resampler."""
    import soundfile as sf
    import librosa

    source, rate = sf.read(path, dtype="float32", always_2d=True)
    if source.shape[0] == 0 or not np.isfinite(source).all():
        raise ValueError("Input audio must be nonempty and finite")
    audio = source.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE,
                                 res_type="soxr_hq")
    metadata = {
        "format": TOKEN_FORMAT,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "source_rate": int(rate),
        "source_channels": int(source.shape[1]),
        "source_samples": int(source.shape[0]),
        "native_samples": int(audio.size),
        "padded_samples": int(audio.size + HOP_LENGTH - audio.size % HOP_LENGTH),
        "source_sha256": sha256_file(path),
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    return np.asarray(audio, dtype=np.float32), metadata


def restore_audio(audio: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Trim codec padding, then restore the input WAV rate and exact length."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    native_samples = int(metadata["native_samples"])
    if audio.size < native_samples or not np.isfinite(audio).all():
        raise ValueError("Decoded audio is too short or contains nonfinite samples")
    audio = audio[:native_samples]
    source_rate = int(metadata["source_rate"])
    if source_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=source_rate,
                                 res_type="soxr_hq")
    size = int(metadata["source_samples"])
    return np.pad(audio, (0, max(0, size - audio.size)))[:size]


def validate_tokens(tokens: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    tokens = np.asarray(tokens)
    if tokens.ndim != 1 or tokens.size == 0 or tokens.dtype.kind not in "iu":
        raise ValueError("Tokens must be a nonempty one-dimensional integer array")
    if tokens.min() < 0 or tokens.max() >= CODEBOOK_SIZE:
        raise ValueError("Token index outside the 8192-entry codebook")
    if metadata.get("format") != TOKEN_FORMAT:
        raise ValueError("Unsupported token file format")
    if metadata.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Token checkpoint does not match the official model")
    if metadata.get("sample_rate") != SAMPLE_RATE or metadata.get("hop_length") != HOP_LENGTH:
        raise ValueError("Tokens require the 16 kHz, 200-sample BigCodec configuration")
    for key in ("source_rate", "source_samples", "source_channels", "native_samples", "padded_samples"):
        if not isinstance(metadata.get(key), int) or metadata[key] <= 0:
            raise ValueError(f"Invalid token metadata: {key}")
    n = metadata["native_samples"]
    if metadata["padded_samples"] != n + HOP_LENGTH - n % HOP_LENGTH:
        raise ValueError("Token padding does not match official inference")
    if tokens.size * HOP_LENGTH != metadata["padded_samples"]:
        raise ValueError("Token count does not match audio duration")
    expected_native = (metadata["source_samples"] * SAMPLE_RATE + metadata["source_rate"] - 1) // metadata["source_rate"]
    if n != expected_native:
        raise ValueError("Native duration does not match source duration")
    return tokens.astype(np.int64)


def save_tokens(path: str | Path, tokens: np.ndarray, metadata: dict[str, Any]) -> None:
    tokens = validate_tokens(tokens, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, tokens=tokens.astype(np.uint16),
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_tokens(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        tokens = validate_tokens(archive["tokens"], metadata)
    return tokens, metadata


def encode_audio(encoder, decoder, audio: np.ndarray):
    """Return token IDs and official quantizer output, preserving upstream math."""
    import torch
    with torch.inference_mode():
        features = encoder(torch.from_numpy(pad_audio(audio)).reshape(1, 1, -1))
        quantized, tokens, _ = decoder(features, vq=True)
    return tokens[0, 0].cpu().numpy(), quantized


def decode_tokens(decoder, tokens: np.ndarray) -> np.ndarray:
    import torch
    tokens = np.asarray(tokens)
    if tokens.ndim != 1 or tokens.size == 0 or tokens.dtype.kind not in "iu" or tokens.min() < 0 or tokens.max() >= CODEBOOK_SIZE:
        raise ValueError("Invalid BigCodec token indices")
    with torch.inference_mode():
        indices = torch.from_numpy(tokens.astype(np.int64)).reshape(1, -1, 1)
        quantized = decoder.vq2emb(indices).transpose(1, 2)
        return decoder(quantized, vq=False).reshape(-1).cpu().numpy()
