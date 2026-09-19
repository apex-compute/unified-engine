#!/usr/bin/env python3
"""Speech output for Qwen2.5-Omni: Talker + Token2Wav, on the HOST.

WHY THIS IS HOST-SIDE, FOR NOW. The speech path is three networks, and only two
of them are things this accelerator can express today:

  Talker    1351 M params   a Qwen2-style GQA decoder over CODEC tokens, taking
                            the Thinker's hidden states as its conditioning.
                            Structurally identical to the LM already on the
                            FPGA -- q/k/v/o with biases, gate/up/down MLP,
                            RMSNorm -- so it is a port, not new kernels.
  DiT        334 M params   22 transformer blocks with AdaLN modulation plus a
                            small conv1d speaker encoder. Also expressible:
                            attention, matmuls, eltwise, and conv1d via the
                            im2col the audio encoder already does.
  BigVGAN    115 M params   the vocoder, and the one that does NOT fit. It is
                            built on the SNAKE activation, x + sin^2(ax)/a with
                            a learned per-channel a. LALU_MODE offers BYPASS,
                            ACT (a+x)*sigmoid(-bx), RECIP, RSQRT, CLAMP and LOG
                            -- there is no trig unit. RoPE gets its trig from
                            host-precomputed tables, which does not help here
                            because the argument is data-dependent, and there is
                            no gather-by-value to index a table with. A
                            polynomial approximation would need range reduction
                            (mod 2*pi, hence floor), which is also absent.

So BigVGAN stays on the host permanently unless the ISA grows, and the Talker
and DiT move to the FPGA in that order. This module is the reference the FPGA
ports get verified against, and the thing that makes speech work today.

THE THINKER->TALKER INTERFACE is the part the accelerator has to satisfy, and
it is narrow: per token, the Talker wants the Thinker's FINAL HIDDEN STATE plus
that token's INPUT EMBEDDING, summed, both 3584 wide.

Three of the four tensors are free today:

  step_embeds     the embedding lookup already happens host-side
  prefill_embeds  likewise
  step_hidden     LM_OUT_NORM holds exactly this after each decode step; it is
                  a 3584-element readback per token

The fourth is not. PREFILL_HIDDEN wants the final norm applied to EVERY prompt
row, and prefill does not compute it: LM_OUT_NORM is allocated as a SINGLE row
(`alloc(H, "lm.out_norm")`) because only the last position's logits are needed
to pick the first token. Supplying it costs one extra
`rms_norm_core_dram(M=seq_len, N=H)` over the final layer output into a [T, H]
buffer, plus a T*3584*2 byte readback -- about 13 MiB at a 1899-token prompt.
Cheap, but it is real work rather than a readback, and it is the one thing
standing between this module and speech driven by the accelerator.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

SPEAKERS = ("Chelsie", "Ethan")
DEFAULT_SPEAKER = "Chelsie"
SAMPLE_RATE = 24000


def _load_submodule_state(model_dir: str, prefix: str) -> dict[str, torch.Tensor]:
    """Read one top-level module's tensors out of the sharded checkpoint.

    Only the shards that actually carry the prefix are opened, so pulling the
    Talker does not fault in the 7B Thinker.
    """
    from safetensors import safe_open

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    wanted = {n: s for n, s in weight_map.items() if n.startswith(prefix + ".")}
    if not wanted:
        raise KeyError(f"no tensors under {prefix!r} in {model_dir}")
    missing = sorted({s for s in wanted.values()
                      if not os.path.exists(os.path.join(model_dir, s))})
    if missing:
        raise FileNotFoundError(
            f"{prefix}: checkpoint shard(s) {missing} are not present. The "
            f"speech path needs the full checkpoint; the Thinker-only weight "
            f"conversion skips them.")
    # DERIVED BUFFERS ARE NOT WEIGHTS. RoPE inverse frequencies are recomputed
    # from the config at construction, and this transformers version does not
    # register them as persistent, so the checkpoint's copy is an unexpected
    # key. Drop it by name rather than relaxing to strict=False, which would
    # also swallow a genuinely missing projection.
    derived = (".inv_freq",)
    state: dict[str, torch.Tensor] = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)
    for shard, names in by_shard.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
            for name in names:
                if name.endswith(derived):
                    continue
                state[name[len(prefix) + 1:]] = f.get_tensor(name)
    return state


class HostSpeech:
    """Talker + Token2Wav, loaded once and reused across requests."""

    def __init__(self, model_dir: str, speaker: str = DEFAULT_SPEAKER,
                 dtype: torch.dtype = torch.float32):
        from transformers import (Qwen2_5OmniConfig,
                                  Qwen2_5OmniTalkerForConditionalGeneration,
                                  Qwen2_5OmniToken2WavModel)

        if speaker not in SPEAKERS:
            raise ValueError(f"speaker must be one of {SPEAKERS}, got {speaker!r}")
        self.model_dir = model_dir
        self.speaker = speaker
        cfg = Qwen2_5OmniConfig.from_pretrained(model_dir)

        self.talker = Qwen2_5OmniTalkerForConditionalGeneration(cfg.talker_config)
        self.talker.load_state_dict(_load_submodule_state(model_dir, "talker"),
                                    strict=True)
        self.talker.to(dtype=dtype).eval()

        self.token2wav = Qwen2_5OmniToken2WavModel(cfg.token2wav_config)
        self.token2wav.load_state_dict(_load_submodule_state(model_dir, "token2wav"),
                                       strict=True)
        # The vocoder is numerically touchy; HF runs it in float32.
        self.token2wav.to(dtype=torch.float32).eval()

        spk = torch.load(os.path.join(model_dir, "spk_dict.pt"),
                         map_location="cpu", weights_only=False)
        self.speaker_params = spk[speaker]

    @property
    def codec_tokens(self) -> dict[str, int]:
        t = self.talker
        return {"mask": t.codec_mask_token, "pad": t.codec_pad_token,
                "bos": t.codec_bos_token, "text_eos": t.text_eos_token,
                "text_pad": t.text_pad_token}

    @torch.no_grad()
    def speak(self, *, input_ids: torch.Tensor, prefill_hidden: torch.Tensor,
              prefill_embeds: torch.Tensor, step_hidden: torch.Tensor,
              step_embeds: torch.Tensor, embed_lookup,
              max_new_tokens: int = 4096, do_sample: bool = True,
              top_k: int = 40, top_p: float = 0.8, temperature: float = 0.9,
              repetition_penalty: float = 1.05) -> torch.Tensor:
        """Thinker state -> codec tokens -> waveform.

        The four tensors are everything the accelerator has to export:

          prefill_hidden [1, T, 3584]   final hidden over the prompt
          prefill_embeds [1, T, 3584]   input embeddings of the prompt
          step_hidden    [1, G, 3584]   final hidden, one row per generated token
          step_embeds    [1, G, 3584]   input embedding of each generated token

        The Talker conditions on their SUM, not on either alone, and it reads
        the reply shifted by one -- it is predicting speech for the text the
        Thinker is about to say, so position g of its conditioning carries
        token g+1, with the text EOS and PAD embeddings closing the sequence.
        """
        talker = self.talker
        dev, dt = prefill_hidden.device, self.talker.dtype

        bos = torch.tensor([[self.speaker_params["bos_token"]]], dtype=torch.long,
                           device=dev) if "bos_token" in self.speaker_params else None
        if bos is None:
            raise KeyError("speaker entry has no bos_token; spk_dict.pt is not the "
                           "one this checkpoint expects")

        # Text stream: the prompt, then the speaker's BOS, then the reply.
        talker_input_text_ids = torch.cat([input_ids, bos], dim=1)
        # Codec stream: the prompt is masked (there is no speech for it yet),
        # then pad, then the codec BOS the model actually starts decoding from.
        talker_input_ids = torch.cat([
            torch.full_like(input_ids, fill_value=talker.codec_mask_token),
            torch.tensor([[talker.codec_pad_token]], dtype=torch.long, device=dev),
            torch.tensor([[talker.codec_bos_token]], dtype=torch.long, device=dev),
        ], dim=1)

        reply = (step_hidden + step_embeds).to(dt)
        inputs_embeds = (prefill_hidden + prefill_embeds).to(dt)
        inputs_embeds = torch.cat([
            inputs_embeds,
            embed_lookup(bos).to(dt),
            reply[:, :1, :],
        ], dim=1)
        # Shift by one and close with EOS/PAD, mirroring the reference.
        thinker_reply_part = torch.cat([
            reply[:, 1:, :],
            embed_lookup(torch.tensor([[talker.text_eos_token]], dtype=torch.long,
                                      device=dev)).to(dt),
            embed_lookup(torch.tensor([[talker.text_pad_token]], dtype=torch.long,
                                      device=dev)).to(dt),
        ], dim=1)

        codes = talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=inputs_embeds,
            suppress_tokens=[talker.codec_bos_token],
            max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
            top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=[8292, 8294],
        )
        codes = codes[:, talker_input_ids.shape[1]:-1]
        return self.synthesize(codes)

    @torch.no_grad()
    def generate_codes(self, **kwargs) -> torch.Tensor:
        """Just the codec tokens, for inspecting the Talker without the vocoder."""
        self._codes_only = True
        try:
            return self.speak(**kwargs)
        finally:
            self._codes_only = False

    @torch.no_grad()
    def synthesize(self, codes: torch.Tensor) -> torch.Tensor:
        """Codec tokens -> waveform, through DiT then BigVGAN."""
        if getattr(self, "_codes_only", False):
            return codes
        if codes.numel() == 0:
            # The DiT reshapes its rotary embedding by the code count, so an
            # empty sequence fails deep inside attention with an unrelated
            # message about an ambiguous -1. Say what actually happened: the
            # Talker emitted EOS immediately, which means its conditioning was
            # degenerate.
            raise ValueError(
                "the Talker produced no codec tokens -- it emitted EOS at the "
                "first step. The conditioning (thinker hidden + embeddings) is "
                "wrong or degenerate; there is nothing to vocode.")
        return self.token2wav(
            codes,
            conditioning=self.speaker_params["cond"].float(),
            reference_mel=self.speaker_params["ref_mel"].float(),
        )


def write_wav(path: str, waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> str:
    import soundfile as sf
    audio = waveform.detach().float().cpu().reshape(-1).numpy()
    sf.write(path, audio, sample_rate)
    return path
