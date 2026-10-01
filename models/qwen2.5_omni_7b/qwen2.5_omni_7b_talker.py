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

Three of the four tensors are already available:

  step_embeds     the embedding lookup already happens host-side
  prefill_embeds  likewise
  step_hidden     LM_OUT_NORM holds exactly this after each decode step; it is
                  a 3584-element readback per token

The fourth, PREFILL_HIDDEN, needs the final norm applied to EVERY prompt row.
When speech is requested, prefill emits one extra
`rms_norm_core_dram(M=seq_len, N=H)` over the final layer output into a [T, H]
buffer, plus a T*3584*2 byte readback -- about 13 MiB at a 1899-token prompt.
"""

from __future__ import annotations

import json
import os
import queue
import time
from typing import Any

import torch
from transformers import Qwen2_5OmniTalkerForConditionalGeneration

SPEAKERS = ("Chelsie", "Ethan")
DEFAULT_SPEAKER = "Chelsie"
SAMPLE_RATE = 24000


class ReplyStream:
    """Completed Thinker steps, consumed by the host Talker as they arrive."""

    _END = object()

    def __init__(self, embed_lookup, text_eos: int, text_pad: int):
        self._queue: queue.Queue = queue.Queue()
        self._embed_lookup = embed_lookup
        self._text_eos = text_eos
        self._text_pad = text_pad
        self._tail = 0
        self._closed = False
        self._error: BaseException | None = None

    def push(self, token: int, hidden: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("cannot publish a Thinker step after speech stream closure")
        self._queue.put((int(token), hidden.clone()))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(self._END)

    def abort(self, error: BaseException) -> None:
        self._error = error
        self.close()

    def next_step(self) -> tuple[int, torch.Tensor] | None:
        if self._tail:
            return None
        try:
            item = self._queue.get(timeout=60)
        except queue.Empty as exc:
            raise TimeoutError("waiting for the next FPGA Thinker step") from exc
        if item is self._END:
            self._tail = 1
            if self._error is not None:
                raise RuntimeError("FPGA Thinker stopped before speech completed") from self._error
            return None
        return item

    def next_condition(self) -> torch.Tensor:
        item = self.next_step()
        if item is not None:
            token, hidden = item
            embed = self._embed_lookup(torch.tensor([[token]], dtype=torch.long))
            return hidden.float().reshape(1, 1, -1) + embed.float()
        token = self._text_eos if self._tail == 1 else self._text_pad
        self._tail = 2
        return self._embed_lookup(torch.tensor([[token]], dtype=torch.long)).float()


class _StreamingTalker(Qwen2_5OmniTalkerForConditionalGeneration):
    """Inject each newly available text condition into HF's cached decode."""

    reply_stream = None

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        if self.reply_stream is not None:
            model_kwargs["thinker_reply_part"] = self.reply_stream.next_condition().to(self.dtype)
        return model_kwargs


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

        self.talker = _StreamingTalker(cfg.talker_config)
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
        self.last_metrics: dict[str, float | int | str] | None = None

    def _record_metrics(self, codes: torch.Tensor, waveform: torch.Tensor,
                        talker_s: float, token2wav_s: float) -> None:
        audio = waveform[0] if isinstance(waveform, (tuple, list)) else waveform
        self.last_metrics = {
            "talker_device": "Host CPU",
            "talker_wall_s": talker_s,
            "codec_tokens": int(codes.shape[1]),
            "token2wav_device": "Host CPU",
            "token2wav_wall_s": token2wav_s,
            "audio_samples": int(audio.numel()),
        }

    @property
    def codec_tokens(self) -> dict[str, int]:
        t = self.talker
        return {"mask": t.codec_mask_token, "pad": t.codec_pad_token,
                "bos": t.codec_bos_token, "text_eos": t.text_eos_token,
                "text_pad": t.text_pad_token}

    @torch.no_grad()
    def speak(self, *, input_ids: torch.Tensor, prefill_hidden: torch.Tensor,
              prefill_embeds: torch.Tensor, step_hidden: torch.Tensor,
              step_embeds: torch.Tensor, first_reply_token: int, embed_lookup,
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

        # The initial codec prefix includes the first generated text token.
        # Its length must match the prompt-mask + codec PAD/BOS prefix below.
        first_reply = torch.tensor([[first_reply_token]], dtype=torch.long, device=dev)
        talker_input_text_ids = torch.cat([input_ids, bos, first_reply], dim=1)
        # Codec stream: the prompt is masked (there is no speech for it yet),
        # then pad, then the codec BOS the model actually starts decoding from.
        talker_input_ids = torch.cat([
            torch.full_like(input_ids, fill_value=talker.codec_mask_token),
            torch.tensor([[talker.codec_pad_token]], dtype=torch.long, device=dev),
            torch.tensor([[talker.codec_bos_token]], dtype=torch.long, device=dev),
        ], dim=1)
        if talker_input_text_ids.shape != talker_input_ids.shape:
            raise AssertionError("Talker text and codec prefixes must align")
        # The reference supplies this mask even for an unpadded text prompt.
        # In Talker.forward it also triggers the codec PAD/BOS embeddings and
        # multimodal position setup for the two prefix rows.
        talker_attention_mask = torch.ones_like(talker_input_text_ids)

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

        talker_started = time.perf_counter()
        codes = talker.generate(
            input_ids=talker_input_ids,
            input_text_ids=talker_input_text_ids,
            attention_mask=talker_attention_mask,
            thinker_reply_part=thinker_reply_part,
            inputs_embeds=inputs_embeds,
            suppress_tokens=[talker.codec_bos_token],
            max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k,
            top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=[8292, 8294],
        )
        codes = codes[:, talker_input_ids.shape[1]:-1]
        talker_s = time.perf_counter() - talker_started
        print(f"  [Speak] Talker generated {codes.shape[1]} codec tokens in "
              f"{talker_s:.1f}s; starting CPU "
              "Token2Wav", flush=True)
        vocoder_started = time.perf_counter()
        waveform = self.synthesize(codes)
        token2wav_s = time.perf_counter() - vocoder_started
        self._record_metrics(codes, waveform, talker_s, token2wav_s)
        print(f"  [Speak] CPU Token2Wav finished in "
              f"{token2wav_s:.1f}s", flush=True)
        return waveform

    @torch.no_grad()
    def speak_stream(self, *, input_ids: torch.Tensor,
                     prefill_hidden_rows: torch.Tensor,
                     prefill_embeds: torch.Tensor, reply_stream: ReplyStream,
                     embed_lookup, max_new_tokens: int = 4096,
                     do_sample: bool = True, top_k: int = 40,
                     top_p: float = 0.8, temperature: float = 0.9,
                     repetition_penalty: float = 1.05) -> torch.Tensor:
        """Start Talker after the seed and first reply step; await later steps."""
        seed = reply_stream.next_step()
        first_reply = reply_stream.next_step()
        if seed is None or first_reply is None:
            raise RuntimeError("Thinker ended before it supplied the speech prefix")
        if seed[0] != int(input_ids[0, -1]):
            raise RuntimeError("speech stream's first step is not the prompt seed")

        talker = self.talker
        dt = talker.dtype
        bos = torch.tensor([[self.speaker_params["bos_token"]]], dtype=torch.long)
        first_token = torch.tensor([[first_reply[0]]], dtype=torch.long)
        talker_input_text_ids = torch.cat([input_ids, bos, first_token], dim=1)
        talker_input_ids = torch.cat([
            torch.full_like(input_ids, fill_value=talker.codec_mask_token),
            torch.tensor([[talker.codec_pad_token]], dtype=torch.long),
            torch.tensor([[talker.codec_bos_token]], dtype=torch.long),
        ], dim=1)
        if talker_input_ids.shape != talker_input_text_ids.shape:
            raise AssertionError("Talker text and codec prefixes must align")
        prefix_hidden = torch.cat([
            prefill_hidden_rows.float().reshape(1, -1, prefill_embeds.shape[-1]),
            seed[1].float().reshape(1, 1, -1),
        ], dim=1)
        first_condition = (
            first_reply[1].float().reshape(1, 1, -1)
            + embed_lookup(first_token).float()
        )
        inputs_embeds = torch.cat([
            prefix_hidden + prefill_embeds.float(),
            embed_lookup(bos).float(),
            first_condition,
        ], dim=1).to(dt)

        talker.reply_stream = reply_stream
        started = time.perf_counter()
        reply_stream.talker_started_at = started
        try:
            codes = talker.generate(
                input_ids=talker_input_ids,
                input_text_ids=talker_input_text_ids,
                attention_mask=torch.ones_like(talker_input_text_ids),
                thinker_reply_part=torch.zeros((1, 1, prefill_embeds.shape[-1]), dtype=dt),
                inputs_embeds=inputs_embeds,
                suppress_tokens=[talker.codec_bos_token],
                max_new_tokens=max_new_tokens, do_sample=do_sample,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=[8292, 8294],
            )
        finally:
            reply_stream.talker_finished_at = time.perf_counter()
            talker.reply_stream = None
        codes = codes[:, talker_input_ids.shape[1]:-1]
        talker_s = reply_stream.talker_finished_at - started
        print(f"  [Speak] streaming Talker generated {codes.shape[1]} codec tokens "
              f"in {talker_s:.1f}s; starting CPU Token2Wav",
              flush=True)
        vocoder_started = time.perf_counter()
        waveform = self.synthesize(codes)
        token2wav_s = time.perf_counter() - vocoder_started
        self._record_metrics(codes, waveform, talker_s, token2wav_s)
        print(f"  [Speak] CPU Token2Wav finished in "
              f"{token2wav_s:.1f}s", flush=True)
        return waveform

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


class ThinkerEmbeddings:
    """Row lookups into the Thinker's embedding table, straight from the shards.

    This model keeps its BF16 Thinker embedding table on the host and sends
    selected rows to FPGA DRAM. Reading the same checkpoint rows for the
    Talker is exact and matches the reference implementation. Rows are fetched
    lazily by slice, so a few hundred of them never materialise the 1 GiB table.
    """

    def __init__(self, model_dir: str, name: str = "thinker.model.embed_tokens.weight"):
        from safetensors import safe_open
        index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        shard = index["weight_map"][name]
        self._f = safe_open(os.path.join(model_dir, shard), framework="pt")
        self._name = name
        self._cache: dict[int, torch.Tensor] = {}
        self.hidden = self._f.get_slice(name).get_shape()[1]

    def rows(self, ids, zero_at=()) -> torch.Tensor:
        """[1, N, H] embeddings for ``ids``; positions in ``zero_at`` come back 0.

        MEDIA POSITIONS ARE ZEROED, not looked up. A vision or audio placeholder
        has no meaningful embedding -- the encoder output was substituted for it
        downstream -- and the reference zeroes exactly those rows before handing
        the stream to the Talker.
        """
        sl = self._f.get_slice(self._name)
        zero = set(zero_at)
        out = torch.zeros(len(ids), self.hidden, dtype=torch.float32)
        for i, tok in enumerate(ids):
            if i in zero:
                continue
            t = self._cache.get(tok)
            if t is None:
                t = sl[tok:tok + 1].to(torch.float32).reshape(-1)
                self._cache[tok] = t
            out[i] = t
        return out.unsqueeze(0)
