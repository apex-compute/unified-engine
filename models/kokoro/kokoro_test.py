#!/usr/bin/env python3
"""
Kokoro-82M standalone CUDA reference inference.

This is a self-contained port of hexgrad/kokoro's model code (StyleTTS2 text/
prosody encoders + ISTFTNet decoder) into this single file, so we no longer
depend on the `kokoro` pip package. Only two things are still pulled from
outside:
  - `transformers` for AlbertModel/AlbertConfig (the PL-BERT text encoder
    backbone) -- a general ML library, not kokoro-specific.
  - `misaki` + espeak-ng for G2P (text -> IPA phonemes). This is a
    linguistic front-end, not model weights/forward-pass code, so it's kept
    as a dependency rather than reimplemented.

Weights (kokoro-v1_0.pth + config.json) and voice packs are downloaded from
HF on first run and cached under kokoro_bin/.

Stage 1: reference correctness on CUDA. Porting to our accelerator
convention (weight dump to bin/, custom kernels) comes next, following the
pattern in models/parakeet.
"""
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # repo root, for user_dma_core
BIN_DIR = os.path.join(SCRIPT_DIR, "kokoro_bin")
HF_REPO = "hexgrad/Kokoro-82M"
MODEL_FILENAME = "kokoro-v1_0.pth"

DEFAULT_TEXT = (
    "Hello, this is Kokoro, an open weight text to speech model with "
    "eighty two million parameters."
)


# ---------------------------------------------------------------------------
# StyleTTS2 modules (text encoder, prosody predictor)
# Ported from kokoro/modules.py (itself adapted from yl4579/StyleTTS2)
# ---------------------------------------------------------------------------

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear_layer(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)          # [B, T, emb]
        x = x.transpose(1, 2)          # [B, emb, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)          # [B, T, chn]
        lengths = input_lengths if input_lengths.device == torch.device('cpu') else input_lengths.to('cpu')
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad
        x.masked_fill_(m, 0.0)
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1,
                                       batch_first=True, bidirectional=True, dropout=dropout))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                lengths = text_lengths if text_lengths.device == torch.device('cpu') else text_lengths.to('cpu')
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)


# ---------------------------------------------------------------------------
# ISTFTNet decoder (AdaIN residual blocks + NSF source + iSTFT vocoder)
# Ported from kokoro/istftnet.py (itself adapted from yl4579/StyleTTS2)
# ---------------------------------------------------------------------------

def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), style_dim=64):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                   padding=get_padding(kernel_size, d))) for d in dilation
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                   padding=get_padding(kernel_size, 1))) for _ in dilation
        ])
        self.convs2.apply(init_weights)
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilation])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in dilation])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in self.convs1])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in self.convs2])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x


class TorchSTFT(nn.Module):
    """Real torch.stft/istft based vocoder transform (CUDA fast path)."""

    def __init__(self, filter_length=800, hop_length=200, win_length=800):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length, periodic=True, dtype=torch.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data, self.filter_length, self.hop_length, self.win_length,
            window=self.window.to(input_data.device), return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length,
            window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)


class SineGen(nn.Module):
    def __init__(self, samp_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale,
                                    mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale,
                               mode="linear").transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0):
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                 resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=math.prod(upsample_rates) * gen_istft_hop_size,
            harmonic_num=8, voiced_threshod=10)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates) * gen_istft_hop_size)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                    k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(
                    gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0,
                    padding=(stride_f0 + 1) // 2))
                self.noise_res.append(AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)

    def forward(self, x, s, f0):
        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            har_source, noi_source, uv = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                xs = self.resblocks[i * self.num_kernels + j](x, s) if xs is None else \
                    xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        return F.interpolate(x, scale_factor=2, mode='nearest')


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))
        self.dropout = nn.Dropout(dropout_p)
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in,
                                                         padding=1, output_padding=1))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
        return out


class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, dim_out, resblock_kernel_sizes, upsample_rates,
                 upsample_initial_channel, resblock_dilation_sizes, upsample_kernel_sizes,
                 gen_istft_n_fft, gen_istft_hop_size):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList()
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True))
        self.F0_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1))
        self.asr_res = nn.Sequential(weight_norm(nn.Conv1d(512, 64, kernel_size=1)))
        self.generator = Generator(style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                                    resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size)

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return self.generator(x, s, F0_curve)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ])
        self.N = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ])
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


# ---------------------------------------------------------------------------
# PL-BERT text encoder (AlbertModel wrapper that returns last_hidden_state)
# ---------------------------------------------------------------------------

class CustomAlbert(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import AlbertModel
        self.albert = AlbertModel(config)
        self.config = config

    def forward(self, *args, **kwargs):
        return self.albert(*args, **kwargs).last_hidden_state

    def load_state_dict(self, state_dict, strict=True):
        return self.albert.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------
# Top-level model: glues text encoder, prosody predictor, and decoder
# Ported from kokoro/model.py
# ---------------------------------------------------------------------------

class KokoroModel(nn.Module):
    def __init__(self, config: Union[Dict, str]):
        super().__init__()
        if not isinstance(config, dict):
            with open(config, "r", encoding="utf-8") as f:
                config = json.load(f)
        self.vocab = config["vocab"]
        from transformers import AlbertConfig
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"], d_hid=config["hidden_dim"],
            nlayers=config["n_layer"], max_dur=config["max_dur"], dropout=config["dropout"])
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"], kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"], n_symbols=config["n_token"])
        self.decoder = Decoder(
            dim_in=config["hidden_dim"], style_dim=config["style_dim"], dim_out=config["n_mels"],
            **config["istftnet"])

    def load_weights(self, model_path):
        for key, state_dict in torch.load(model_path, map_location="cpu", weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except Exception:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1):
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1],
                                    device=input_ids.device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(self.device)
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=self.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        return audio, pred_dur

    def forward(self, phonemes: str, ref_s: torch.FloatTensor, speed: float = 1) -> "KokoroModel.Output":
        input_ids = [self.vocab[p] for p in phonemes if p in self.vocab]
        assert len(input_ids) + 2 <= self.context_length, (len(input_ids) + 2, self.context_length)
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)
        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        return self.Output(audio=audio.squeeze().cpu(), pred_dur=pred_dur.cpu())


# ---------------------------------------------------------------------------
# Weight / voice download + G2P frontend + CLI driver
# ---------------------------------------------------------------------------

def ensure_weights():
    os.makedirs(BIN_DIR, exist_ok=True)
    model_path = os.path.join(BIN_DIR, MODEL_FILENAME)
    config_path = os.path.join(BIN_DIR, "config.json")
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        print(f"Model files not found, downloading {HF_REPO} to {BIN_DIR} ...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILENAME, local_dir=BIN_DIR)
        config_path = hf_hub_download(repo_id=HF_REPO, filename="config.json", local_dir=BIN_DIR)
    else:
        print(f"Found cached weights in {BIN_DIR}")
    return model_path, config_path


def ensure_voice(voice: str, device: str):
    voice_path = os.path.join(BIN_DIR, "voices", f"{voice}.pt")
    if not os.path.exists(voice_path):
        print(f"Voice '{voice}' not found, downloading ...")
        from huggingface_hub import hf_hub_download
        voice_path = hf_hub_download(repo_id=HF_REPO, filename=f"voices/{voice}.pt", local_dir=BIN_DIR)
    return torch.load(voice_path, weights_only=True).to(device)


def text_to_phonemes(text: str, british: bool = False) -> str:
    """G2P frontend: text -> IPA phoneme string, matching kokoro's American/British English path."""
    from misaki import en, espeak
    try:
        fallback = espeak.EspeakFallback(british=british)
    except Exception as e:
        print(f"WARNING: EspeakFallback not enabled ({e}); OOD words will be skipped")
        fallback = None
    g2p = en.G2P(trf=False, british=british, fallback=fallback, unk='')
    _, tokens = g2p(text)
    return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens if t.phonemes).strip()


def main():
    parser = argparse.ArgumentParser(description="Kokoro-82M standalone CUDA reference inference")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--voice", type=str, default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=os.path.join(SCRIPT_DIR, "kokoro_out.wav"))
    parser.add_argument("--fpga", action="store_true",
                         help="Run on the UnifiedEngine FPGA accelerator instead of CUDA/CPU. "
                              "Being built section-by-section (see fpga_forward.py); sections not "
                              "yet ported fall back to the CUDA/CPU KokoroModel path for now, "
                              "compared via SNR against the FPGA section's own output.")
    parser.add_argument("--dev", type=str, default="xdma0",
                         help="XDMA device name for --fpga (e.g. xdma0).")
    args = parser.parse_args()

    if args.fpga:
        args.device = "cpu"  # reference/comparison tensors still computed on host

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    model_path, config_path = ensure_weights()

    print(f"Loading Kokoro model on {args.device} ...")
    model = KokoroModel(config_path)
    model.load_weights(model_path)
    model = model.to(args.device).eval()

    ref_s = ensure_voice(args.voice, args.device)

    print(f"Phonemizing text: {args.text!r}")
    phonemes = text_to_phonemes(args.text, british=args.voice.startswith("b"))
    print(f"Phonemes: {phonemes}")

    if args.fpga:
        from fpga_forward import run_fpga_forward
        print("Running FPGA inference (only sections currently ported to hardware) ...")
        run_fpga_forward(model, phonemes, ref_s[len(phonemes) - 1], speed=args.speed, dev=args.dev)
        return  # no audio yet -- see fpga_forward.py's section checklist

    print("Running inference ...")
    output = model(phonemes, ref_s[len(phonemes) - 1], speed=args.speed)

    import soundfile as sf
    sf.write(args.out, output.audio.numpy(), 24000)
    print(f"Wrote {len(output.audio) / 24000:.2f}s of audio to {args.out}")


if __name__ == "__main__":
    main()
