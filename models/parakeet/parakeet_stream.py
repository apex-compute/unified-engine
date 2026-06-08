#!/usr/bin/env python3
"""
Parakeet-TDT-0.6B real-time streaming on FPGA (dual-engine).

Loads pre-compiled bins from parakeet_bin/, receives audio via WebSocket,
transcribes in 5-second chunks, displays via curses TUI.

Usage:
  python parakeet_stream.py
  python parakeet_stream.py --dev xdma0 --chunk-seconds 5 --port 8000
  # Then open http://localhost:8000/mic in a browser to stream audio.
"""

import argparse
import asyncio
import json
import math
import os
import struct
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import builtins
_original_print = builtins.print
builtins.print = lambda *a, **k: None  # suppress all prints (user_dma_core, etc.)

import user_dma_core
from user_dma_core import (
    DMA_DEVICE_H2C, DMA_DEVICE_C2H, UnifiedEngine, set_dma_device,
    UE_ARGMAX_INDEX,
)

# Re-use helpers from parakeet_test
from parakeet_test import (
    Parakeet_UnifiedEngine, load_config, pad_to_multiple, conv2d_outsize,
    compute_mel_spectrogram, read_dram,
    PARAKEET_PARAMS_BASE, PARAKEET_TENSOR_BASE, PARAKEET_PROGRAM_BASE,
    WEIGHTS_PATH, TOKENIZER_PATH,
)

# ============================================================================
# Shared state
# ============================================================================
SAMPLE_RATE = 16000

audio_lock = threading.Lock()
audio_buffer: list[np.ndarray] = []
audio_total_samples: int = 0

transcript_lock = threading.Lock()
transcript_segments: list[str] = []

ws_connected = False
chunks_processed = 0

# ============================================================================
# FPGA engine globals (set in main)
# ============================================================================
engine = None
engine2 = None
cfg = None
sp = None  # sentencepiece
chunk_seconds = 5
chunk_samples = SAMPLE_RATE * chunk_seconds

# Program addresses (loaded from bin)
progs = {}       # master programs
progs2 = {}      # slave programs

# Precomputed dims for the fixed chunk size
L_pad = 0
L = 0
H2 = 0
W2 = 0
N0 = 0
SC = 0
D = 0
bpe = 2

def init_engine(args):
    """Load params + programs from bin, set up engines for streaming."""
    global engine, engine2, cfg, sp, chunk_seconds, chunk_samples
    global progs, progs2, L_pad, L, H2, W2, N0, SC, D, bpe

    chunk_seconds = args.chunk_seconds
    chunk_samples = SAMPLE_RATE * chunk_seconds

    set_dma_device(args.dev)
    cfg = load_config()

    # Compute dims for the fixed chunk size
    T_mel_max = int(chunk_seconds * SAMPLE_RATE / cfg["preprocessing"]["hop_length"]) + 2
    H0, W0 = conv2d_outsize(T_mel_max), conv2d_outsize(cfg["encoder"]["n_mels"])
    H1, W1 = conv2d_outsize(H0), conv2d_outsize(W0)
    H2, W2 = conv2d_outsize(H1), conv2d_outsize(W1)
    N0 = H0 * W0
    L = H2
    L_pad = pad_to_multiple(L, cfg["hardware"]["block_size"])
    SC = cfg["encoder"]["sub_channels"]
    D = cfg["encoder"]["d_model"]
    bpe = cfg["encoder"]["bytes_per_element"]

    _original_print(f"Chunk: {chunk_seconds}s, T_mel≈{T_mel_max}, L={L}, L_pad={L_pad}")

    # Master engine
    engine = Parakeet_UnifiedEngine(clock_period_ns=args.cycle, dual_engine=True)
    params_ok = engine.load_params(L_pad)
    assert params_ok, "params.bin not found — run parakeet_test.py --dual-engine first"

    engine.tensor_init(L_pad)

    loaded = engine.load_programs(L_pad)
    assert loaded, "programs.bin not found — run parakeet_test.py --dual-engine first"
    progs.update(loaded)

    # Position embedding + attention mask
    P_pad = pad_to_multiple(2 * L_pad - 1, cfg["hardware"]["block_size"])
    rel_pe = engine.make_rel_pos_emb(L_pad)
    if rel_pe.shape[0] < P_pad:
        pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
        pe_padded[:rel_pe.shape[0], :] = rel_pe
        rel_pe = pe_padded
    engine.dma_to_accelerator_memory(engine.POS_EMB_DRAM, rel_pe.contiguous())
    mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
    mask[:L, :L] = 0.0
    mask[L:, 0] = 0.0
    engine.dma_to_accelerator_memory(engine.ATTN_MASK_DRAM, mask.contiguous())

    # Slave engine
    engine2 = Parakeet_UnifiedEngine(clock_period_ns=args.cycle, dual_engine=True, engine_slave=True)
    engine2.copy_dram_layout(engine)
    loaded2 = engine2.load_programs(L_pad)
    assert loaded2, "programs_slave.bin not found — run parakeet_test.py --dual-engine first"
    progs2.update(loaded2)

    engine.progs = {
        "pred": (progs["pred"], 0),
        "joint_tok": (progs["tok"], 0),
        "joint_dur": (progs["dur"], 0),
        "state_restore": (progs["restore"], 0),
    }

    # Tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)

    # Load checkpoint for mel filterbank
    Parakeet_UnifiedEngine.ensure_model_files()
    global _ckpt_sd
    _ckpt_sd = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)

    _original_print("Engine ready.")

_ckpt_sd = None

def transcribe_chunk(samples: np.ndarray) -> str | None:
    """Run FPGA inference on a PCM chunk. Returns transcript text or None."""
    if len(samples) < 1600:
        return None

    waveform = torch.from_numpy(samples).unsqueeze(0).float()

    # Mel spectrogram (CPU)
    mel = compute_mel_spectrogram(waveform, cfg, ckpt_sd=_ckpt_sd)
    T_mel = mel.shape[1]

    # Recompute actual dims for this chunk
    h0 = conv2d_outsize(T_mel)
    w0 = conv2d_outsize(cfg["encoder"]["n_mels"])
    n0 = h0 * w0
    h1, w1 = conv2d_outsize(h0), conv2d_outsize(w0)
    h2, w2 = conv2d_outsize(h1), conv2d_outsize(w1)
    actual_L = h2

    N1 = h1 * w1
    N1_pad = pad_to_multiple(N1, cfg["hardware"]["block_size"])
    N2 = h2 * w2
    N2_pad = pad_to_multiple(N2, cfg["hardware"]["block_size"])

    # Upload mel
    mel_flat = mel.squeeze(0).to(torch.bfloat16).contiguous()
    engine.dma_to_accelerator_memory(engine.MEL_DRAM, mel_flat)
    engine.dma_to_accelerator_memory(engine.SUB_OUT0_DRAM, torch.zeros(n0 * SC, dtype=torch.bfloat16))

    # Subsampling
    for key in ["im2col_s0", "prog_s0", "im2col_s1", "prog_s1", "im2col_s2", "prog_s2"]:
        if key in progs2:
            engine2.start_execute_from_dram(progs2[key])
        engine.program_execute(progs[key])

    engine.dma_to_accelerator_memory(engine.INPUT_DRAM, torch.zeros(L_pad * D, dtype=torch.bfloat16))
    if "prog_flatten_lin" in progs2:
        engine2.start_execute_from_dram(progs2["prog_flatten_lin"])
    engine.program_execute(progs["prog_flatten_lin"])

    if L_pad > h2:
        ef = torch.zeros((L_pad - h2) * D, dtype=torch.bfloat16)
        ef[0::2] = 0.1; ef[1::2] = -0.1
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM + h2 * D * bpe, ef)

    # Encoder
    if "encoder" in progs2:
        engine2.start_execute_from_dram(progs2["encoder"])
    engine.start_execute_from_dram(progs["encoder"])
    engine.wait_queue(120.0)

    # Decoder
    hw_enc_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D)
    engine.dma_to_accelerator_memory(engine.ENC_OUT_DRAM, hw_enc_out.contiguous())
    hw_tokens = engine.run_decode(engine.ENC_OUT_DRAM, actual_L)

    if not hw_tokens:
        return None
    text = sp.DecodeIds([t for t in hw_tokens if 0 <= t < sp.GetPieceSize()])
    return text.strip() if text else None

# ============================================================================
# FastAPI server
# ============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    yield

app = FastAPI(lifespan=lifespan)

transcript_ws_clients: list[WebSocket] = []

MIC_HTML = """<!DOCTYPE html><html><head><title>Parakeet Live</title>
<style>
  body{font-family:monospace;background:#111;color:#eee;margin:2em;}
  #transcript{white-space:pre-wrap;line-height:1.6;font-size:1.1em;}
  button{font-size:1.2em;padding:0.4em 1.2em;margin-bottom:1em;cursor:pointer;}
  .seg{color:#8f8;}
  .status{color:#888;font-size:0.9em;}
</style></head><body>
<button id="btn" onclick="toggle()">Start</button>
<span id="status" class="status"></span>
<div id="transcript"></div>
<script>
let aws,tws,ctx,proc,src,stream,running=false;
function setStatus(s){document.getElementById('status').textContent=' '+s;}
async function toggle(){
  if(running){stop();return;}
  aws=new WebSocket('ws://'+location.host+'/ws/audio');
  aws.binaryType='arraybuffer';
  aws.onclose=()=>{setStatus('disconnected');running=false;document.getElementById('btn').textContent='Start';};
  tws=new WebSocket('ws://'+location.host+'/ws/transcript');
  tws.onmessage=e=>{
    let d=JSON.parse(e.data);
    let div=document.getElementById('transcript');
    let line=document.createElement('div');
    line.className='seg';
    div.appendChild(line);
    let chars=d.text.split('');
    let delay=5000/Math.max(chars.length,1);
    chars.forEach((c,i)=>{setTimeout(()=>{line.textContent+=c;window.scrollTo(0,document.body.scrollHeight);},i*delay);});
  };
  ctx=new AudioContext({sampleRate:16000});
  stream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1,echoCancellation:true}});
  src=ctx.createMediaStreamSource(stream);
  proc=ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=e=>{if(aws.readyState===1)aws.send(e.inputBuffer.getChannelData(0).buffer);};
  src.connect(proc);proc.connect(ctx.destination);
  running=true;document.getElementById('btn').textContent='Stop';setStatus('streaming...');
}
function stop(){
  if(proc){proc.disconnect();src.disconnect();}
  if(stream)stream.getTracks().forEach(t=>t.stop());
  if(aws)aws.close();if(tws)tws.close();
  running=false;document.getElementById('btn').textContent='Start';
}
</script></body></html>"""

@app.get("/mic")
async def mic_page():
    return HTMLResponse(MIC_HTML)

@app.websocket("/ws/transcript")
async def ws_transcript(ws: WebSocket):
    await ws.accept()
    transcript_ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        if ws in transcript_ws_clients:
            transcript_ws_clients.remove(ws)

def _broadcast_transcript(text: str):
    """Push transcript to all connected browser clients."""
    msg = json.dumps({"text": text})
    dead = []
    for ws in transcript_ws_clients:
        try:
            loop = _event_loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in transcript_ws_clients:
            transcript_ws_clients.remove(ws)

def _process_thread_fn():
    """Background thread: polls buffer, runs FPGA when 5s ready."""
    while True:
        with audio_lock:
            ready = audio_total_samples >= chunk_samples
        if ready:
            _process_buffer()
        else:
            time.sleep(0.1)

_process_thread = threading.Thread(target=_process_thread_fn, daemon=True)
_process_thread.start()

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    global audio_total_samples, ws_connected
    await ws.accept()
    ws_connected = True
    try:
        while True:
            data = await ws.receive_bytes()
            n = len(data) // 4
            if n == 0:
                continue
            arr = np.frombuffer(data[:n * 4], dtype=np.float32).copy()
            with audio_lock:
                audio_buffer.append(arr)
                audio_total_samples += len(arr)
    except WebSocketDisconnect:
        pass
    finally:
        ws_connected = False
        _process_buffer(flush=True)

OVERLAP_SAMPLES = int(SAMPLE_RATE * 0.5)  # 0.5s overlap
_prev_tail: np.ndarray | None = None

def _process_buffer(flush=False):
    global audio_total_samples, _prev_tail
    with audio_lock:
        if not flush and audio_total_samples < chunk_samples:
            return
        if not audio_buffer:
            return
        # Concatenate everything, then take exactly chunk_samples
        all_audio = np.concatenate(audio_buffer).astype(np.float32)
        if flush:
            to_process = all_audio
            audio_buffer.clear()
            audio_total_samples = 0
        else:
            to_process = all_audio[:chunk_samples]
            leftover = all_audio[chunk_samples:]
            audio_buffer.clear()
            if len(leftover) > 0:
                audio_buffer.append(leftover)
                audio_total_samples = len(leftover)
            else:
                audio_total_samples = 0

    # Prepend overlap from previous chunk for word continuity
    if _prev_tail is not None:
        to_process = np.concatenate([_prev_tail, to_process])
    _prev_tail = to_process[-OVERLAP_SAMPLES:].copy()

    global chunks_processed
    chunks_processed += 1
    dur = len(to_process) / SAMPLE_RATE
    _original_print(f"[chunk {chunks_processed}] {dur:.1f}s audio, processing...", flush=True)
    t0 = time.perf_counter()
    text = transcribe_chunk(to_process)
    elapsed = time.perf_counter() - t0
    _original_print(f"[chunk {chunks_processed}] {elapsed:.2f}s -> {text or '(empty)'}", flush=True)
    if text:
        with transcript_lock:
            transcript_segments.append(text)
        _broadcast_transcript(text)

# ============================================================================
# Main
# ============================================================================
_event_loop = None

def main():
    global _event_loop
    args = argparse.Namespace(dev="xdma0", cycle=5.1594, port=8000, chunk_seconds=5)

    _original_print("Initializing FPGA engines...")
    init_engine(args)
    _original_print(f"Open http://localhost:8000/mic in browser")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

if __name__ == "__main__":
    main()
