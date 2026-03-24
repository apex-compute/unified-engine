#!/usr/bin/env python3
"""Parakeet-TDT-0.6B live transcription on FPGA accelerator.

Compile once at startup, then stream audio chunks through the hardware.
WebSocket server captures mic audio from browser, transcribes on FPGA,
pushes text to a curses TUI.

Usage:
  python parakeet_stream.py --dev xdma0 --port 8000
"""

import json, math, os, sys, threading, time, queue, textwrap, signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

# --- Import engine and ops from parakeet_test ---
from parakeet_test import (
    Parakeet_UnifiedEngine, load_config, compute_mel_spectrogram,
    pad_to_multiple, conv2d_outsize, compile_and_run, read_dram,
    batch_norm_core_dram, half_step_residual_core_dram, silu_core_dram,
    glu_core_dram, rel_shift_core_dram, chunked_transpose_core_dram,
    URAM_A_BASE, URAM_B_BASE, WEIGHTS_PATH, TOKENIZER_PATH,
)
from user_dma_core import DMA_DEVICE_C2H, set_dma_device

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    sys.exit("Missing: pip install fastapi uvicorn websockets")

try:
    import websockets.sync.client as ws_sync_client
    import websockets.exceptions
except ImportError:
    sys.exit("Missing: pip install websockets")

import curses

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHUNK_SECONDS = 3
SAMPLE_RATE = 16000
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

# ---------------------------------------------------------------------------
# Globals (filled at startup)
# ---------------------------------------------------------------------------
engine = None
cfg = None
sd = None
tokenizer = None
enc_prog_addr = None
prog_s0 = None
prog_lin = None
L_pad = None
L = None  # valid timesteps (depends on audio, but max = L_pad)
H2 = None
H0 = W0 = N0 = SC = D = FF = bpe = None
H_heads = dk = P_pad = None
toeplitz_addrs = None
engine_lock = threading.Lock()

# Shared audio/transcript state
audio_lock = threading.Lock()
audio_buffer = []
audio_total_samples = 0
transcript_lock = threading.Lock()
transcript_segments = []
transcript_clients = set()
transcript_clients_lock = threading.Lock()

# ---------------------------------------------------------------------------
# HTML mic page (from audio_streaming.py)
# ---------------------------------------------------------------------------
MIC_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Parakeet FPGA Mic</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#e0e0e0;font-family:'SF Mono','Fira Code',monospace;
display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;gap:24px}
h1{font-size:1.4rem;color:#00d2ff}
#status{padding:8px 18px;border-radius:6px;font-size:.95rem;background:#16213e;border:1px solid #0f3460}
#status.connected{border-color:#00d2ff;color:#00d2ff}
#status.streaming{border-color:#00ff88;color:#00ff88}
#status.error{border-color:#ff4444;color:#ff4444}
#vb-c{width:260px;height:14px;background:#16213e;border-radius:7px;overflow:hidden;border:1px solid #0f3460}
#vb{height:100%;width:0%;background:linear-gradient(90deg,#00d2ff,#00ff88);transition:width 60ms linear}
button{padding:12px 32px;font-size:1rem;border:none;border-radius:8px;cursor:pointer;background:#0f3460;color:#00d2ff;font-weight:600}
button:hover{background:#1a4a80}
small{color:#888;font-size:.8rem}
</style></head><body>
<h1>Parakeet FPGA Mic Capture</h1>
<div id="status">Click Start to begin</div>
<div id="vb-c"><div id="vb"></div></div>
<button id="btn" onclick="toggle()">Start</button>
<small>Audio streamed as 16 kHz mono float32 PCM over WebSocket.</small>
<script>
const SR=16000;let ws=null,streaming=false,actx=null,src=null,proc=null;
function ss(t,c){const e=document.getElementById('status');e.textContent=t;e.className=c||''}
function toggle(){streaming?stop():start()}
async function start(){ss('Requesting mic...','');let s;
try{s=await navigator.mediaDevices.getUserMedia({audio:true})}catch(e){ss('Mic denied: '+e.message,'error');return}
actx=new AudioContext();const isr=actx.sampleRate;src=actx.createMediaStreamSource(s);
const p=location.protocol==='https:'?'wss':'ws';
ws=new WebSocket(`${p}://${location.host}/ws/audio`);ws.binaryType='arraybuffer';
ws.onopen=()=>ss('Connected — streaming','streaming');
ws.onclose=()=>{ss('Disconnected','');stop()};ws.onerror=()=>ss('WebSocket error','error');
const bs=4096;proc=actx.createScriptProcessor(bs,1,1);
proc.onaudioprocess=(e)=>{if(!ws||ws.readyState!==WebSocket.OPEN)return;
const inp=e.inputBuffer.getChannelData(0);const r=SR/isr;const ol=Math.round(inp.length*r);
const re=new Float32Array(ol);for(let i=0;i<ol;i++){const si=i/r;const idx=Math.floor(si);
const f=si-idx;re[i]=(inp[idx]||0)+f*((inp[Math.min(idx+1,inp.length-1)]||0)-(inp[idx]||0))}
let sm=0;for(let i=0;i<re.length;i++)sm+=re[i]*re[i];
document.getElementById('vb').style.width=Math.min(Math.sqrt(sm/re.length)*500,100)+'%';
ws.send(re.buffer)};src.connect(proc);proc.connect(actx.destination);
streaming=true;document.getElementById('btn').textContent='Stop'}
function stop(){if(proc){proc.disconnect();proc=null}if(src){src.disconnect();src=null}
if(actx){actx.close();actx=null}if(ws){ws.close();ws=null}streaming=false;
document.getElementById('btn').textContent='Start';document.getElementById('vb').style.width='0%'}
</script></body></html>"""

# ---------------------------------------------------------------------------
# FPGA transcription
# ---------------------------------------------------------------------------
def transcribe_chunk(waveform_tensor):
    """Run full FPGA pipeline on a waveform tensor. Returns text or None."""
    global L
    if engine is None or waveform_tensor.shape[1] < 1600:
        return None

    mel = compute_mel_spectrogram(waveform_tensor, cfg, ckpt_sd=sd)
    T_mel = mel.shape[1]
    n_mels = cfg["encoder"]["n_mels"]

    # Compute actual L for this audio
    h0 = conv2d_outsize(T_mel)
    w0 = conv2d_outsize(n_mels)
    h1 = conv2d_outsize(h0)
    h2 = conv2d_outsize(h1)
    cur_L = h2
    if cur_L > L_pad:
        print(f"  WARNING: audio too long (L={cur_L} > L_pad={L_pad}), skipping")
        return None
    L = cur_L

    with engine_lock:
        # Subsampling stage 0 — use actual dims for this chunk
        mel_2d = mel.squeeze(0)
        patches0, h0_act, w0_act = engine._im2col_conv2d(mel_2d, T_mel, n_mels)
        n0_act = h0_act * w0_act
        engine.dma_to_accelerator_memory(engine.SUB_OUT0_DRAM,
            torch.zeros(N0 * SC, dtype=torch.bfloat16))
        engine.dma_to_accelerator_memory(engine.SUB_PATCH_DRAM, patches0.contiguous())
        engine.program_execute(prog_s0)

        # CPU stages 1 & 2 — reshape with ACTUAL dims, not max
        s0 = read_dram(engine, engine.SUB_OUT0_DRAM, n0_act * SC)
        s0_4d = s0.reshape(h0_act, w0_act, SC).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)
        s1 = F.relu(F.conv2d(F.conv2d(s0_4d,
            sd["encoder.pre_encode.conv.2.weight"].to(torch.bfloat16),
            sd["encoder.pre_encode.conv.2.bias"].to(torch.bfloat16), stride=2, padding=1, groups=SC),
            sd["encoder.pre_encode.conv.3.weight"].to(torch.bfloat16),
            sd["encoder.pre_encode.conv.3.bias"].to(torch.bfloat16)))
        s2 = F.relu(F.conv2d(F.conv2d(s1,
            sd["encoder.pre_encode.conv.5.weight"].to(torch.bfloat16),
            sd["encoder.pre_encode.conv.5.bias"].to(torch.bfloat16), stride=2, padding=1, groups=SC),
            sd["encoder.pre_encode.conv.6.weight"].to(torch.bfloat16),
            sd["encoder.pre_encode.conv.6.bias"].to(torch.bfloat16)))
        _, _, h2_act, w2_act = s2.shape
        flat = s2.squeeze(0).permute(1, 0, 2).reshape(h2_act, SC * w2_act).to(torch.bfloat16).contiguous()
        if h2_act < L_pad:
            fp = torch.zeros(L_pad, 4096, dtype=torch.bfloat16)
            fp[:h2_act, :] = flat
            flat = fp

        # Sub linear
        engine.dma_to_accelerator_memory(engine.SUB_FLAT_DRAM, flat.contiguous())
        engine.dma_to_accelerator_memory(engine.INPUT_DRAM, torch.zeros(L_pad * D, dtype=torch.bfloat16))
        engine.program_execute(prog_lin)

        # Padding rows
        if L_pad > cur_L:
            ef = torch.zeros((L_pad - cur_L) * D, dtype=torch.bfloat16)
            ef[0::2] = 0.1; ef[1::2] = -0.1
            engine.dma_to_accelerator_memory(engine.INPUT_DRAM + cur_L * D * bpe, ef)

        # Update attention mask for this chunk's actual L
        attn_mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
        attn_mask[:cur_L, :cur_L] = 0.0
        if cur_L < L_pad:
            attn_mask[cur_L:, 0] = 0.0
        engine.dma_to_accelerator_memory(engine.ATTN_MASK_DRAM, attn_mask.contiguous())

        # Encoder (single instruction stream)
        engine.start_execute_from_dram(enc_prog_addr)
        engine.wait_queue(120.0)

        # Decoder
        enc_out = read_dram(engine, engine.INPUT_DRAM, L_pad * D)
        engine.dma_to_accelerator_memory(engine.ENC_OUT_DRAM, enc_out.contiguous())
        tokens = engine.run_decode(engine.ENC_OUT_DRAM, cur_L)

    if not tokens:
        return None
    valid = [t for t in tokens if 0 <= t < tokenizer.GetPieceSize()]
    text = tokenizer.DecodeIds(valid)
    return text.strip() if text else None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Parakeet FPGA Live")

@app.get("/", response_class=HTMLResponse)
@app.get("/mic", response_class=HTMLResponse)
async def mic_page():
    return MIC_HTML

async def _broadcast(text, idx):
    msg = json.dumps({"text": text, "segment": idx})
    with transcript_clients_lock:
        dead = []
        for c in transcript_clients:
            try: await c.send_text(msg)
            except: dead.append(c)
        for d in dead:
            transcript_clients.discard(d)

import asyncio

def _transcribe_sync(samples):
    """Run FPGA transcription (blocking). Called from thread pool."""
    waveform = torch.from_numpy(samples).unsqueeze(0)
    return transcribe_chunk(waveform)

async def _process_buffer():
    global audio_total_samples
    with audio_lock:
        if audio_total_samples < CHUNK_SAMPLES:
            return
        chunks = list(audio_buffer)
        audio_buffer.clear()
        audio_total_samples = 0
    samples = np.concatenate(chunks).astype(np.float32)
    # Run in thread pool so we don't block the async event loop
    text = await asyncio.to_thread(_transcribe_sync, samples)
    if text:
        with transcript_lock:
            transcript_segments.append(text)
            idx = len(transcript_segments) - 1
        await _broadcast(text, idx)

async def _flush_buffer():
    global audio_total_samples
    with audio_lock:
        if not audio_buffer: return
        chunks = list(audio_buffer)
        audio_buffer.clear()
        audio_total_samples = 0
    samples = np.concatenate(chunks).astype(np.float32)
    text = await asyncio.to_thread(_transcribe_sync, samples)
    if text:
        with transcript_lock:
            transcript_segments.append(text)
            idx = len(transcript_segments) - 1
        await _broadcast(text, idx)

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    global audio_total_samples
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            n = len(data) // 4
            if n == 0: continue
            arr = np.frombuffer(data[:n*4], dtype=np.float32).copy()
            with audio_lock:
                audio_buffer.append(arr)
                audio_total_samples += len(arr)
            await _process_buffer()
    except WebSocketDisconnect: pass
    except: pass
    finally: await _flush_buffer()

@app.websocket("/ws/transcript")
async def ws_transcript(ws: WebSocket):
    await ws.accept()
    with transcript_clients_lock:
        transcript_clients.add(ws)
    try:
        with transcript_lock:
            existing = list(transcript_segments)
        for i, seg in enumerate(existing):
            await ws.send_text(json.dumps({"text": seg, "segment": i}))
        while True:
            await ws.receive_text()
    except: pass
    finally:
        with transcript_clients_lock:
            transcript_clients.discard(ws)


# ---------------------------------------------------------------------------
# Curses TUI
# ---------------------------------------------------------------------------
def _curses_main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(200)
    if curses.has_colors():
        curses.start_color(); curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)
    msg_q = queue.Queue()
    ws_connected = threading.Event()
    ws_stop = threading.Event()
    def _ws_reader():
        while not ws_stop.is_set():
            try:
                with ws_sync_client.connect(f"ws://127.0.0.1:{server_port}/ws/transcript", close_timeout=1) as conn:
                    ws_connected.set()
                    while not ws_stop.is_set():
                        try:
                            raw = conn.recv(timeout=0.3)
                            msg_q.put(json.loads(raw))
                        except TimeoutError: continue
                        except websockets.exceptions.ConnectionClosed: break
            except:
                ws_connected.clear()
                if ws_stop.is_set(): return
                time.sleep(1)
        ws_connected.clear()
    threading.Thread(target=_ws_reader, daemon=True).start()
    segments = []
    try:
        while True:
            while True:
                try:
                    msg = msg_q.get_nowait()
                    seg = msg.get("segment", len(segments))
                    while len(segments) <= seg: segments.append("")
                    segments[seg] = msg.get("text", "")
                except queue.Empty: break
            try: ch = stdscr.getch()
            except: ch = -1
            if ch in (ord("q"), 27): break
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            if h < 4 or w < 20:
                stdscr.addstr(0, 0, "Terminal too small"); stdscr.refresh(); continue
            title = " Parakeet FPGA Live Transcription "
            try: stdscr.addstr(0, 0, (title + " " * max(0, w - len(title)))[:w],
                               curses.color_pair(1) | curses.A_BOLD)
            except curses.error: pass
            conn_str = "CONNECTED" if ws_connected.is_set() else "connecting..."
            with audio_lock: buf_s = audio_total_samples / SAMPLE_RATE
            status = f" [{conn_str}]  segments: {len(segments)}  buffer: {buf_s:.1f}s  |  q/Esc: quit "
            try: stdscr.addstr(h-1, 0, (status + " " * max(0, w-len(status)))[:w],
                               curses.color_pair(2 if ws_connected.is_set() else 3) | curses.A_BOLD)
            except curses.error: pass
            body_top, body_h = 2, h - 3
            if body_h <= 0: stdscr.refresh(); continue
            wrapped = []
            for i, seg in enumerate(segments):
                pfx = f"[{i}] "
                ww = max(10, w - len(pfx))
                lines = textwrap.wrap(seg, ww) if seg else [""]
                for j, ln in enumerate(lines):
                    wrapped.append((pfx if j == 0 else " " * len(pfx)) + ln)
            if not wrapped:
                try: stdscr.addstr(body_top, 1, "Waiting for audio... Open localhost:{}/mic".format(server_port)[:w-2],
                                   curses.color_pair(3))
                except curses.error: pass
            else:
                start = max(0, len(wrapped) - body_h)
                for ri, ln in enumerate(wrapped[start:start+body_h]):
                    try: stdscr.addstr(body_top + ri, 0, ln[:w], curses.color_pair(4))
                    except curses.error: pass
            stdscr.refresh()
    finally:
        ws_stop.set()


# ---------------------------------------------------------------------------
# Startup & main
# ---------------------------------------------------------------------------
server_port = 8000

def main():
    global engine, cfg, sd, tokenizer, enc_prog_addr, prog_s0, prog_lin
    global L_pad, L, H2, H0, W0, N0, SC, D, FF, bpe, H_heads, dk, P_pad
    global toeplitz_addrs, server_port

    import argparse
    parser = argparse.ArgumentParser(description="Parakeet FPGA live transcription")
    parser.add_argument("--dev", default="xdma0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server_port = args.port

    print("=" * 60)
    print("  Parakeet FPGA Live Transcription — initializing")
    print("=" * 60)

    cfg = load_config()
    set_dma_device(args.dev)

    # Fixed L_pad for streaming
    L_pad = 192
    n_mels = cfg["encoder"]["n_mels"]

    # Init engine
    engine_obj = Parakeet_UnifiedEngine()
    engine_obj.weight_init()
    engine = engine_obj
    sd = engine._ckpt_sd

    # Dimensions
    SC = engine.sub_channels
    D = engine.d_model
    bpe = engine.bytes_per_element
    FF = engine.ff_dim
    H_heads = engine.num_heads
    dk = engine.head_dim

    # Subsampling dims for max audio that fits L_pad
    # For L_pad=192, max T_mel ≈ 192*8 = 1536
    T_mel_max = L_pad * 8
    H0 = conv2d_outsize(T_mel_max)
    W0 = conv2d_outsize(n_mels)
    N0 = H0 * W0
    H1 = conv2d_outsize(H0)
    H2_max = conv2d_outsize(conv2d_outsize(H1))
    H2 = H2_max
    L = H2

    # Tensor init
    engine.tensor_init(L_pad)

    # Compile subsampling
    prog_s0, _ = engine.compile_sub_stage0(N0, 64, SC)
    prog_lin, _ = engine.compile_sub_linear(L_pad)

    # Setup pos embedding + attention mask
    P_pad = pad_to_multiple(2 * L_pad - 1, engine.block_size)
    rel_pe = engine.make_rel_pos_emb(L_pad)
    if rel_pe.shape[0] < P_pad:
        pe_padded = torch.zeros(P_pad, D, dtype=torch.bfloat16)
        pe_padded[:rel_pe.shape[0], :] = rel_pe
        rel_pe = pe_padded
    engine.dma_to_accelerator_memory(engine.POS_EMB_DRAM, rel_pe.contiguous())
    mask = torch.full((L_pad, L_pad), -1e38, dtype=torch.bfloat16)
    mask[:L_pad, :L_pad] = 0.0  # will be re-set per chunk with actual L
    engine.dma_to_accelerator_memory(engine.ATTN_MASK_DRAM, mask.contiguous())

    # Stage Toeplitz matrices
    print("  Staging Toeplitz DW conv matrices...")
    toeplitz_addrs = []
    for li in range(engine.num_layers):
        la_i = engine.layer_addrs[li]
        k_flat = torch.zeros(D * 64, dtype=torch.bfloat16)
        engine.dma_read(DMA_DEVICE_C2H, la_i["CONV_DW_W"], k_flat, D * 64 * bpe)
        kernel = k_flat.reshape(D, 64)
        toeplitz = torch.zeros(D, L_pad, L_pad, dtype=torch.bfloat16)
        for k in range(9):
            offset = k - 4
            t_idx = torch.arange(max(0, -offset), min(L_pad, L_pad - offset))
            toeplitz[:, t_idx, t_idx + offset] = kernel[:, k:k+1].expand(-1, len(t_idx))
        addr = engine.get_params_dram_addr()
        engine.allocate_params_dram(D * L_pad * L_pad * bpe)
        engine.dma_to_accelerator_memory(addr, toeplitz.reshape(-1).contiguous())
        toeplitz_addrs.append(addr)
    print(f"  Toeplitz staged: {engine.get_params_dram_usage()/1024**2:.0f} MB")

    # Compile encoder (single instruction stream)
    print("  Compiling 24-layer encoder...")
    def emit_all_24_layers():
        for layer_idx in range(engine.num_layers):
            la = engine.layer_addrs[layer_idx]
            t_addr = toeplitz_addrs[layer_idx]
            # FF1
            engine.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=engine.INPUT_DRAM, OUTPUT_DRAM_ADDR=engine.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_FF1_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF1_BIAS"])
            engine.matmat_mul_core(M=L_pad, K=D, N=FF,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["FF1_W1"],
                OUTPUT_DRAM_ADDR=engine.FF_MID_DRAM, silu_enable=True)
            engine.matmat_mul_core(M=L_pad, K=FF, N=D,
                A_DRAM_ADDR=engine.FF_MID_DRAM, B_DRAM_ADDR=la["FF1_W2"],
                OUTPUT_DRAM_ADDR=engine.FF_OUT_DRAM)
            half_step_residual_core_dram(engine, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=engine.INPUT_DRAM, FF_DRAM_ADDR=engine.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=engine.INPUT_DRAM)
            # Attention
            engine.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=engine.INPUT_DRAM, OUTPUT_DRAM_ADDR=engine.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_ATTN_WEIGHT"], BETA_DRAM_ADDR=la["LN_ATTN_BIAS"])
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_Q_W"], OUTPUT_DRAM_ADDR=engine.Q_DRAM)
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_K_W"], OUTPUT_DRAM_ADDR=engine.K_DRAM)
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_V_W"], OUTPUT_DRAM_ADDR=engine.V_DRAM)
            engine.matmat_mul_core(M=P_pad, K=D, N=D,
                A_DRAM_ADDR=engine.POS_EMB_DRAM, B_DRAM_ADDR=la["ATTN_POS_W"],
                OUTPUT_DRAM_ADDR=engine.POS_PROJ_DRAM)
            inv_sqrt_dk = 1.0 / math.sqrt(dk)
            for h in range(H_heads):
                h_off = h * dk * bpe
                engine.accelerator_memory_to_sram(engine.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                engine.accelerator_memory_to_sram(la["ATTN_BIAS_U"] + h_off, URAM_B_BASE, dk)
                for row in range(L_pad):
                    engine.eltwise_add_core(URAM_A_BASE + row * dk * bpe, URAM_B_BASE,
                        URAM_A_BASE + row * dk * bpe, dk)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.CONV_A_DRAM, L_pad * dk)
                engine.accelerator_memory_to_sram(engine.K_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.CONV_B_DRAM, L_pad * dk)
                engine.matmat_mul_core(M=L_pad, K=dk, N=L_pad,
                    A_DRAM_ADDR=engine.CONV_A_DRAM, B_DRAM_ADDR=engine.CONV_B_DRAM,
                    OUTPUT_DRAM_ADDR=engine.SCORE_DRAM)
                engine.accelerator_memory_to_sram(engine.Q_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                engine.accelerator_memory_to_sram(la["ATTN_BIAS_V"] + h_off, URAM_B_BASE, dk)
                for row in range(L_pad):
                    engine.eltwise_add_core(URAM_A_BASE + row * dk * bpe, URAM_B_BASE,
                        URAM_A_BASE + row * dk * bpe, dk)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.CONV_A_DRAM, L_pad * dk)
                engine.accelerator_memory_to_sram(engine.POS_PROJ_DRAM + h_off, URAM_A_BASE, P_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.CONV_B_DRAM, P_pad * dk)
                engine.matmat_mul_core(M=L_pad, K=dk, N=P_pad,
                    A_DRAM_ADDR=engine.CONV_A_DRAM, B_DRAM_ADDR=engine.CONV_B_DRAM,
                    OUTPUT_DRAM_ADDR=engine.CONV_OUT_DRAM)
                rel_shift_core_dram(engine, L=L_pad,
                    INPUT_DRAM_ADDR=engine.CONV_OUT_DRAM, OUTPUT_DRAM_ADDR=engine.REL_SHIFT_DRAM,
                    input_row_stride=P_pad)
                score_elems = L_pad * L_pad
                engine.accelerator_memory_to_sram(engine.SCORE_DRAM, URAM_A_BASE, score_elems)
                engine.accelerator_memory_to_sram(engine.REL_SHIFT_DRAM, URAM_B_BASE, score_elems)
                engine.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, score_elems)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.SCORE_DRAM, score_elems)
                engine.accelerator_memory_to_sram(engine.SCORE_DRAM, URAM_A_BASE, score_elems)
                for row in range(L_pad):
                    engine.broadcast_mul(scalar=inv_sqrt_dk,
                        sram_start_addr=URAM_A_BASE + row * L_pad * bpe,
                        sram_wb_addr=URAM_A_BASE + row * L_pad * bpe, element_size=L_pad)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.SCORE_DRAM, score_elems)
                engine.matmat_mul_core(M=L_pad, K=L_pad, N=L_pad,
                    A_DRAM_ADDR=engine.SCORE_DRAM, B_DRAM_ADDR=engine.IDENTITY_LPAD_DRAM,
                    OUTPUT_DRAM_ADDR=engine.SCORE_DRAM, softmax_enable=True,
                    C_DRAM_ADDR=engine.ATTN_MASK_DRAM, bias_mode="full_matrix")
                engine.accelerator_memory_to_sram(engine.V_DRAM + h_off, URAM_A_BASE, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.CONV_B_DRAM, L_pad * dk)
                chunked_transpose_core_dram(engine, M=L_pad, N=dk,
                    input_dram_addr=engine.CONV_B_DRAM, output_dram_addr=engine.ATTN_VT_DRAM,
                    identity_dram_addr=engine.w["IDENTITY_64"],
                    temp_dram_addr=engine.PERMUTE_TEMP_DRAM)
                engine.matmat_mul_core(M=L_pad, K=L_pad, N=dk,
                    A_DRAM_ADDR=engine.SCORE_DRAM, B_DRAM_ADDR=engine.ATTN_VT_DRAM,
                    OUTPUT_DRAM_ADDR=engine.CONV_A_DRAM)
                engine.accelerator_memory_to_sram(engine.CONV_A_DRAM, URAM_A_BASE, L_pad * dk)
                engine.sram_to_accelerator_memory(URAM_A_BASE, engine.ATTN_OUT_DRAM + h_off, L_pad * dk,
                    stride_bytes_per_chunk=dk * bpe, stride_jump_bytes=D * bpe)
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.ATTN_OUT_DRAM, B_DRAM_ADDR=la["ATTN_OUT_W"],
                OUTPUT_DRAM_ADDR=engine.FF_OUT_DRAM)
            engine.accelerator_memory_to_sram(engine.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            engine.accelerator_memory_to_sram(engine.FF_OUT_DRAM, URAM_B_BASE, L_pad * D)
            engine.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            engine.sram_to_accelerator_memory(URAM_A_BASE, engine.INPUT_DRAM, L_pad * D)
            # ConvModule
            engine.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=engine.INPUT_DRAM, OUTPUT_DRAM_ADDR=engine.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_CONV_WEIGHT"], BETA_DRAM_ADDR=la["LN_CONV_BIAS"])
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1A_W"],
                OUTPUT_DRAM_ADDR=engine.CONV_A_DRAM)
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["CONV_PW1B_W"],
                OUTPUT_DRAM_ADDR=engine.CONV_B_DRAM)
            glu_core_dram(engine, M=L_pad, C=D,
                A_DRAM_ADDR=engine.CONV_A_DRAM, B_DRAM_ADDR=engine.CONV_B_DRAM,
                OUTPUT_DRAM_ADDR=engine.CONV_A_DRAM, IDENTITY_DRAM_ADDR=engine.w["IDENTITY_1024"])
            chunked_transpose_core_dram(engine, M=L_pad, N=D,
                input_dram_addr=engine.CONV_A_DRAM, output_dram_addr=engine.CONV_T_DRAM,
                identity_dram_addr=engine.w["IDENTITY_64"],
                temp_dram_addr=engine.PERMUTE_TEMP_DRAM)
            for ch in range(D):
                engine.matmat_mul_core(M=1, K=L_pad, N=L_pad,
                    A_DRAM_ADDR=engine.CONV_T_DRAM + ch * L_pad * bpe,
                    B_DRAM_ADDR=t_addr + ch * L_pad * L_pad * bpe,
                    OUTPUT_DRAM_ADDR=engine.CONV_DW_DRAM + ch * L_pad * bpe)
            batch_norm_core_dram(engine, C=D, L=L_pad,
                A_DRAM_ADDR=engine.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=engine.CONV_DW_DRAM,
                SCALE_DRAM_ADDR=la["CONV_BN_SCALE"], SHIFT_DRAM_ADDR=la["CONV_BN_SHIFT"])
            silu_core_dram(engine, M=D, N=L_pad,
                A_DRAM_ADDR=engine.CONV_DW_DRAM, OUTPUT_DRAM_ADDR=engine.CONV_OUT_DRAM,
                IDENTITY_DRAM_ADDR=engine.IDENTITY_LPAD_DRAM)
            chunked_transpose_core_dram(engine, M=D, N=L_pad,
                input_dram_addr=engine.CONV_OUT_DRAM, output_dram_addr=engine.CONV_T_DRAM,
                identity_dram_addr=engine.w["IDENTITY_64"],
                temp_dram_addr=engine.PERMUTE_TEMP_DRAM)
            engine.matmat_mul_core(M=L_pad, K=D, N=D,
                A_DRAM_ADDR=engine.CONV_T_DRAM, B_DRAM_ADDR=la["CONV_PW2_W"],
                OUTPUT_DRAM_ADDR=engine.CONV_OUT_DRAM)
            engine.accelerator_memory_to_sram(engine.INPUT_DRAM, URAM_A_BASE, L_pad * D)
            engine.accelerator_memory_to_sram(engine.CONV_OUT_DRAM, URAM_B_BASE, L_pad * D)
            engine.eltwise_add_core(URAM_A_BASE, URAM_B_BASE, URAM_A_BASE, L_pad * D)
            engine.sram_to_accelerator_memory(URAM_A_BASE, engine.INPUT_DRAM, L_pad * D)
            # FF2
            engine.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=engine.INPUT_DRAM, OUTPUT_DRAM_ADDR=engine.LN_OUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_FF2_WEIGHT"], BETA_DRAM_ADDR=la["LN_FF2_BIAS"])
            engine.matmat_mul_core(M=L_pad, K=D, N=FF,
                A_DRAM_ADDR=engine.LN_OUT_DRAM, B_DRAM_ADDR=la["FF2_W1"],
                OUTPUT_DRAM_ADDR=engine.FF_MID_DRAM, silu_enable=True)
            engine.matmat_mul_core(M=L_pad, K=FF, N=D,
                A_DRAM_ADDR=engine.FF_MID_DRAM, B_DRAM_ADDR=la["FF2_W2"],
                OUTPUT_DRAM_ADDR=engine.FF_OUT_DRAM)
            half_step_residual_core_dram(engine, M=L_pad, N=D,
                RESIDUAL_DRAM_ADDR=engine.INPUT_DRAM, FF_DRAM_ADDR=engine.FF_OUT_DRAM,
                OUTPUT_DRAM_ADDR=engine.INPUT_DRAM)
            # Final LN
            engine.layer_norm_core_dram(M=L_pad, N=D,
                A_DRAM_ADDR=engine.INPUT_DRAM, OUTPUT_DRAM_ADDR=engine.INPUT_DRAM,
                GAMMA_DRAM_ADDR=la["LN_OUT_WEIGHT"], BETA_DRAM_ADDR=la["LN_OUT_BIAS"])
            print(f"    Layer {layer_idx:2d} emitted")

    enc_prog_addr = engine.get_program_dram_addr()
    compile_and_run(engine, emit_all_24_layers)
    print(f"  Encoder compiled: single instruction stream")

    # Compile decoder
    pred_prog, tok_prog, dur_prog, _ = engine.compile_decoder()
    engine.progs = {"pred": (pred_prog, 0), "joint_tok": (tok_prog, 0), "joint_dur": (dur_prog, 0)}
    print(f"  Decoder compiled")

    # Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(TOKENIZER_PATH)

    print(f"\n  Ready! Starting server on 0.0.0.0:{server_port}")
    print(f"  Open http://localhost:{server_port}/mic in your browser")
    print(f"  Press Enter to launch TUI...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted.")
        return

    # Start server
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=server_port,
                                                 log_level="warning"), daemon=True).start()
    time.sleep(0.5)

    # TUI
    try:
        curses.wrapper(_curses_main)
    except KeyboardInterrupt:
        pass
    print("\nExiting.")


if __name__ == "__main__":
    main()
