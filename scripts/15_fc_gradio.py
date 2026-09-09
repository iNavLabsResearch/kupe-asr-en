#!/usr/bin/env python
"""Gradio inference app for the FastConformer + Nandi-Mini ASR model.

Loads the TRAINED model (FastConformer encoder + projector + Nandi decoder) and
serves two modes:

  1. File / record  — upload or record a clip -> transcript + timing/throughput
                       stats (RTFx, latency, words/s), plus optional WER/CER if
                       you paste a reference.
  2. Real-time      — stream from the mic; a background ENCODER thread turns audio
                       chunks into FastConformer features while the DECODER thread
                       runs Nandi on the growing feature buffer -> live transcript.
                       Encoder and SLM run in parallel (producer/consumer), so the
                       SLM is never blocked waiting on the encoder and vice-versa.

Usage:
    python scripts/15_fc_gradio.py                       # pulls best model from the Hub
    python scripts/15_fc_gradio.py --model-dir artifacts/runs/<run>/model
    python scripts/15_fc_gradio.py --share               # public gradio link

Needs: gradio (pip install gradio), and the same env as training (NeMo etc.).
"""
import os
import _bootstrap  # noqa: F401
import argparse
import threading
import time
from collections import deque

import numpy as np
import torch

from kupe_asr_en.config import load_config
from kupe_asr_en.constants import FC_SAMPLE_RATE
from kupe_asr_en.env import hf_login, log, require_token
from kupe_asr_en.modeling.fc_asr_model import FastConformerASR
from kupe_asr_en.text import normalize

_DEFAULT_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "configs", "fastconformer_nandi.yaml")

MODEL = None          # (FastConformerASR, tokenizer, device, cfg) — loaded once
_GEN_LOCK = threading.Lock()   # the HF decoder isn't reentrant across threads


# ----------------------------------------------------------------------- loading
def load_model(cfg, model_dir=None):
    global MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"
    if model_dir is None:
        from huggingface_hub import snapshot_download
        hf_login()
        log.info("downloading best model from %s ...", cfg.repos.model)
        model_dir = snapshot_download(cfg.repos.model, token=require_token())
    log.info("loading FastConformerASR from %s (device=%s) ...", model_dir, device)
    model, tok = FastConformerASR.load(model_dir, device=device, dtype=dtype, with_encoder=True)
    model.eval()
    MODEL = (model, tok, device, cfg)
    log.info("model ready.")
    return MODEL


# ----------------------------------------------------------------------- helpers
def _to_16k_mono(sr, data):
    """gradio audio (sr, np array) -> float32 mono @16k in [-1,1]."""
    a = np.asarray(data)
    if a.dtype.kind in "iu":                       # int PCM -> float
        a = a.astype(np.float32) / np.iinfo(a.dtype).max
    else:
        a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != FC_SAMPLE_RATE:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=FC_SAMPLE_RATE, res_type="soxr_hq")
    return np.ascontiguousarray(a, dtype=np.float32)


@torch.inference_mode()
def _encode(model, device, wave16k):
    """float32 [S] @16k -> (feats [1,T,512], num_frames [1]) via the FastConformer encoder."""
    wt = torch.from_numpy(wave16k)[None].to(device)
    wl = torch.tensor([wave16k.shape[0]], dtype=torch.long, device=device)
    ac = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with ac:
        feats, flen = model.encoder.features(wt, wl)
    return feats, flen


@torch.inference_mode()
def _decode(model, tok, cfg, feats, num_frames, max_new_tokens=200):
    """feats -> text. Uses the same greedy path as eval, with anti-repetition."""
    prefixes = model.build_prefix(feats=feats, num_frames=num_frames)
    L = prefixes[0].shape[0]
    inp = prefixes[0][None].to(model._dtype)
    attn = torch.ones(1, L, dtype=torch.long, device=model.device)
    with _GEN_LOCK:
        gen = model.lm.generate(
            inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=model.eos_id,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=float(getattr(cfg.stream, "repetition_penalty", 1.3)),
            no_repeat_ngram_size=int(getattr(cfg.stream, "no_repeat_ngram_size", 3)))
    ids = gen[0].tolist()
    if model.eos_id in ids:
        ids = ids[: ids.index(model.eos_id)]
    return normalize(tok.decode(ids, skip_special_tokens=True)), len(ids)


# ----------------------------------------------------------------------- file mode
def transcribe_file(audio, reference):
    if MODEL is None or audio is None:
        return "", "_load a model and provide audio_"
    model, tok, device, cfg = MODEL
    sr, data = audio
    wave = _to_16k_mono(sr, data)
    dur = len(wave) / FC_SAMPLE_RATE
    if dur < 0.1:
        return "", "_clip too short_"

    t0 = time.time()
    feats, flen = _encode(model, device, wave)
    t_enc = time.time()
    text, ntok = _decode(model, tok, cfg, feats, flen)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    enc_s, dec_s, proc_s = t_enc - t0, t1 - t_enc, t1 - t0
    words = len(text.split())
    stats = [
        f"| metric | value |", "|---|---|",
        f"| audio duration | {dur:.2f} s |",
        f"| total latency | {proc_s*1000:.0f} ms |",
        f"| ├─ encoder | {enc_s*1000:.0f} ms |",
        f"| └─ SLM decode | {dec_s*1000:.0f} ms |",
        f"| **RTFx (audio/proc)** | **{dur/proc_s:.0f}×** |",
        f"| RTF (proc/audio) | {proc_s/dur:.4f} |",
        f"| words / chars / tokens | {words} / {len(text)} / {ntok} |",
        f"| throughput | {words/proc_s:.0f} words/s |",
    ]
    if reference and reference.strip():
        import jiwer
        ref = normalize(reference)
        stats += [f"| **WER vs reference** | **{jiwer.wer(ref, text)*100:.1f}%** |",
                  f"| CER vs reference | {jiwer.cer(ref, text)*100:.1f}% |"]
    return text, "\n".join(stats)


# ----------------------------------------------------------------------- realtime mode
class LiveSession:
    """Parallel encoder/decoder pipeline for streaming mic audio.

    Producer (encoder thread): pulls raw audio chunks off `audio_q`, encodes the
    trailing `max_context_s` window to FastConformer features, publishes them.
    Consumer (decoder thread): whenever fresh features exist, runs Nandi on them
    and updates `self.text`. The two run concurrently, so the SLM decodes the
    previous window while the encoder is already building the next one.
    """

    def __init__(self, model, tok, device, cfg):
        self.model, self.tok, self.device, self.cfg = model, tok, device, cfg
        self.max_ctx = int(float(getattr(cfg.stream, "max_context_s", 30)) * FC_SAMPLE_RATE)
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_lock = threading.Lock()
        self.feats = None
        self.feats_lock = threading.Lock()
        self.feats_ready = threading.Event()
        self.text = ""
        self.rtf_line = ""
        self.stop = threading.Event()
        self.enc_t = threading.Thread(target=self._encoder_loop, daemon=True)
        self.dec_t = threading.Thread(target=self._decoder_loop, daemon=True)
        self.enc_t.start(); self.dec_t.start()

    def add_audio(self, wave16k):
        with self.buf_lock:
            self.buf = np.concatenate([self.buf, wave16k])[-self.max_ctx:]

    def _encoder_loop(self):
        while not self.stop.is_set():
            time.sleep(0.25)                       # ~4 encodes/s of the trailing window
            with self.buf_lock:
                w = self.buf.copy()
            if w.shape[0] < FC_SAMPLE_RATE // 2:   # <0.5s, nothing yet
                continue
            try:
                t0 = time.time()
                feats, flen = _encode(self.model, self.device, w)
                enc_ms = (time.time() - t0) * 1000
                with self.feats_lock:
                    self.feats = (feats, flen, w.shape[0] / FC_SAMPLE_RATE, enc_ms)
                self.feats_ready.set()
            except Exception as e:
                log.warning("encoder loop: %s", e)

    def _decoder_loop(self):
        while not self.stop.is_set():
            if not self.feats_ready.wait(timeout=0.5):
                continue
            self.feats_ready.clear()
            with self.feats_lock:
                pack = self.feats
            if pack is None:
                continue
            feats, flen, dur, enc_ms = pack
            try:
                t0 = time.time()
                text, _ = _decode(self.model, self.tok, self.cfg, feats, flen)
                dec_ms = (time.time() - t0) * 1000
                self.text = text
                self.rtf_line = (f"window {dur:.1f}s | enc {enc_ms:.0f}ms + dec {dec_ms:.0f}ms "
                                 f"| RTFx {dur/((enc_ms+dec_ms)/1000):.0f}×")
            except Exception as e:
                log.warning("decoder loop: %s", e)

    def close(self):
        self.stop.set()


def stream_step(new_chunk, state):
    if MODEL is None:
        return "load a model first", "", state
    model, tok, device, cfg = MODEL
    if state is None:
        state = LiveSession(model, tok, device, cfg)
    if new_chunk is not None:
        sr, data = new_chunk
        try:
            state.add_audio(_to_16k_mono(sr, data))
        except Exception as e:
            log.warning("stream ingest: %s", e)
    return state.text or "…", state.rtf_line, state


def stream_reset(state):
    if isinstance(state, LiveSession):
        state.close()
    return "", "", None


# ----------------------------------------------------------------------- UI
def build_ui():
    import gradio as gr
    with gr.Blocks(title="Kupe FastConformer + Nandi ASR") as demo:
        gr.Markdown("# Kupe ASR — FastConformer + Nandi-Mini-150M\n"
                    "FastConformer encoder → projector → Nandi SLM. Trained model, real-time inference.")

        with gr.Tab("File / Record"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio")
                    ref = gr.Textbox(label="Reference transcript (optional — enables WER/CER)",
                                     lines=2, placeholder="paste the ground-truth text to score WER")
                    btn = gr.Button("Transcribe", variant="primary")
                with gr.Column():
                    out_text = gr.Textbox(label="Transcript", lines=6)
                    out_stats = gr.Markdown(label="Stats")
            btn.click(transcribe_file, inputs=[audio, ref], outputs=[out_text, out_stats])

        with gr.Tab("Real-time (mic)"):
            gr.Markdown("Speak — the **encoder** and **SLM** run in parallel threads; the "
                        "transcript refreshes ~4×/s over the trailing context window.")
            st = gr.State(None)
            mic = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Live mic")
            live_text = gr.Textbox(label="Live transcript", lines=4)
            live_rtf = gr.Markdown()
            clear = gr.Button("Reset session")
            mic.stream(stream_step, inputs=[mic, st], outputs=[live_text, live_rtf, st])
            clear.click(stream_reset, inputs=[st], outputs=[live_text, live_rtf, st])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_DEFAULT_CFG)
    ap.add_argument("--model-dir", default=None, help="local run model dir; default pulls repos.model from Hub")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_model(cfg, args.model_dir)
    demo = build_ui()
    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
