#!/usr/bin/env python
"""Gradio demo for Kupe-SLM-EN ASR — runs on Kaggle (T4×2 or P100×1), no token needed.

Two modes:
  • Upload a file            -> transcribe the whole clip (long files auto-windowed,
                                windows fanned out across GPUs), reports Real-Time Factor.
  • Live mic (streaming)     -> browser mic streams to the server over Gradio's websocket;
                                we keep a rolling context buffer and re-decode it as speech
                                arrives, showing the transcript grow live + per-update
                                latency and RTF.

Audio is always normalised to 24 kHz mono float32 (Mimi's rate) regardless of the
browser/file sample rate, so capture is correct on any device.

Run on Kaggle (Internet ON, GPU T4×2 or P100):
    !git clone https://github.com/iNavLabsResearch/kupe-asr-en.git
    %cd kupe-asr-en
    !pip -q install "transformers==5.4.0" "huggingface_hub>=0.34" gradio librosa soundfile soxr
    !python scripts/10_gradio_kaggle.py           # prints a public *.gradio.live URL
"""
import _bootstrap  # noqa: F401
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from kupe_asr_en.config import load_config
from kupe_asr_en.constants import MIMI_SAMPLE_RATE, MIMI_FRAME_RATE
from kupe_asr_en.env import log
from kupe_asr_en.modeling.asr_model import LummaASR

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = "anuj-inavlabs/kupe-asr-en"
MAX_FRAMES = 750                      # model's max audio frames (=60s @12.5Hz); keep windows under this
WINDOW_S = 28.0                      # file mode: window length for long clips
CONTEXT_S = 24.0                    # live mode: rolling context kept for each decode
MIN_INFER_S = 1.0                   # live mode: only re-decode after this much NEW audio


# ----------------------------------------------------------------- audio utils
def to_24k_mono(sr, data) -> np.ndarray:
    """Any (sr, ndarray int16/float) -> float32 mono @ 24 kHz in [-1,1]."""
    x = np.asarray(data)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:                                  # stereo -> mono
        x = x.mean(axis=1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:                                  # guard odd scaling
        x = x / peak
    if sr and sr != MIMI_SAMPLE_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=MIMI_SAMPLE_RATE)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_file_24k(path) -> np.ndarray:
    import librosa
    x, _ = librosa.load(path, sr=MIMI_SAMPLE_RATE, mono=True)
    return np.ascontiguousarray(x, dtype=np.float32)


# ----------------------------------------------------------------- engine (multi-GPU pool)
class Engine:
    def __init__(self, model_dir, codebooks, max_new_tokens, rep_pen, no_ngram):
        from transformers import MimiModel
        if torch.cuda.is_available():
            self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            self.devices = ["cpu"]
        self.codebooks = int(codebooks)
        self.max_new_tokens = int(max_new_tokens)
        self.rep_pen = float(rep_pen)
        self.no_ngram = int(no_ngram)
        self.models, self.mimis, self.toks = [], [], []
        for d in self.devices:
            dt = "float16" if d.startswith("cuda") else "float32"
            m, tok = LummaASR.load(model_dir, device=d, dtype=dt)
            self.models.append(m); self.toks.append(tok)
            self.mimis.append(MimiModel.from_pretrained("kyutai/mimi").to(d).eval())
        self._locks = [threading.Lock() for _ in self.devices]
        self._pool = ThreadPoolExecutor(max_workers=len(self.devices))
        log.info("Engine ready on %s", self.devices)

    @torch.inference_mode()
    def _decode(self, gi: int, audio24k: np.ndarray) -> str:
        """Transcribe one <=60s window on GPU `gi`."""
        if audio24k.size == 0:
            return ""
        with self._locks[gi]:
            dev = self.devices[gi]
            m, mimi, tok = self.models[gi], self.mimis[gi], self.toks[gi]
            iv = torch.from_numpy(audio24k).view(1, 1, -1).to(dev)
            codes = mimi.encode(iv, num_quantizers=self.codebooks).audio_codes[0]  # [K,T]
            codes = codes[:, :MAX_FRAMES].to("cpu")
            nf = torch.tensor([codes.shape[1]], dtype=torch.long)
            prefix = m.build_prefix(codes[None], nf)[0]
            gen = m.lumma.generate(
                inputs_embeds=prefix[None].to(dev),
                attention_mask=torch.ones(1, prefix.shape[0], dtype=torch.long, device=dev),
                max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                eos_token_id=m.eos_id, pad_token_id=tok.pad_token_id,
                repetition_penalty=self.rep_pen, no_repeat_ngram_size=self.no_ngram)
            ids = gen[0].tolist()
            if m.eos_id in ids:
                ids = ids[: ids.index(m.eos_id)]
            return tok.decode(ids, skip_special_tokens=True).strip()

    def transcribe_full(self, audio24k: np.ndarray):
        """Whole-file transcription. Long clips are windowed and fanned across GPUs."""
        dur = len(audio24k) / MIMI_SAMPLE_RATE
        win = int(WINDOW_S * MIMI_SAMPLE_RATE)
        windows = [audio24k[i:i + win] for i in range(0, len(audio24k), win)] or [audio24k]
        t0 = time.time()
        # round-robin windows across GPUs, decode in parallel
        futs = [self._pool.submit(self._decode, i % len(self.devices), w)
                for i, w in enumerate(windows)]
        parts = [f.result() for f in futs]
        compute = time.time() - t0
        text = " ".join(p for p in parts if p).strip()
        rtf = compute / max(1e-6, dur)
        return text, dur, compute, rtf

    def transcribe_live(self, audio24k: np.ndarray):
        """Decode the current rolling context on GPU 0; return text + timing."""
        ctx = audio24k[-int(CONTEXT_S * MIMI_SAMPLE_RATE):]
        t0 = time.time()
        text = self._decode(0, ctx)
        compute = time.time() - t0
        rtf = compute / max(1e-6, len(ctx) / MIMI_SAMPLE_RATE)
        return text, compute, rtf


# ----------------------------------------------------------------- gradio app
def build_app(engine: Engine):
    import gradio as gr

    def do_file(path):
        if not path:
            return "Upload an audio file first.", ""
        audio = load_file_24k(path)
        text, dur, compute, rtf = engine.transcribe_full(audio)
        stats = (f"audio {dur:.1f}s · compute {compute:.1f}s · "
                 f"RTF {rtf:.2f}× ({1/rtf:.1f}× realtime) · GPUs {len(engine.devices)}")
        return text or "(no speech detected)", stats

    def live_reset():
        return {"buf": np.zeros(0, np.float32), "since": 0.0, "text": ""}, "", ""

    def do_live(stream_chunk, state):
        if state is None:
            state = {"buf": np.zeros(0, np.float32), "since": 0.0, "text": ""}
        if stream_chunk is None:
            return state, state.get("text", ""), ""
        sr, data = stream_chunk
        chunk = to_24k_mono(sr, data)
        state["buf"] = np.concatenate([state["buf"], chunk])
        state["since"] += len(chunk) / MIMI_SAMPLE_RATE
        # only re-decode once enough new audio arrived (keeps it responsive, not thrashing)
        if state["since"] < MIN_INFER_S:
            return state, state.get("text", ""), ""
        state["since"] = 0.0
        text, compute, rtf = engine.transcribe_live(state["buf"])
        state["text"] = text
        stats = f"latency {compute*1000:.0f} ms · RTF {rtf:.2f}× · heard {len(state['buf'])/MIMI_SAMPLE_RATE:.1f}s"
        return state, text, stats

    with gr.Blocks(title="Kupe-SLM-EN ASR") as demo:
        gr.Markdown("# 🗣️ Kupe-SLM-EN — English ASR (Lumma-0.6B + Mimi)\n"
                    f"Model: `{REPO}` · running on **{len(engine.devices)}× {engine.devices[0]}**")
        with gr.Tab("📁 Upload or record a file"):
            f_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                            label="Upload an audio file  —  or click the mic to record")
            f_btn = gr.Button("Transcribe", variant="primary")
            f_out = gr.Textbox(label="Transcript", lines=4)
            f_stats = gr.Textbox(label="Stats (RTF / compute)", lines=1)
            f_btn.click(do_file, inputs=f_in, outputs=[f_out, f_stats])
        with gr.Tab("🎤 Live mic (realtime)"):
            gr.Markdown("Click record and speak — transcript updates as you talk.")
            st = gr.State(None)
            m_in = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            m_out = gr.Textbox(label="Live transcript", lines=4)
            m_stats = gr.Textbox(label="Latency / RTF", lines=1)
            m_in.stream(do_live, inputs=[m_in, st], outputs=[st, m_out, m_stats],
                        time_limit=None, stream_every=0.5)
            m_in.start_recording(live_reset, outputs=[st, m_out, m_stats])
    return demo


def main():
    from huggingface_hub import snapshot_download
    cfg = load_config()
    log.info("downloading %s …", REPO)
    model_dir = snapshot_download(REPO, repo_type="model")   # public: no token needed
    engine = Engine(model_dir, codebooks=cfg.audio.codebooks,
                    max_new_tokens=cfg.eval.max_new_tokens,
                    rep_pen=cfg.stream.repetition_penalty,
                    no_ngram=cfg.stream.no_repeat_ngram_size)
    demo = build_app(engine)
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
