#!/usr/bin/env python
"""Kaggle Gradio demo for anuj-inavlabs/kupe-asr-en (public weights, no HF token).

Two decode modes:
  • Full file     — transcribe the whole upload in one (or windowed) pass
  • Realtime      — feed the file in chunks as if it were live audio; the
                    transcript grows hop by hop (PARTIAL / PRE_HIT_LLM / EOS)

Multi-GPU (Kaggle T4×2): pipeline-parallel
  cuda:0  Mimi encoder
  cuda:1  Kupe-LM (LummaASR)
Single GPU (P100×1 / T4×1): both on cuda:0
CPU fallback if no CUDA.

Realtime factor (RTF) = wall_clock / audio_duration   ( <1 means faster than live )

================================================================================
Kaggle (GPU T4×2 or P100×1, Internet ON)
================================================================================

    !git clone https://github.com/iNavLabsResearch/kupe-asr-en.git
    %cd kupe-asr-en

    # Do NOT reinstall torch — Kaggle GPU images already ship CUDA torch.
    !pip install -q "transformers==5.4.0" "tokenizers>=0.22" accelerate \
        huggingface_hub librosa soundfile soxr pyyaml numpy gradio

    !python scripts/10_gradio_kaggle.py --share

Open the printed https://*.gradio.live URL. Upload audio → Transcribe.
No HF_TOKEN / WANDB / env secrets required (the model repo is public).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from kupe_asr_en.constants import MIMI_FRAME_RATE, MIMI_SAMPLE_RATE
from kupe_asr_en.env import log
from kupe_asr_en.modeling.asr_model import LummaASR

MODEL_ID = "anuj-inavlabs/kupe-asr-en"
MIMI_ID = "kyutai/mimi"

# Public Hub downloads — never require a token.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------- GPU
def _cuda_count() -> int:
    return int(torch.cuda.is_available() and torch.cuda.device_count() or 0)


def _gpu_names() -> list[str]:
    return [torch.cuda.get_device_name(i) for i in range(_cuda_count())]


def _pick_dtype(device: str) -> str:
    """T4 (7.5) and P100 (6.0) have no bf16 — use fp16. Ampere+ can use bf16."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "float32"
    major, _ = torch.cuda.get_device_capability(0)
    return "bfloat16" if major >= 8 else "float16"


def _pick_devices():
    n = _cuda_count()
    if n >= 2:
        return "cuda:0", "cuda:1", "pipeline-2gpu"
    if n == 1:
        return "cuda:0", "cuda:0", "single-gpu"
    return "cpu", "cpu", "cpu"


def _gpu_mem_line() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        parts.append(f"cuda:{i} {torch.cuda.get_device_name(i)}  {alloc:.2f}/{total:.1f} GB")
    return " · ".join(parts)


# --------------------------------------------------------------------------- audio
def load_audio(src, sr: int = MIMI_SAMPLE_RATE) -> np.ndarray:
    """Gradio may hand a filepath, a (sr, ndarray) tuple, or a dict."""
    if src is None:
        raise ValueError("No audio uploaded.")
    if isinstance(src, (tuple, list)) and len(src) == 2 and not isinstance(src[0], str):
        in_sr, arr = src
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.max() > 1.5:                       # int16-style
            arr = arr / 32768.0
        if int(in_sr) != sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=int(in_sr), target_sr=sr)
        return np.ascontiguousarray(arr, dtype=np.float32)
    if isinstance(src, dict):
        src = src.get("path") or src.get("name")
    path = str(src)
    try:
        import soundfile as sf
        arr, in_sr = sf.read(path, dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if int(in_sr) != sr:
            import librosa
            arr = librosa.resample(np.asarray(arr, dtype=np.float32),
                                   orig_sr=int(in_sr), target_sr=sr)
        return np.ascontiguousarray(arr, dtype=np.float32)
    except Exception:
        import librosa
        arr, _ = librosa.load(path, sr=sr, mono=True)
        return np.ascontiguousarray(arr, dtype=np.float32)


def _download_model(model_id: str, model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    log.info("downloading %s (public, no token) …", model_id)
    return snapshot_download(model_id, repo_type="model")


# --------------------------------------------------------------------------- engine
@dataclass
class Metrics:
    mode: str
    audio_s: float
    wall_s: float
    hops: int = 1
    last_hop_s: float = 0.0
    chunk_ms: int = 0
    windows: int = 1
    flags: list[str] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.wall_s / max(self.audio_s, 1e-6)

    @property
    def x_realtime(self) -> float:
        return self.audio_s / max(self.wall_s, 1e-6)

    def markdown(self, engine: "KaggleASR") -> str:
        rtf = self.rtf
        verdict = "faster than realtime" if rtf < 1 else "slower than realtime"
        hop = (f"| Mean hop | {self.wall_s / max(self.hops, 1):.3f} s |\n"
               f"| Last hop | {self.last_hop_s:.3f} s |\n"
               f"| Chunk | {self.chunk_ms} ms × {self.hops} hops |\n") if self.mode == "realtime" else ""
        flags = (" · ".join(self.flags) + "\n\n") if self.flags else ""
        return (
            f"{flags}"
            f"| | |\n|---|---|\n"
            f"| Mode | **{self.mode}** |\n"
            f"| Audio | **{self.audio_s:.2f} s** |\n"
            f"| Wall clock | **{self.wall_s:.2f} s** |\n"
            f"| **RTF** | **{rtf:.3f}** ({verdict}) |\n"
            f"| Throughput | **{self.x_realtime:.2f}× realtime** |\n"
            f"{hop}"
            f"| Windows | {self.windows} |\n"
            f"| GPUs | {engine.n_gpu}× ({', '.join(engine.gpu_names) or 'CPU'}) |\n"
            f"| Placement | Mimi → `{engine.mimi_device}` · Kupe-LM → `{engine.lm_device}` |\n"
            f"| Parallelism | `{engine.parallel}` · dtype `{engine.dtype}` |\n"
            f"| VRAM | {_gpu_mem_line()} |\n"
        )


class KaggleASR:
    """Pipeline-parallel ASR: Mimi on GPU0, Kupe-LM on GPU1 when 2 devices exist."""

    def __init__(self, model_id: str = MODEL_ID, model_dir: str | None = None,
                 codebooks: int = 8, max_new_tokens: int = 256,
                 max_context_s: float = 30.0, repetition_penalty: float = 1.3,
                 no_repeat_ngram_size: int = 3, pre_llm_threshold: float = 0.30,
                 eos_threshold: float = 0.85):
        self.mimi_device, self.lm_device, self.parallel = _pick_devices()
        self.dtype = _pick_dtype(self.lm_device)
        self.n_gpu = _cuda_count()
        self.gpu_names = _gpu_names()
        self.codebooks = int(codebooks)
        self.max_new_tokens = int(max_new_tokens)
        self.max_context_s = float(max_context_s)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.pre_llm_threshold = float(pre_llm_threshold)
        self.eos_threshold = float(eos_threshold)
        self.lock = threading.Lock()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        local = _download_model(model_id, model_dir)
        log.info("loading Kupe-LM on %s (%s) …", self.lm_device, self.dtype)
        self.model, self.tok = LummaASR.load(local, device=self.lm_device, dtype=self.dtype)
        self.model.eval()

        from transformers import MimiModel
        log.info("loading Mimi on %s …", self.mimi_device)
        self.mimi = MimiModel.from_pretrained(MIMI_ID).to(self.mimi_device).eval()
        log.info("ready | %s | %s", self.parallel, _gpu_mem_line())

    def banner(self) -> str:
        gpus = ", ".join(self.gpu_names) or "CPU"
        return (
            f"**Kupe-SLM-EN 600M** · `{MODEL_ID}`  \n"
            f"GPUs: **{self.n_gpu}×** {gpus} · parallelism `{self.parallel}`  \n"
            f"Mimi `{self.mimi_device}` → Kupe-LM `{self.lm_device}` · `{self.dtype}`"
        )

    @torch.inference_mode()
    def encode(self, audio: np.ndarray) -> torch.Tensor:
        """Mimi on mimi_device → codes [K, T] on CPU (tiny)."""
        buf = np.ascontiguousarray(audio, dtype=np.float32)
        if buf.size == 0:
            return torch.zeros(self.codebooks, 0, dtype=torch.long)
        cap = int(self.model.max_audio_frames * MIMI_SAMPLE_RATE / MIMI_FRAME_RATE)
        if buf.size > cap:
            buf = buf[-cap:]
        wav = torch.from_numpy(buf).view(1, 1, -1).to(self.mimi_device)
        codes = self.mimi.encode(wav, num_quantizers=self.codebooks).audio_codes
        return codes[0].to("cpu")                                          # [K, T]

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, with_scores: bool = False):
        """Kupe-LM generate on lm_device. Returns (text, flags)."""
        if codes.numel() == 0 or codes.shape[-1] == 0:
            return "", []
        t = min(int(codes.shape[-1]), int(self.model.max_audio_frames))
        codes = codes[:, :t]
        nf = torch.tensor([t], dtype=torch.long)
        prefix = self.model.build_prefix(codes[None], nf)[0]
        inp = prefix[None].to(self.lm_device)
        attn = torch.ones(1, prefix.shape[0], dtype=torch.long, device=self.lm_device)
        kw = dict(
            inputs_embeds=inp, attention_mask=attn,
            max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
            eos_token_id=self.model.eos_id, pad_token_id=self.tok.pad_token_id,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        flags: list[str] = []
        if with_scores:
            gen = self.model.lumma.generate(
                **kw, output_scores=True, return_dict_in_generate=True)
            ids = gen.sequences[0].tolist()
            for i, logits in enumerate(gen.scores):
                p_eos = torch.softmax(logits[0].float(), dim=-1)[self.model.eos_id].item()
                if p_eos >= self.pre_llm_threshold and "PRE_HIT_LLM" not in flags:
                    flags.append("PRE_HIT_LLM")
                if p_eos >= self.eos_threshold or (i < len(ids) and ids[i] == self.model.eos_id):
                    flags.append("END_OF_SPEECH")
                    break
        else:
            gen = self.model.lumma.generate(**kw)
            ids = gen[0].tolist()
        if self.model.eos_id in ids:
            ids = ids[: ids.index(self.model.eos_id)]
        text = self.tok.decode(ids, skip_special_tokens=True).strip()
        return text, flags

    def transcribe_full(self, audio: np.ndarray) -> tuple[str, Metrics]:
        """Whole-file decode. Long clips are windowed at max_context_s."""
        audio_s = len(audio) / MIMI_SAMPLE_RATE
        win = int(self.max_context_s * MIMI_SAMPLE_RATE)
        hop = win  # non-overlap; each window is a complete utterance-sized chunk
        starts = list(range(0, len(audio), hop)) or [0]
        t0 = time.perf_counter()
        parts: list[str] = []
        # 2-GPU pipeline: encode window i+1 on GPU0 while decoding window i on GPU1
        next_codes = self.encode(audio[starts[0]: starts[0] + win])
        for i, start in enumerate(starts):
            codes = next_codes
            if i + 1 < len(starts):
                nxt = audio[starts[i + 1]: starts[i + 1] + win]
            else:
                nxt = None
            if nxt is not None and self.mimi_device != self.lm_device:
                # overlap: encode next on GPU0, decode current on GPU1
                text, _ = self.decode(codes)
                next_codes = self.encode(nxt)
            else:
                if nxt is not None:
                    next_codes = self.encode(nxt)
                text, _ = self.decode(codes)
            if text:
                parts.append(text)
        wall = time.perf_counter() - t0
        return " ".join(parts).strip(), Metrics(
            mode="full", audio_s=audio_s, wall_s=wall, windows=len(starts))

    def transcribe_realtime(self, audio: np.ndarray, chunk_ms: int):
        """Yield (text, metrics) after every chunk — growing transcript."""
        audio_s = len(audio) / MIMI_SAMPLE_RATE
        hop = max(int(MIMI_SAMPLE_RATE * chunk_ms / 1000.0), 1)
        mx = int(self.max_context_s * MIMI_SAMPLE_RATE)
        buf = np.zeros(0, dtype=np.float32)
        t0 = time.perf_counter()
        hops = 0
        last_hop = 0.0
        text = ""
        flags: list[str] = []
        for start in range(0, len(audio), hop):
            hops += 1
            t_hop = time.perf_counter()
            buf = np.concatenate([buf, audio[start: start + hop]])
            if buf.size > mx:
                buf = buf[-mx:]
            codes = self.encode(buf)
            text, flags = self.decode(codes, with_scores=True)
            last_hop = time.perf_counter() - t_hop
            wall = time.perf_counter() - t0
            heard = min(start + hop, len(audio)) / MIMI_SAMPLE_RATE
            yield text, Metrics(
                mode="realtime", audio_s=heard, wall_s=wall, hops=hops,
                last_hop_s=last_hop, chunk_ms=chunk_ms, windows=1, flags=flags)
        wall = time.perf_counter() - t0
        yield text, Metrics(
            mode="realtime", audio_s=audio_s, wall_s=wall, hops=hops,
            last_hop_s=last_hop, chunk_ms=chunk_ms, windows=1, flags=flags)


ENGINE: KaggleASR | None = None


def get_engine(model_id: str, model_dir: str | None) -> KaggleASR:
    global ENGINE
    if ENGINE is None:
        ENGINE = KaggleASR(model_id=model_id, model_dir=model_dir)
    return ENGINE


# --------------------------------------------------------------------------- Gradio
def _run(audio, mode: str, chunk_ms: int, model_id: str, model_dir: str | None):
    if audio is None:
        yield "Upload an audio file first.", "Waiting for audio."
        return
    eng = get_engine(model_id, model_dir)
    wav = load_audio(audio)
    if wav.size < int(0.1 * MIMI_SAMPLE_RATE):
        yield "", "Audio too short (<0.1 s)."
        return
    with eng.lock:
        if mode == "Realtime chunks":
            for text, met in eng.transcribe_realtime(wav, int(chunk_ms)):
                yield text, met.markdown(eng)
        else:
            text, met = eng.transcribe_full(wav)
            yield text, met.markdown(eng)


def build_app(model_id: str, model_dir: str | None):
    import gradio as gr

    eng = get_engine(model_id, model_dir)

    with gr.Blocks(title="Kupe-SLM-EN ASR", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Kupe-SLM-EN — English ASR")
        gr.Markdown(eng.banner())
        gr.Markdown(
            "Upload a wav/mp3/flac (or record). **Full file** transcribes the clip in "
            "one shot. **Realtime chunks** feeds the same clip hop-by-hop so you see "
            "the transcript grow — the way a voice agent would hear it. "
            "RTF is measured on the live GPU(s)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(
                    label="Audio (upload or record)",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                mode = gr.Radio(
                    ["Full file", "Realtime chunks"],
                    value="Full file",
                    label="Decode mode",
                )
                chunk = gr.Slider(
                    160, 2000, value=480, step=80,
                    label="Realtime chunk size (ms)",
                )
                btn = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=1):
                text = gr.Textbox(label="Transcript", lines=10)
                metrics = gr.Markdown(label="Realtime factor")
        def _click(a, m, c):
            yield from _run(a, m, c, model_id, model_dir)

        btn.click(fn=_click, inputs=[audio, mode, chunk], outputs=[text, metrics])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Kaggle Gradio demo for kupe-asr-en")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--model-dir", default=None, help="local snapshot; else Hub download")
    ap.add_argument("--share", action="store_true", default=True,
                    help="Gradio public URL (needed on Kaggle)")
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    log.info("GPUs detected: %d %s", _cuda_count(), _gpu_names())
    share = False if args.no_share else args.share
    demo = build_app(args.model_id, args.model_dir)
    log.info("launching Gradio share=%s — watch for the *.gradio.live URL", share)
    kw = dict(share=share, server_name=args.host, server_port=args.port,
              show_error=True, inline=False)
    app = demo.queue(max_size=8)
    try:
        app.launch(**kw, ssr_mode=False)
    except TypeError:
        app.launch(**kw)


if __name__ == "__main__":
    main()
