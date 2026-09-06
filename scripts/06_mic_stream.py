#!/usr/bin/env python
"""Live English mic transcription (Mac MPS / CPU / CUDA).

    python scripts/06_mic_stream.py --model-dir artifacts/runs/<run>/model
    python scripts/06_mic_stream.py --model-dir <dir> --device cpu --samplerate 16000

Speak; partials stream live, each turn ends on silence or the model's EOS.
Needs a mic library:  pip install sounddevice
"""
import _bootstrap  # noqa: F401
import argparse
import queue
import sys

from kupe_asr_en.config import load_config
from kupe_asr_en.env import hf_login, log
from kupe_asr_en.stream import END_OF_SPEECH, PARTIAL, PRE_HIT_LLM, StreamingASR


def _resolve(cfg, model_dir):
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    log.info("downloading model from Hub %s …", cfg.repos.model)
    return snapshot_download(cfg.repos.model, repo_type="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--chunk-ms", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login()
    try:
        import sounddevice as sd
    except Exception:
        sys.exit("mic library missing — run: pip install sounddevice")

    s = cfg.stream
    asr = StreamingASR(
        _resolve(cfg, args.model_dir), device=args.device, codebooks=int(cfg.audio.codebooks),
        pre_llm_threshold=s.pre_llm_threshold, eos_threshold=s.eos_threshold,
        max_context_s=s.max_context_s, silence_ms=s.silence_ms, silence_rms=s.silence_rms,
        max_new_tokens=cfg.eval.max_new_tokens, repetition_penalty=s.repetition_penalty,
        no_repeat_ngram_size=s.no_repeat_ngram_size, denoise=s.denoise)
    log.info("model on %s | mic %d Hz | speak now (Ctrl+C to stop)", asr.device, args.samplerate)

    q: queue.Queue = queue.Queue()
    sd_cb = lambda indata, frames, ti, st: q.put(indata[:, 0].copy())  # noqa: E731
    chunk_ms = args.chunk_ms or s.chunk_ms
    blocksize = int(args.samplerate * chunk_ms / 1000)
    last = ""
    with sd.InputStream(samplerate=args.samplerate, channels=1, dtype="float32",
                        blocksize=blocksize, callback=sd_cb):
        try:
            while True:
                block = q.get()
                asr.add_audio(block, args.samplerate)
                for ev in asr.step():
                    if ev.type == PARTIAL and ev.text and ev.text != last:
                        last = ev.text
                        print(f"\r{ev.text}", end="", flush=True)
                    elif ev.type == PRE_HIT_LLM:
                        print(f"\r⚡ prefetch-LLM  {ev.text}", flush=True)
                    elif ev.type == END_OF_SPEECH:
                        print(f"\r■ {ev.text}", flush=True)
                        last = ""
                        asr.reset()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
