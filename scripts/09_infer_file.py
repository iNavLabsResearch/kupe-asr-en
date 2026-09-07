#!/usr/bin/env python
"""Transcribe an audio FILE two ways: (1) full one-shot, (2) simulated realtime chunks.

    python scripts/09_infer_file.py --audio test.mp3
    python scripts/09_infer_file.py --audio test.mp3 --model-dir <local dir> --chunk-ms 480

Full mode = feed the whole clip and decode once.
Chunk mode = feed the audio in `--chunk-ms` pieces as if the user were speaking live;
after each chunk it prints the CURRENT accumulated transcript on a new line, so you see
it grow chunk by chunk.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import torch

from kupe_asr_en.config import load_config
from kupe_asr_en.constants import MIMI_SAMPLE_RATE
from kupe_asr_en.env import hf_login, log
from kupe_asr_en.stream import StreamingASR, PARTIAL, PRE_HIT_LLM, END_OF_SPEECH


def _resolve(cfg, model_dir):
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    log.info("downloading model from Hub %s …", cfg.repos.model)
    return snapshot_download(cfg.repos.model, repo_type="model")


def _load_audio(path, sr=MIMI_SAMPLE_RATE):
    import librosa
    a, _ = librosa.load(path, sr=sr, mono=True)          # decodes mp3/wav/flac -> 24k mono
    return np.ascontiguousarray(a, dtype=np.float32)


@torch.inference_mode()
def full_transcribe(asr: StreamingASR, audio, max_new_tokens=200):
    asr.reset()
    asr.add_audio(audio, MIMI_SAMPLE_RATE)
    if asr.buffer.size == 0:
        return ""
    codes = asr._encode()                                 # [K, T] for the whole clip
    nf = torch.tensor([codes.shape[1]], dtype=torch.long)
    prefix = asr.model.build_prefix(codes[None], nf)[0]
    inp = prefix[None].to(asr.device)
    attn = torch.ones(1, prefix.shape[0], dtype=torch.long, device=asr.device)
    gen = asr.model.lumma.generate(
        inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new_tokens,
        do_sample=False, num_beams=1, eos_token_id=asr.model.eos_id,
        pad_token_id=asr.tok.pad_token_id, repetition_penalty=asr.repetition_penalty,
        no_repeat_ngram_size=asr.no_repeat_ngram_size)
    ids = gen[0].tolist()
    if asr.model.eos_id in ids:
        ids = ids[: ids.index(asr.model.eos_id)]
    return asr.tok.decode(ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--chunk-ms", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        hf_login()                          # optional: public model downloads work tokenless
    except Exception:
        log.info("no HF token — using anonymous access (fine for a public model)")
    s = cfg.stream
    asr = StreamingASR(_resolve(cfg, args.model_dir), device=args.device,
                       codebooks=int(cfg.audio.codebooks),
                       pre_llm_threshold=s.pre_llm_threshold, eos_threshold=s.eos_threshold,
                       max_context_s=s.max_context_s, silence_ms=s.silence_ms,
                       silence_rms=s.silence_rms, max_new_tokens=cfg.eval.max_new_tokens,
                       repetition_penalty=s.repetition_penalty,
                       no_repeat_ngram_size=s.no_repeat_ngram_size, denoise=False)

    audio = _load_audio(args.audio)
    dur = len(audio) / MIMI_SAMPLE_RATE
    frames = int(dur * 12.5)
    log.info("audio: %.1fs  (~%d Mimi frames; model max = %d frames) on %s",
             dur, frames, asr.model.max_audio_frames, asr.device)

    # ---------- 1) FULL ONE-SHOT ----------
    print("\n" + "=" * 70 + "\n[1] FULL TRANSCRIPTION (whole file at once)\n" + "=" * 70)
    print(full_transcribe(asr, audio))

    # ---------- 2) REALTIME CHUNKS (growing/accumulated transcript) ----------
    chunk_ms = args.chunk_ms or s.chunk_ms
    hop = int(MIMI_SAMPLE_RATE * chunk_ms / 1000.0)
    print("\n" + "=" * 70 +
          f"\n[2] REALTIME SIMULATION ({chunk_ms} ms chunks) — each line adds one chunk;\n"
          "    the transcript grows as more audio arrives\n" + "=" * 70)
    n = 0
    for end in range(hop, len(audio) + hop, hop):
        n += 1
        so_far = audio[:min(end, len(audio))]            # everything heard up to now
        text = full_transcribe(asr, so_far, max_new_tokens=cfg.eval.max_new_tokens)
        t = min(end, len(audio)) / MIMI_SAMPLE_RATE
        print(f"chunk {n:2d} [{t:5.1f}s]: {text}")
    print("=" * 70)


if __name__ == "__main__":
    main()
