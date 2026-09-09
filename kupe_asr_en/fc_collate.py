"""Collators for the FastConformer + Nandi track.

FcCollator (default) — rows of stored FastConformer features + text -> padded
tensors for FastConformerASR.forward. Emitted keys:
  feats         Float [B, T_max, D]   (0-padded past each clip's real frames)
  num_frames    Long  [B]             (real frames, capped at max_audio_frames)
  text_ids      Long  [B, M_max]      (text tokens + eos, pad-padded)
  text_lengths  Long  [B]
  labels        Long  [B, M_max]

FcWaveCollator — for LIGHT encoder fine-tuning: decodes `raw` audio bytes to 16 kHz
waveforms so gradients flow through the live encoder. Emits `wave`/`wave_len`
instead of `feats`/`num_frames`.
"""
from __future__ import annotations

import numpy as np
import torch

from .audio import decode_bytes
from .constants import FC_SAMPLE_RATE


class _TextMixin:
    def _text_ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False).input_ids
        if ids and ids[0] == self.bos_id:
            ids = ids[1:]
        ids = ids[: self.max_text_tokens]
        return ids + [self.eos_id]

    def _pack_text(self, text_lists):
        M = max(len(t) for t in text_lists)
        B = len(text_lists)
        text_ids = np.full((B, M), self.pad_id, dtype=np.int64)
        text_lengths = np.zeros(B, dtype=np.int64)
        for i, t in enumerate(text_lists):
            text_ids[i, : len(t)] = t
            text_lengths[i] = len(t)
        t = torch.from_numpy(text_ids)
        return {"text_ids": t, "text_lengths": torch.from_numpy(text_lengths),
                "labels": t.clone()}


class FcCollator(_TextMixin):
    def __init__(self, tokenizer, max_audio_frames: int, max_text_tokens: int,
                 bos_id: int, eos_id: int, pad_id: int):
        self.tok = tokenizer
        self.max_audio_frames = int(max_audio_frames)
        self.max_text_tokens = int(max_text_tokens)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)

    def __call__(self, rows: list[dict]) -> dict:
        feats_list, nframes, text_lists = [], [], []
        for r in rows:
            d = int(r["feat_dim"])
            f = np.frombuffer(r["feats"], dtype="<f2").reshape(-1, d)
            f = np.asarray(f[: self.max_audio_frames], dtype=np.float32)
            feats_list.append(f)
            nframes.append(f.shape[0])
            text_lists.append(self._text_ids(r["text"]))

        D = feats_list[0].shape[1]
        T = max(f.shape[0] for f in feats_list)
        B = len(rows)
        feats = np.zeros((B, T, D), dtype=np.float32)
        for i, f in enumerate(feats_list):
            feats[i, : f.shape[0]] = f

        out = {"feats": torch.from_numpy(feats),
               "num_frames": torch.tensor(nframes, dtype=torch.long)}
        out.update(self._pack_text(text_lists))
        return out


class FcWaveCollator(_TextMixin):
    """Decode `raw` audio bytes -> 16 kHz waveforms (encoder fine-tuning path)."""

    def __init__(self, tokenizer, max_audio_s: float, max_text_tokens: int,
                 bos_id: int, eos_id: int, pad_id: int):
        self.tok = tokenizer
        self.max_samples = int(max_audio_s * FC_SAMPLE_RATE)
        self.max_text_tokens = int(max_text_tokens)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)

    def __call__(self, rows: list[dict]) -> dict:
        import librosa
        waves, wlens, text_lists = [], [], []
        for r in rows:
            arr, sr = decode_bytes(r["audio_bytes"])
            if sr != FC_SAMPLE_RATE:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=FC_SAMPLE_RATE)
            arr = np.asarray(arr[: self.max_samples], dtype=np.float32)
            waves.append(arr)
            wlens.append(arr.shape[0])
            text_lists.append(self._text_ids(r["text"]))

        S = max(w.shape[0] for w in waves)
        B = len(rows)
        wave = np.zeros((B, S), dtype=np.float32)
        for i, w in enumerate(waves):
            wave[i, : w.shape[0]] = w
        out = {"wave": torch.from_numpy(wave),
               "wave_len": torch.tensor(wlens, dtype=torch.long)}
        out.update(self._pack_text(text_lists))
        return out
