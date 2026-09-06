"""Collator: rows of Mimi codes + English text -> padded tensors for LummaASR.

Emitted batch keys match `LummaASR.forward`:
  codes         Long [B, K, T_max]   (0-padded past each clip's real frames)
  num_frames    Long [B]             (real frames, capped at max_audio_frames)
  text_ids      Long [B, M_max]      (text tokens + eos, pad-padded)
  text_lengths  Long [B]
  labels        Long [B, M_max]      (== text_ids; the model slices [:m], so pad is unused)
"""
from __future__ import annotations

import numpy as np
import torch


class AsrCollator:
    def __init__(self, tokenizer, num_codebooks: int, max_audio_frames: int,
                 max_text_tokens: int, bos_id: int, eos_id: int, pad_id: int):
        self.tok = tokenizer
        self.k = int(num_codebooks)
        self.max_audio_frames = int(max_audio_frames)
        self.max_text_tokens = int(max_text_tokens)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)

    def _text_ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False).input_ids
        if ids and ids[0] == self.bos_id:      # Lumma tokenizer prepends bos even here
            ids = ids[1:]
        ids = ids[: self.max_text_tokens]
        return ids + [self.eos_id]

    def __call__(self, rows: list[dict]) -> dict:
        codes_list, nframes, text_lists = [], [], []
        for r in rows:
            cb = np.asarray(r["codes"], dtype=np.int64)          # [K, T]
            if cb.ndim == 1:                                     # single-codebook stored flat
                cb = cb[None, :]
            cb = cb[: self.k, : self.max_audio_frames]
            codes_list.append(cb)
            nframes.append(cb.shape[1])
            text_lists.append(self._text_ids(r["text"]))

        K = max(c.shape[0] for c in codes_list)
        T = max(c.shape[1] for c in codes_list)
        B = len(rows)
        codes = np.zeros((B, K, T), dtype=np.int64)
        for i, c in enumerate(codes_list):
            codes[i, : c.shape[0], : c.shape[1]] = c

        M = max(len(t) for t in text_lists)
        text_ids = np.full((B, M), self.pad_id, dtype=np.int64)
        text_lengths = np.zeros(B, dtype=np.int64)
        for i, t in enumerate(text_lists):
            text_ids[i, : len(t)] = t
            text_lengths[i] = len(t)

        text_ids_t = torch.from_numpy(text_ids)
        return {
            "codes": torch.from_numpy(codes),
            "num_frames": torch.tensor(nframes, dtype=torch.long),
            "text_ids": text_ids_t,
            "text_lengths": torch.from_numpy(text_lengths),
            "labels": text_ids_t.clone(),
        }
