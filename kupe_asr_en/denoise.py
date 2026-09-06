"""Optional RNNoise front-end for streaming (kept dependency-light).

If `pyrnnoise` is installed we denoise 48 kHz mono frames before ASR; otherwise
this is a transparent passthrough so the mic demo still runs everywhere.
"""
from __future__ import annotations

import numpy as np

from .env import log


class StreamingRNNoise:
    def __init__(self):
        self._ok = False
        try:
            from pyrnnoise import RNNoise
            self._rn = RNNoise(48000)
            self._ok = True
        except Exception as e:
            log.warning("pyrnnoise unavailable (%s) -> denoise passthrough", e)
        self.reset()

    def reset(self):
        self._tail = np.zeros(0, dtype=np.float32)

    def process(self, chunk: np.ndarray, sr: int):
        if not self._ok:
            return chunk, sr
        import librosa
        x = librosa.resample(np.asarray(chunk, np.float32), orig_sr=sr, target_sr=48000) \
            if sr != 48000 else np.asarray(chunk, np.float32)
        try:
            out = np.concatenate([o for o in self._rn.process_chunk(x)]) if hasattr(self._rn, "process_chunk") \
                else self._rn.filter(x)
        except Exception:
            out = x
        return np.asarray(out, np.float32), 48000
