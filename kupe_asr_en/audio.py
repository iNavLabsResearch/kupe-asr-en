"""Audio helpers: decode any HF audio cell, resample to 24 kHz mono, and
encode/decode the compact storage codec (FLAC lossless, or Opus for 10x smaller).

FLAC is the default: lossless, so the `raw` config is a faithful copy that the
Mimi encoder sees exactly as the source did. Opus is offered for disk-tight boxes
(3000 h of FLAC is ~380 GB; the same as Opus@24k is ~35 GB) at a small quality
cost that is acceptable for ASR training data.
"""
from __future__ import annotations

import io

import numpy as np

from .constants import MIMI_SAMPLE_RATE


def decode_audio(cell) -> tuple[np.ndarray | None, int | None]:
    """(float32 mono-or-multi array, sr) from any HF audio cell shape."""
    if cell is None:
        return None, None
    if isinstance(cell, dict):
        if cell.get("array") is not None and cell.get("sampling_rate"):
            return np.asarray(cell["array"], dtype=np.float32), int(cell["sampling_rate"])
        import soundfile as sf
        if cell.get("bytes"):
            arr, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=False)
            return arr, sr
        if cell.get("path"):
            arr, sr = sf.read(cell["path"], dtype="float32", always_2d=False)
            return arr, sr
    return None, None


def to_mono_24k(array, sr: int, target_sr: int = MIMI_SAMPLE_RATE) -> np.ndarray:
    import librosa
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:                                  # down-mix to mono
        ch_axis = int(np.argmin(array.shape))
        array = np.asarray(array.mean(axis=ch_axis), dtype=np.float32).reshape(-1)
    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(array, dtype=np.float32)


def encode_bytes(array: np.ndarray, sr: int, fmt: str, opus_bitrate: int = 24000) -> bytes:
    """Float mono [-1,1] -> compressed bytes in `fmt` ('flac' or 'opus')."""
    import soundfile as sf
    buf = io.BytesIO()
    array = np.asarray(array, dtype=np.float32)
    if fmt == "flac":
        sf.write(buf, array, sr, format="FLAC")
    elif fmt == "opus":
        # libsndfile Opus is fixed at 48 kHz; resample up, tag the true sr in the row.
        import librosa
        a48 = librosa.resample(array, orig_sr=sr, target_sr=48000)
        sf.write(buf, a48, 48000, format="OGG", subtype="OPUS")
    else:
        raise ValueError(f"unknown audio_format: {fmt}")
    return buf.getvalue()


def make_silence(dur_s: float, sr: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a near-silent clip labeled "" for anti-hallucination training.

    'pure'    : digital silence + tiny dither (avoids an all-zero waveform that a
                codec might treat degenerately).
    'ambient' : low-level white noise (fan/AC/room tone) at ~-40 dBFS.
    """
    n = max(1, int(dur_s * sr))
    if kind == "ambient":
        a = rng.normal(0.0, 0.005, n).astype(np.float32)     # ~ -40 dBFS
    else:                                                     # pure silence + dither
        a = rng.normal(0.0, 1e-4, n).astype(np.float32)
    return np.ascontiguousarray(np.clip(a, -1.0, 1.0), dtype=np.float32)


def decode_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """Compressed bytes (flac/opus) -> (float32 mono @ its own sr, sr)."""
    import soundfile as sf
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), int(sr)
