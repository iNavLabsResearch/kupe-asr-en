"""Static contract shared by every stage. No config, no env, no side effects.

Phase 1 is ENGLISH ONLY. There are deliberately no language tokens and no
auto-language-detection machinery here — that is a later phase. The model
transcribes English directly.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Mimi codec geometry (kyutai/mimi) — verified from the model config 2026-09.
# --------------------------------------------------------------------------
MIMI_SAMPLE_RATE = 24_000
MIMI_FRAME_RATE = 12.5
MIMI_SAMPLES_PER_FRAME = int(MIMI_SAMPLE_RATE / MIMI_FRAME_RATE)  # 1920
MIMI_CODEBOOK_SIZE = 2048        # codes per codebook
MIMI_MAX_CODEBOOKS = 8           # c0 semantic + c1..c7 acoustic (what we encode)

# --------------------------------------------------------------------------
# Base model — Lumma requires a specific transformers version. We assert it at
# import time in modeling code so a wrong environment fails loudly, not weirdly.
# --------------------------------------------------------------------------
LUMMA_REQUIRED_TRANSFORMERS = "5.4.0"

# Dataset config (== HF `name`) for the two loadable views of the data repo.
CONFIG_RAW = "raw"      # resampled 24 kHz audio (flac/opus bytes) — encode reads this
CONFIG_MIMI = "mimi"    # Mimi c0..c7 codes + text — train reads this

# Splits carried in every row so the split is fixed once, at fetch time.
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
