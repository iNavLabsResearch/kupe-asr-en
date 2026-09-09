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

# Dataset config (== HF `name`) for the loadable views of the data repo.
CONFIG_RAW = "raw"      # resampled 24 kHz audio (flac/opus bytes) — encode reads this
CONFIG_MIMI = "mimi"    # Mimi c0..c7 codes + text — Lumma/Nandi-on-codes train reads this
CONFIG_FC = "fc"        # FastConformer encoder features + text — fc_train reads this

# --------------------------------------------------------------------------
# FastConformer encoder geometry (nvidia/stt_en_fastconformer_hybrid_large_pc).
# 16 kHz in; 8x depthwise subsampling of 10 ms mel frames -> 80 ms/frame = 12.5 fps
# (same frame rate as Mimi, by coincidence). d_model = 512 (discovered at runtime,
# never hardcoded into the projection — see fc_frontend).
# --------------------------------------------------------------------------
FC_SAMPLE_RATE = 16_000
FC_FRAME_RATE = 12.5
FC_DEFAULT_ID = "nvidia/stt_en_fastconformer_hybrid_large_pc"
NANDI_DECODER_ID = "FrontiersMind/Nandi-Mini-150M"

# Splits carried in every row so the split is fixed once, at fetch time.
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
