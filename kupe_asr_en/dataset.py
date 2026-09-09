"""Load the `mimi` config for training — always from the Hub (rule: never train on
local-only data), unless a local mirror already exists from a just-finished encode.

Returns train / val / test HF Datasets split on the fixed `split` column.
"""
from __future__ import annotations

import glob
import os

from .constants import (CONFIG_FC, CONFIG_MIMI, CONFIG_RAW, SPLIT_TEST,
                        SPLIT_TRAIN, SPLIT_VAL)
from .env import log, require_token
from .hub import download_bunches
from .data.bunch import dataset_from_parquets


def _load_config(cfg, config_name: str, local_dir: str, *, from_hub: bool):
    local_bunches = sorted(glob.glob(os.path.join(local_dir, "bunches", "bunch_*.parquet")))
    if not from_hub and local_bunches:
        log.info("loading %s from local bunches (%d files)", config_name, len(local_bunches))
        return dataset_from_parquets(local_bunches)
    paths = download_bunches(cfg.repos.data, config_name, require_token())
    return dataset_from_parquets(paths)


def load_mimi(cfg, *, from_hub: bool = True):
    """Load the full mimi dataset (all splits) as one HF Dataset."""
    return _load_config(cfg, CONFIG_MIMI, cfg.paths.mimi_dir, from_hub=from_hub)


def load_fc(cfg, *, from_hub: bool = True):
    """Load the full FastConformer-features dataset (all splits) as one HF Dataset."""
    return _load_config(cfg, CONFIG_FC, cfg.paths.fc_dir, from_hub=from_hub)


def load_raw(cfg, *, from_hub: bool = True):
    """Load the `raw` (waveform) dataset — used only for encoder fine-tuning."""
    return _load_config(cfg, CONFIG_RAW, cfg.paths.raw_dir, from_hub=from_hub)


def splits(ds):
    tr = ds.filter(lambda s: s == SPLIT_TRAIN, input_columns="split", num_proc=4)
    va = ds.filter(lambda s: s == SPLIT_VAL, input_columns="split", num_proc=4)
    te = ds.filter(lambda s: s == SPLIT_TEST, input_columns="split", num_proc=4)
    if va.num_rows == 0:                       # safety: carve a holdout if none tagged
        sp = tr.train_test_split(test_size=0.02, seed=1337)
        tr, va = sp["train"], sp["test"]
    return tr, va, te
