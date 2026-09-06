"""Load the `mimi` config for training — always from the Hub (rule: never train on
local-only data), unless a local mirror already exists from a just-finished encode.

Returns train / val / test HF Datasets split on the fixed `split` column.
"""
from __future__ import annotations

import glob
import os

from .constants import CONFIG_MIMI, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from .env import log, require_token
from .hub import download_bunches
from .data.bunch import dataset_from_parquets


def load_mimi(cfg, *, from_hub: bool = True):
    """Load the full mimi dataset (all splits) as one HF Dataset."""
    local_bunches = sorted(glob.glob(os.path.join(cfg.paths.mimi_dir, "bunches", "bunch_*.parquet")))
    if not from_hub and local_bunches:
        log.info("loading mimi from local bunches (%d files)", len(local_bunches))
        return dataset_from_parquets(local_bunches)
    paths = download_bunches(cfg.repos.data, CONFIG_MIMI, require_token())
    return dataset_from_parquets(paths)


def splits(ds):
    tr = ds.filter(lambda s: s == SPLIT_TRAIN, input_columns="split", num_proc=4)
    va = ds.filter(lambda s: s == SPLIT_VAL, input_columns="split", num_proc=4)
    te = ds.filter(lambda s: s == SPLIT_TEST, input_columns="split", num_proc=4)
    if va.num_rows == 0:                       # safety: carve a holdout if none tagged
        sp = tr.train_test_split(test_size=0.02, seed=1337)
        tr, va = sp["train"], sp["test"]
    return tr, va, te
