#!/usr/bin/env python
"""Stage 1 (FC track): FastConformer-encode the `raw` config into the `fc` config.

    python scripts/12_fc_encode.py             # download raw, encode features, push `fc`
    python scripts/12_fc_encode.py --status    # print encode progress and exit

Multi-GPU: uses ALL visible GPUs (round-robin, OOM auto-split). Resumable per raw
bunch via the fc ledger. Run inside tmux. This is a DISTINCT step — do it once,
then train (13_fc_train.py) as many times as you like off the cached features.
"""
import os
import _bootstrap  # noqa: F401
import argparse

from kupe_asr_en.config import load_config
from kupe_asr_en.data.fc_encode import encode, status
from kupe_asr_en.env import hf_login

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "fastconformer_nandi.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_DEFAULT)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_push:
        cfg.fc.push = False
    hf_login()
    if args.status:
        status(cfg)
        return
    encode(cfg)


if __name__ == "__main__":
    main()
