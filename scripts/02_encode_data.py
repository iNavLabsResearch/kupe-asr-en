#!/usr/bin/env python
"""Stage 2: Mimi-encode the `raw` config into the `mimi` config (c0..c7).

    python scripts/02_encode_data.py            # download raw bunches, encode, push mimi
    python scripts/02_encode_data.py --status   # print encode progress and exit

GPU strongly recommended. Resumable per raw bunch via the mimi ledger.
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr_en.config import load_config
from kupe_asr_en.data.encode import encode, status
from kupe_asr_en.env import hf_login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_push:
        cfg.mimi.push = False
    hf_login()
    if args.status:
        status(cfg)
        return
    encode(cfg)


if __name__ == "__main__":
    main()
