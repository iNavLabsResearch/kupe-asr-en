#!/usr/bin/env python
"""Stage 1: collect English audio into the `raw` config on the Hub.

    python scripts/01_fetch_data.py            # fetch + upload bunches (resumable)
    python scripts/01_fetch_data.py --status   # print the data ledger and exit
    python scripts/01_fetch_data.py --no-push  # local shards only
    python scripts/01_fetch_data.py --reset    # zero ledger hour counts (keeps Hub data)

Runs for hours — use tmux (see README). Safe to Ctrl-C and re-run; it resumes.
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr_en.config import load_config
from kupe_asr_en.data.fetch import fetch, status
from kupe_asr_en.env import hf_login


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.no_push:
        cfg.data.push = False
    hf_login()
    if args.status:
        status(cfg)
        return
    fetch(cfg, reset=args.reset)


if __name__ == "__main__":
    main()
