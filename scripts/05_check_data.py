#!/usr/bin/env python
"""Stage check: print both ledgers and assert the Phase-1 hour targets.

    python scripts/05_check_data.py           # print status, exit 0 if >= min_hours
    python scripts/05_check_data.py --strict  # exit 1 unless >= target_hours

Use before encoding/training to fail fast on a dataset that missed target.
"""
import _bootstrap  # noqa: F401
import argparse
import sys

from kupe_asr_en.config import load_config
from kupe_asr_en.data.encode import status as mimi_status
from kupe_asr_en.data.fetch import status as raw_status
from kupe_asr_en.env import hf_login, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--strict", action="store_true", help="require >= target_hours, not just min")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login()
    log.info("==================== RAW ====================")
    d = raw_status(cfg)
    log.info("==================== MIMI ===================")
    mimi_status(cfg)

    kept = d["totals"]["kept_h"]
    need = d["target_hours"] if args.strict else d["min_hours"]
    if kept < need:
        log.error("GATE FAILED: %.1f h < required %.0f h. Do NOT proceed to training.",
                  kept, need)
        sys.exit(1)
    log.info("GATE OK: %.1f h >= %.0f h ✓", kept, need)


if __name__ == "__main__":
    main()
