#!/usr/bin/env python
"""Stage 0: create the three Hub repos and seed the dataset card.

    python scripts/00_create_repos.py            # public
    python scripts/00_create_repos.py --private
"""
import _bootstrap  # noqa: F401
import argparse
import os
import tempfile

from kupe_asr_en.config import load_config
from kupe_asr_en.env import ensure_repo, hf_login, log, upload_file


CARD = """---
license: other
task_categories: [automatic-speech-recognition]
language: [en]
tags: [asr, mimi, lumma, kupe]
configs:
  - config_name: raw
    data_files: raw/data/bunch_*.parquet
  - config_name: mimi
    data_files: mimi/data/bunch_*.parquet
---

# {project} — data

Two loadable configs, packed into ~20-25 `bunch_*.parquet` files each (Hub-quota friendly):

- **raw**  — 24 kHz mono English audio ({fmt} bytes) + text. The encode stage reads this.
- **mimi** — Mimi c0..c7 codes (12.5 Hz) + text. **Training reads this.**

Ledgers under `ledger/` (`data.json`, `mimi.json`) track collected/encoded hours and resume state.

```python
from datasets import load_dataset
ds = load_dataset("{repo}", "mimi", split="train")
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-data-card", action="store_true",
                    help="reuse an existing data repo: create runs/model only, don't reseed its card")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login()
    ensure_repo(cfg.repos.data, "dataset", private=args.private)
    ensure_repo(cfg.repos.runs, "model", private=args.private)
    ensure_repo(cfg.repos.model, "model", private=args.private)

    if not args.skip_data_card:
        card = CARD.format(project=cfg.project, repo=cfg.repos.data, fmt=cfg.data.audio_format)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "README.md")
            open(p, "w", encoding="utf-8").write(card)
            upload_file(p, cfg.repos.data, "dataset", "README.md", "seed data card")
    log.info("repos ready: %s | %s | %s", cfg.repos.data, cfg.repos.runs, cfg.repos.model)


if __name__ == "__main__":
    main()
