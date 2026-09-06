#!/usr/bin/env python
"""Stage 3: train LummaASR on the `mimi` config.

    python scripts/03_train.py                      # pulls `mimi` from Hub
    python scripts/03_train.py --frontend flatten --codebooks 4   # the A/B variant
    python scripts/03_train.py --epochs 2 --bs 16 --lr 2e-4
    python scripts/03_train.py --resume kupe-asr-en-per_frame_sum-20260906-...
    accelerate launch scripts/03_train.py           # multi-GPU DDP

Refuses to start below data.min_hours. Requires transformers==5.4.0 (Lumma).
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr_en.config import load_config
from kupe_asr_en.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--frontend", choices=["per_frame_sum", "flatten"], default=None)
    ap.add_argument("--codebooks", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--local", action="store_true", help="use local mimi bunches if present")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.frontend:
        cfg.audio.frontend = args.frontend
    if args.codebooks:
        cfg.audio.codebooks = args.codebooks
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.bs:
        cfg.train.per_device_batch_size = args.bs
    if args.grad_accum:
        cfg.train.grad_accum = args.grad_accum
    if args.lr:
        cfg.train.lr = args.lr
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.no_push:
        cfg.train.push_to_hub = False
        cfg.train.push_checkpoints = False
    train(cfg, from_hub=not args.local, resume=args.resume)


if __name__ == "__main__":
    main()
