#!/usr/bin/env python
"""Stage 2 (FC track): train the projector + Nandi-Mini-150M decoder on `fc` features.

    python scripts/13_fc_train.py                        # pulls `fc` features from Hub
    python scripts/13_fc_train.py --epochs 3 --bs 32 --lr 3e-4
    python scripts/13_fc_train.py --finetune-encoder --lr 1e-5   # + light encoder FT (reads `raw`)
    python scripts/13_fc_train.py --resume kupe-asr-en-fc-nandi-fc-20260909-...
    accelerate launch scripts/13_fc_train.py             # multi-GPU DDP

Refuses to start below data.min_hours. Requires transformers==5.4.0 (Nandi) and,
for --finetune-encoder, nemo_toolkit[asr].
"""
import os
import _bootstrap  # noqa: F401
import argparse

from kupe_asr_en.config import load_config
from kupe_asr_en.fc_train import train

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "fastconformer_nandi.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_DEFAULT)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--projector", choices=["linear", "mlp"], default=None)
    ap.add_argument("--finetune-encoder", action="store_true",
                    help="light-fine-tune the FastConformer encoder (re-encodes `raw` on the fly)")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--local", action="store_true", help="use local bunches if present")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
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
    if args.projector:
        cfg.audio.projector = args.projector
    if args.finetune_encoder:
        cfg.encoder.finetune = True
    if args.no_push:
        cfg.train.push_to_hub = False
        cfg.train.push_checkpoints = False
    train(cfg, from_hub=not args.local, resume=args.resume)


if __name__ == "__main__":
    main()
