#!/usr/bin/env python
"""Stage 4: (re)evaluate a trained model on val + test and write a report.

    python scripts/04_evaluate.py --model-dir artifacts/runs/<run>/model
    python scripts/04_evaluate.py --model-dir <dir> --split test --push
"""
import _bootstrap  # noqa: F401
import argparse

from kupe_asr_en.config import load_config
from kupe_asr_en.dataset import load_mimi, splits
from kupe_asr_en.env import ensure_repo, hf_login, log, upload_folder
from kupe_asr_en.evaluate import run_eval, save_report
from kupe_asr_en.modeling.asr_model import LummaASR


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hf_login()
    dev = _device()
    dtype = "float32" if dev in ("cpu", "mps") else "bfloat16"
    model, tok = LummaASR.load(args.model_dir, device=dev, dtype=dtype)

    _, val_ds, test_ds = splits(load_mimi(cfg, from_hub=True))
    targets = {"val": val_ds, "test": test_ds}
    if args.split != "both":
        targets = {args.split: targets[args.split]}

    reports = {}
    for name, ds in targets.items():
        if ds.num_rows == 0:
            continue
        reports[name] = run_eval(model, tok, ds, dev,
                                 num_codebooks=int(cfg.audio.codebooks),
                                 max_audio_frames=int(cfg.model.max_audio_frames),
                                 max_samples=int(cfg.eval.max_samples),
                                 max_new_tokens=int(cfg.eval.max_new_tokens),
                                 batch_size=int(cfg.eval.batch_size), split_name=name)
        log.info("%s WER=%.4f CER=%.4f (n=%d)", name, reports[name]["wer"],
                 reports[name]["cer"], reports[name]["n"])
    save_report(reports, args.model_dir, title="standalone-eval")
    if args.push:
        ensure_repo(cfg.repos.runs, "model")
        upload_folder(args.model_dir, cfg.repos.runs, "model",
                      path_in_repo="standalone_evals", commit_message="standalone eval",
                      ignore_patterns=["lumma/*.safetensors", "frontend.pt"])


if __name__ == "__main__":
    main()
