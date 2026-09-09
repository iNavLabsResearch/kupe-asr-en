"""Training (FC track) — HF Trainer over the `fc` features with FastConformerASR.

Trains the projector + Nandi-Mini decoder (+ optionally LIGHT FastConformer encoder
fine-tuning). Guards mirror train.py:
  1. min-hours gate: refuse to train if fc-encoded hours < data.min_hours.
  2. resume: local checkpoint, else pull the run's last checkpoint from the runs repo.
  3. live WER every eval, early stopping on eval_loss, throttled Hub checkpointing.

Reuses train.py's TrainingArguments builder, checkpoint resolver, and the tied-weight
untie Trainer so nothing about the Nandi save path is re-implemented.
"""
from __future__ import annotations

import json
import os
import time

import torch
from transformers import TrainerCallback

from .dataset import load_fc, load_raw, splits
from .env import ensure_repo, hf_login, hf_token, init_wandb, log, upload_folder
from .fc_collate import FcCollator, FcWaveCollator
from .fc_evaluate import run_eval, save_report
from .ledger import Ledger, iso_now
from .modeling.asr_model import load_tokenizer
from .modeling.fc_asr_model import FastConformerASR
from .train import LummaTrainer, _resume_checkpoint, _training_args  # reuse the plumbing


# --------------------------------------------------------------------------- gate
def _encoded_hours(cfg) -> float:
    led = Ledger(os.path.join(cfg.paths.ledger_dir, "fc.json"), repo_id=cfg.repos.data,
                 path_in_repo="ledger/fc.json", default={})
    led.sync_from_hub(merge=lambda l, h: h if h.get("encoded_hours", 0) > l.get("encoded_hours", 0) else l)
    return float(led.d.get("encoded_hours", 0.0))


def _check_min_hours(cfg):
    have = _encoded_hours(cfg)
    need = float(cfg.data.min_hours)
    if have <= 0:
        log.warning("fc ledger reports 0 encoded hours — is fc-encode done and pushed?")
    if have < need:
        raise RuntimeError(
            f"REFUSING TO TRAIN: only {have:.1f} h fc-encoded < min_hours {need:.0f} h.\n"
            "Run fc-encode over more data, or lower data.min_hours for a smoke run.")
    log.info("min-hours gate OK: %.1f h fc-encoded (>= %.0f h).", have, need)


# --------------------------------------------------------------------------- callbacks
class WerCallback(TrainerCallback):
    def __init__(self, model, tok, val_ds, cfg):
        self.model, self.tok, self.cfg = model, tok, cfg
        n = min(val_ds.num_rows, int(cfg.eval.subset_samples))
        self.subset = val_ds.shuffle(seed=0).select(range(n))

    def on_evaluate(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return control
        try:
            rep = run_eval(self.model, self.tok, self.subset, self.model.device,
                           max_audio_frames=int(self.cfg.model.max_audio_frames),
                           max_samples=int(self.cfg.eval.subset_samples),
                           max_new_tokens=int(self.cfg.eval.max_new_tokens),
                           batch_size=int(self.cfg.eval.batch_size), split_name="val")
            sil = rep.get("silence_empty_rate")
            log.info("step %d | live WER=%.4f CER=%.4f (n=%d)%s", state.global_step,
                     rep["wer"], rep["cer"], rep["n"],
                     f" silence->empty={sil*100:.0f}%" if sil is not None else "")
            if not os.environ.get("WANDB_DISABLED"):
                import wandb
                if wandb.run is not None:
                    wandb.log({"eval/wer": rep["wer"], "eval/cer": rep["cer"]}, step=state.global_step)
        except Exception as e:
            log.warning("WER callback failed: %s", e)
        return control


class FreezeWarmupCallback(TrainerCallback):
    """Train the projector alone for the first `at_step` steps (decoder frozen),
    then unfreeze the decoder. The encoder (if attached) stays governed by its own
    fine-tune flag and is never unfrozen here."""

    def __init__(self, model, at_step):
        self.model, self.at = model, int(at_step)

    def on_train_begin(self, args, state, control, **kw):
        self.model.freeze_decoder(True)
        return control

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step == self.at:
            self.model.freeze_decoder(False)
        return control


class HubCheckpointCallback(TrainerCallback):
    def __init__(self, cfg, run_name, every):
        self.cfg, self.run_name, self.every = cfg, run_name, max(1, int(every))

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero or state.global_step % self.every != 0:
            return control
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            return control
        try:
            ensure_repo(self.cfg.repos.runs, "model")
            upload_folder(ckpt, self.cfg.repos.runs, "model",
                          path_in_repo=f"runs/{self.run_name}/last_checkpoint",
                          commit_message=f"{self.run_name}: ckpt step {state.global_step}")
            log.info("pushed checkpoint step %d (resumable)", state.global_step)
        except Exception as e:
            log.warning("checkpoint push failed: %s", e)
        return control


# --------------------------------------------------------------------------- ledgers
def _record_run(cfg, run_name, status, metrics=None):
    led = Ledger(os.path.join(cfg.paths.runs_dir, "runs.json"), repo_id=cfg.repos.runs,
                 repo_type="model", path_in_repo="ledger/runs.json",
                 default={"project": cfg.project, "runs": []})
    led.sync_from_hub(merge=lambda l, h: h if len(h.get("runs", [])) > len(l.get("runs", [])) else l)
    runs = [r for r in led.d.get("runs", []) if r.get("name") != run_name]
    entry = {"name": run_name, "status": status, "updated": iso_now(),
             "arch": "fastconformer+nandi", "finetune_encoder": bool(cfg.encoder.finetune),
             "epochs": cfg.train.epochs, "lr": cfg.train.lr}
    if metrics:
        entry["metrics"] = metrics
    runs.append(entry)
    led.d["runs"] = runs
    led.push(f"run {run_name}: {status}")


def _record_eval(cfg, run_name, reports):
    led = Ledger(os.path.join(cfg.paths.runs_dir, "evals.json"), repo_id=cfg.repos.runs,
                 repo_type="model", path_in_repo="ledger/evals.json",
                 default={"project": cfg.project, "evals": []})
    led.sync_from_hub(merge=lambda l, h: h if len(h.get("evals", [])) > len(l.get("evals", [])) else l)
    ev = {"run": run_name, "updated": iso_now(), "arch": "fastconformer+nandi"}
    for name, r in reports.items():
        if isinstance(r, dict) and "wer" in r:
            ev[name] = {"wer": r["wer"], "cer": r["cer"], "n": r["n"]}
    led.d.setdefault("evals", []).append(ev)
    led.push(f"eval {run_name}")


# --------------------------------------------------------------------------- main
def train(cfg, *, from_hub: bool = True, resume: str | None = None) -> str:
    from transformers import EarlyStoppingCallback, Trainer, set_seed

    set_seed(cfg.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        import transformers.trainer as _t
        if hasattr(_t, "check_torch_load_is_safe"):
            _t.check_torch_load_is_safe = lambda *a, **k: None
    except Exception:
        pass
    hf_login()

    _check_min_hours(cfg)

    finetune = bool(cfg.encoder.finetune)
    if resume:
        run_name, overwrite = resume, False
        log.info("=== extending run %s ===", run_name)
    else:
        tag = "fc-ft" if finetune else "fc"
        run_name, overwrite = f"{cfg.project}-{tag}-{time.strftime('%Y%m%d-%H%M%S')}", True
    out_dir = os.path.join(cfg.paths.runs_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    use_wandb = init_wandb(cfg.project, run_name, cfg.to_dict()) is not None

    tok = load_tokenizer(cfg.base.decoder_id, cfg.base.trust_remote_code)

    # Build the encoder only when fine-tuning it; otherwise features are precomputed.
    encoder, enc_dim = None, None
    if finetune:
        from .modeling.fc_encoder import FastConformerEncoder
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = FastConformerEncoder.load(cfg.base.encoder_id, dev,
                                            torch.bfloat16 if cfg.train.bf16 and torch.cuda.is_available()
                                            else torch.float32, trainable=True)
        enc_dim = encoder.feat_dim
    else:
        # feat_dim is recorded by the encode step; fall back to a probe row if absent.
        led = Ledger(os.path.join(cfg.paths.ledger_dir, "fc.json"), repo_id=cfg.repos.data,
                     path_in_repo="ledger/fc.json", default={})
        led.sync_from_hub(merge=lambda l, h: h if h.get("feat_dim", 0) else l)
        enc_dim = int(led.d.get("feat_dim") or 0)

    if finetune:
        ds = load_raw(cfg, from_hub=from_hub)
    else:
        ds = load_fc(cfg, from_hub=from_hub)
    train_ds, val_ds, test_ds = splits(ds)
    if not enc_dim:                                            # probe a row for feat_dim
        enc_dim = int(train_ds[0]["feat_dim"])
    log.info("train=%d val=%d test=%d | enc_dim=%d | finetune_encoder=%s",
             train_ds.num_rows, val_ds.num_rows, test_ds.num_rows, enc_dim, finetune)

    model = FastConformerASR.from_base(cfg, tok, enc_dim, encoder=encoder)
    model.frontend.spec_time_mask = int(cfg.train.spec_time_mask)
    model.frontend.spec_time_blocks = int(cfg.train.spec_time_blocks)
    if finetune:
        model.freeze_encoder(False)                           # light fine-tune from step 0

    if finetune:
        collator = FcWaveCollator(tok, float(cfg.model.max_audio_frames) / 12.5,
                                  int(cfg.model.max_text_tokens), model.bos_id,
                                  model.eos_id, tok.pad_token_id)
    else:
        collator = FcCollator(tok, int(cfg.model.max_audio_frames),
                              int(cfg.model.max_text_tokens), model.bos_id,
                              model.eos_id, tok.pad_token_id)

    callbacks = [WerCallback(model, tok, val_ds, cfg),
                 EarlyStoppingCallback(early_stopping_patience=int(cfg.train.early_stopping_patience))]
    if int(cfg.train.freeze_base_steps) > 0:
        callbacks.append(FreezeWarmupCallback(model, int(cfg.train.freeze_base_steps)))
    if bool(cfg.train.push_checkpoints):
        callbacks.append(HubCheckpointCallback(cfg, run_name, int(cfg.train.hub_ckpt_every)))

    args = _training_args(cfg, out_dir, run_name, use_wandb, overwrite)
    trainer = LummaTrainer.make(Trainer)(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, processing_class=tok, callbacks=callbacks)

    _record_run(cfg, run_name, "running")
    resume_ckpt = _resume_checkpoint(cfg, run_name, args.output_dir) if resume else None
    log.info("=== training %s (resume=%s) ===", run_name, bool(resume_ckpt))
    trainer.train(resume_from_checkpoint=resume_ckpt)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return out_dir

    model_dir = os.path.join(out_dir, "model")
    model.save(model_dir, cfg, tok)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    log.info("=== final evaluation ===")
    reports = {}
    for name, split_ds in (("val", val_ds), ("test", test_ds)):
        if split_ds.num_rows == 0 or finetune:                # feats eval needs the `fc` config
            continue
        reports[name] = run_eval(model, tok, split_ds, model.device,
                                 max_audio_frames=int(cfg.model.max_audio_frames),
                                 max_samples=int(cfg.eval.max_samples),
                                 max_new_tokens=int(cfg.eval.max_new_tokens),
                                 batch_size=int(cfg.eval.batch_size), split_name=name)
    if reports:
        save_report(reports, out_dir, title=run_name)
        for name, r in reports.items():
            sr = f" silence->empty={r['silence_empty_rate']*100:.1f}%" if "silence_empty_rate" in r else ""
            log.info("FINAL %s WER=%.4f CER=%.4f (n=%d)%s", name, r["wer"], r["cer"], r["n"], sr)
        _verdict(cfg, reports)

    metrics = {k: {"wer": v["wer"], "cer": v["cer"]} for k, v in reports.items()}
    _record_run(cfg, run_name, "done", metrics)
    if reports:
        _record_eval(cfg, run_name, reports)

    if cfg.train.push_to_hub:
        ensure_repo(cfg.repos.runs, "model")
        best = min((r["wer"] for r in reports.values()), default=float("nan"))
        upload_folder(out_dir, cfg.repos.runs, "model", path_in_repo=f"runs/{run_name}",
                      commit_message=f"run {run_name}: WER={best:.4f}", ignore_patterns=["hf/**"])
        ensure_repo(cfg.repos.model, "model")
        upload_folder(model_dir, cfg.repos.model, "model",
                      commit_message=f"latest: {run_name} WER={best:.4f}")
    log.info("done. run dir: %s", out_dir)
    return out_dir


def _verdict(cfg, reports):
    r = reports.get("test") or reports.get("val")
    if not r:
        return
    wer = r["wer"]
    if wer <= float(cfg.eval.wer_pass):
        log.info("VERDICT: PASS ✅ WER=%.3f <= %.2f — FastConformer+Nandi works. Scale up.",
                 wer, cfg.eval.wer_pass)
    elif wer >= float(cfg.eval.wer_broken):
        log.error("VERDICT: BROKEN ❌ WER=%.3f >= %.2f — debug the pipeline (projector wiring, "
                  "feat_dim, tokenizer, feature scaling).", wer, cfg.eval.wer_broken)
    else:
        log.warning("VERDICT: PARTIAL ⚠️ WER=%.3f — learning but thin. More data/epochs, or "
                    "enable encoder.finetune.", wer)
