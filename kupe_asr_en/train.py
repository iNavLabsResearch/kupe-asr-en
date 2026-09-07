"""Training — HF Trainer over the `mimi` dataset with LummaASR.

Guards, in order:
  1. min-hours gate: refuse to train if encoded hours < data.min_hours (rule).
  2. resume: local checkpoint, else pull the run's last checkpoint from the runs repo.
  3. live WER every eval, early stopping on eval_loss, throttled Hub checkpointing.
End of run: full val+test eval, save weights+frontend+reports, update runs & evals
ledgers, push the run to the runs repo and the best weights to the model repo.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import time

import torch
from transformers import TrainerCallback

from .collate import AsrCollator
from .dataset import load_mimi, splits
from .evaluate import run_eval, save_report
from .env import (ensure_repo, hf_login, hf_token, init_wandb, log, require_token,
                  upload_folder)
from .ledger import Ledger, iso_now
from .modeling.asr_model import LummaASR, load_tokenizer


class LummaTrainer:
    """Mixin factory: a Trainer whose model-save breaks Lumma's tied lm_head/embed
    sharing so safetensors (forced in transformers 5.x) doesn't reject the wrapper."""

    @staticmethod
    def make(base_trainer_cls):
        class _T(base_trainer_cls):
            def _save(self, output_dir=None, state_dict=None):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                # generic untie: clone any tensor that shares storage with an earlier
                # one (safetensors rejects shared storage). Works for ANY tied base model.
                seen, sd, cloned = {}, dict(state_dict), False
                for k, v in list(sd.items()):
                    if not hasattr(v, "data_ptr"):
                        continue
                    ptr = v.data_ptr()
                    if ptr in seen:
                        sd[k] = v.clone(); cloned = True
                    else:
                        seen[ptr] = k
                return super()._save(output_dir, state_dict=sd if cloned else state_dict)
        return _T


# --------------------------------------------------------------------------- gate
def _encoded_hours(cfg) -> float:
    led = Ledger(os.path.join(cfg.paths.ledger_dir, "mimi.json"), repo_id=cfg.repos.data,
                 path_in_repo="ledger/mimi.json", default={})
    led.sync_from_hub(merge=lambda l, h: h if h.get("encoded_hours", 0) > l.get("encoded_hours", 0) else l)
    return float(led.d.get("encoded_hours", 0.0))


def _check_min_hours(cfg):
    have = _encoded_hours(cfg)
    need = float(cfg.data.min_hours)
    if have <= 0:
        log.warning("mimi ledger reports 0 encoded hours — is encode done and pushed?")
    if have < need:
        raise RuntimeError(
            f"REFUSING TO TRAIN: only {have:.1f} h encoded < min_hours {need:.0f} h.\n"
            "Collect/encode more data, or lower data.min_hours in the config if you "
            "intend a smaller smoke run.")
    log.info("min-hours gate OK: %.1f h encoded (>= %.0f h).", have, need)


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
                           num_codebooks=int(self.cfg.audio.codebooks),
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
                    m = {"eval/wer": rep["wer"], "eval/cer": rep["cer"]}
                    if sil is not None:
                        m["eval/silence_empty_rate"] = sil
                    wandb.log(m, step=state.global_step)
        except Exception as e:
            log.warning("WER callback failed: %s", e)
        return control


class FreezeWarmupCallback(TrainerCallback):
    """Train the audio frontend alone for the first `at_step` steps, then unfreeze
    the LM. The optimizer is built (in Trainer.train) BEFORE on_train_begin fires,
    so it already owns all params; frozen params just get no grad until unfrozen."""

    def __init__(self, model, at_step):
        self.model, self.at = model, int(at_step)

    def on_train_begin(self, args, state, control, **kw):
        self.model.freeze_lumma(True)
        return control

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step == self.at:
            self.model.freeze_lumma(False)
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


# --------------------------------------------------------------------------- args
def _training_args(cfg, out_dir, run_name, use_wandb, overwrite):
    import inspect

    from transformers import TrainingArguments
    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    kw = dict(
        output_dir=os.path.join(out_dir, "hf"), overwrite_output_dir=overwrite,
        num_train_epochs=cfg.train.epochs,
        max_steps=int(getattr(cfg.train, "max_steps", 0) or -1),
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=float(cfg.train.lr), weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio, lr_scheduler_type="cosine",
        bf16=bool(cfg.train.bf16) and torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),        # transformers errors on tf32 without a GPU
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        max_grad_norm=float(cfg.train.max_grad_norm),
        label_smoothing_factor=float(cfg.train.label_smoothing),
        gradient_checkpointing=bool(cfg.train.gradient_checkpointing),
        group_by_length=True, length_column_name="num_frames",
        logging_steps=cfg.train.logging_steps, logging_first_step=True,
        eval_strategy="steps", eval_steps=cfg.train.eval_steps,
        prediction_loss_only=True,
        save_strategy="steps", save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        # transformers 5.x forces safetensors, which rejects Lumma's tied lm_head/embed.
        # LummaTrainer._save unshares the tensor before writing (see below).
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        dataloader_num_workers=cfg.train.num_workers, dataloader_pin_memory=True,
        report_to=["wandb"] if use_wandb else ["none"], run_name=run_name,
        remove_unused_columns=False, label_names=["labels"],
        ddp_find_unused_parameters=False, seed=cfg.seed,
    )
    return TrainingArguments(**{k: v for k, v in kw.items() if k in allowed})


def _resume_checkpoint(cfg, run_name, hf_dir):
    local = sorted(glob.glob(os.path.join(hf_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1)
    if local:
        log.info("resuming from local checkpoint %s", local[-1])
        return local[-1]
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(cfg.repos.runs, repo_type="model", token=hf_token(),
                              allow_patterns=[f"runs/{run_name}/last_checkpoint/*"])
        src = os.path.join(p, "runs", run_name, "last_checkpoint")
        if os.path.isdir(src) and os.listdir(src):
            dest = os.path.join(hf_dir, "checkpoint-hub")
            os.makedirs(hf_dir, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)
            log.info("resuming from Hub checkpoint runs/%s/last_checkpoint", run_name)
            return dest
    except Exception as e:
        log.warning("no Hub checkpoint for %s (%s) — starting fresh", run_name, e)
    return None


# --------------------------------------------------------------------------- ledgers
def _record_run(cfg, run_name, status, metrics=None):
    led = Ledger(os.path.join(cfg.paths.runs_dir, "runs.json"), repo_id=cfg.repos.runs,
                 repo_type="model", path_in_repo="ledger/runs.json",
                 default={"project": cfg.project, "runs": []})
    led.sync_from_hub(merge=lambda l, h: h if len(h.get("runs", [])) > len(l.get("runs", [])) else l)
    runs = [r for r in led.d.get("runs", []) if r.get("name") != run_name]
    entry = {"name": run_name, "status": status, "updated": iso_now(),
             "frontend": cfg.audio.frontend, "codebooks": int(cfg.audio.codebooks),
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
    ev = {"run": run_name, "updated": iso_now(),
          "frontend": cfg.audio.frontend, "codebooks": int(cfg.audio.codebooks)}
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
    # Resume reloads our own optimizer/model .bin via torch.load; some transformers
    # builds gate that. The checkpoint is our trusted file, so allow it.
    try:
        import transformers.trainer as _t
        if hasattr(_t, "check_torch_load_is_safe"):
            _t.check_torch_load_is_safe = lambda *a, **k: None
    except Exception:
        pass
    hf_login()

    _check_min_hours(cfg)

    if resume:
        run_name, overwrite = resume, False
        log.info("=== extending run %s ===", run_name)
    else:
        run_name, overwrite = f"{cfg.project}-{cfg.audio.frontend}-{time.strftime('%Y%m%d-%H%M%S')}", True
    out_dir = os.path.join(cfg.paths.runs_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    use_wandb = init_wandb(cfg.project, run_name, cfg.to_dict()) is not None

    tok = load_tokenizer(cfg.base.lumma_id, cfg.base.trust_remote_code)
    model = LummaASR.from_base(cfg, tok)
    model.frontend.spec_time_mask = int(cfg.train.spec_time_mask)
    model.frontend.spec_time_blocks = int(cfg.train.spec_time_blocks)

    ds = load_mimi(cfg, from_hub=from_hub)
    train_ds, val_ds, test_ds = splits(ds)
    log.info("train=%d val=%d test=%d", train_ds.num_rows, val_ds.num_rows, test_ds.num_rows)

    collator = AsrCollator(tok, int(cfg.audio.codebooks), int(cfg.model.max_audio_frames),
                           int(cfg.model.max_text_tokens), model.bos_id, model.eos_id,
                           tok.pad_token_id)

    callbacks = [WerCallback(model, tok, val_ds, cfg),
                 EarlyStoppingCallback(early_stopping_patience=int(cfg.train.early_stopping_patience))]
    if int(cfg.train.freeze_base_steps) > 0:
        # do NOT freeze here — the optimizer must be built with all params first.
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

    # save model + tokenizer + frontend + config
    model_dir = os.path.join(out_dir, "model")
    model.save(model_dir, cfg, tok)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # final eval on val + test
    log.info("=== final evaluation ===")
    reports = {}
    for name, split_ds in (("val", val_ds), ("test", test_ds)):
        if split_ds.num_rows == 0:
            continue
        reports[name] = run_eval(model, tok, split_ds, model.device,
                                 num_codebooks=int(cfg.audio.codebooks),
                                 max_audio_frames=int(cfg.model.max_audio_frames),
                                 max_samples=int(cfg.eval.max_samples),
                                 max_new_tokens=int(cfg.eval.max_new_tokens),
                                 batch_size=int(cfg.eval.batch_size), split_name=name)
    save_report(reports, out_dir, title=run_name)
    for name, r in reports.items():
        sr = f" silence->empty={r['silence_empty_rate']*100:.1f}%" if "silence_empty_rate" in r else ""
        log.info("FINAL %s WER=%.4f CER=%.4f (n=%d)%s", name, r["wer"], r["cer"], r["n"], sr)
    _phase1_verdict(cfg, reports)

    metrics = {k: {"wer": v["wer"], "cer": v["cer"]} for k, v in reports.items()}
    _record_run(cfg, run_name, "done", metrics)
    _record_eval(cfg, run_name, reports)

    if cfg.train.push_to_hub:
        ensure_repo(cfg.repos.runs, "model")
        best = min((r["wer"] for r in reports.values()), default=float("nan"))
        upload_folder(out_dir, cfg.repos.runs, "model", path_in_repo=f"runs/{run_name}",
                      commit_message=f"run {run_name}: WER={best:.4f}",
                      ignore_patterns=["hf/**"])
        ensure_repo(cfg.repos.model, "model")
        upload_folder(model_dir, cfg.repos.model, "model",
                      commit_message=f"latest: {run_name} WER={best:.4f}")
    log.info("done. run dir: %s", out_dir)
    return out_dir


def _phase1_verdict(cfg, reports):
    r = reports.get("test") or reports.get("val")
    if not r:
        return
    wer = r["wer"]
    if wer <= float(cfg.eval.wer_pass):
        log.info("PHASE-1 VERDICT: PASS ✅ WER=%.3f <= %.2f — pipeline + Lumma work. "
                 "Proceed to scale-up.", wer, cfg.eval.wer_pass)
    elif wer >= float(cfg.eval.wer_broken):
        log.error("PHASE-1 VERDICT: BROKEN ❌ WER=%.3f >= %.2f — debug the pipeline "
                  "before anything else (frontend wiring, tokenizer, codes).", wer, cfg.eval.wer_broken)
    else:
        log.warning("PHASE-1 VERDICT: PARTIAL ⚠️ WER=%.3f — learning but thin. More data / "
                    "epochs, or try the other audio.frontend.", wer)
