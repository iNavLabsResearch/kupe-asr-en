"""Stage 2 — Mimi-encode the `raw` config into the `mimi` config (c0..c7 codes).

Reads ONLY from the Hub (never local raw): downloads one `raw` bunch parquet at a
time, decodes its clips, batches them by frame budget, runs kyutai/mimi to get all
8 codebooks, and writes `mimi` shards that are rolled into ~`mimi_target_shards`
bunches on the Hub. Resumable via the mimi ledger (which raw bunches are done).

We always encode all 8 codebooks so the training A/B (per_frame_sum c0-c7 vs
flatten c0-c3) can be run without re-encoding.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
from tqdm import tqdm

from ..audio import decode_bytes
from ..constants import CONFIG_MIMI, CONFIG_RAW, MIMI_FRAME_RATE
from ..env import ensure_repo, hf_token, log, require_token
from ..hub import CommitPacer, list_config_parquets, next_bunch_index
from ..ledger import Ledger, new_mimi_ledger
from .bunch import compact_to_bunches, upload_bunches
from .shards import MIMI_SCHEMA, ShardWriter, list_local_shards


def _paths(cfg):
    m = cfg.paths.mimi_dir
    return {
        "shards": os.path.join(m, "shards"),
        "bunches": os.path.join(m, "bunches"),
        "dl": os.path.join(m, "dl"),
        "ledger": os.path.join(cfg.paths.ledger_dir, "mimi.json"),
    }


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _frames_of(dur: float) -> int:
    return max(1, int(math.ceil(dur * MIMI_FRAME_RATE)))


def _load(cfg):
    led = Ledger(_paths(cfg)["ledger"], repo_id=cfg.repos.data,
                 path_in_repo="ledger/mimi.json", default=new_mimi_ledger(cfg))
    led.sync_from_hub(merge=lambda l, h: h if len(h.get("raw_files_done", [])) >
                      len(l.get("raw_files_done", [])) else l)
    return led


def status(cfg) -> dict:
    ensure_repo(cfg.repos.data, "dataset")
    token = require_token()
    raw_bunches, _ = list_config_parquets(cfg.repos.data, CONFIG_RAW, token)
    led = _load(cfg)
    done = set(led.d["raw_files_done"])
    left = [f for f in raw_bunches if f not in done]
    log.info("encode: %d/%d raw bunches done, %d LEFT | encoded=%.1fh clips=%d hub_bunches=%d",
             len(done), len(raw_bunches), len(left), led.d["encoded_hours"],
             led.d["clips"], led.d["shards"]["hub_bunches"])
    return {"total": len(raw_bunches), "done": len(done), "left": len(left)}


def encode(cfg) -> str:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import MimiModel

    ensure_repo(cfg.repos.data, "dataset")
    token = require_token()
    P = _paths(cfg)
    for d in (P["shards"], P["bunches"], P["dl"]):
        os.makedirs(d, exist_ok=True)

    device = _pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    n_cb = int(cfg.mimi.num_codebooks)
    budget = int(cfg.mimi.batch_max_frames)
    log.info("Mimi encode on %s | codebooks=%d | frame budget=%d", device, n_cb, budget)
    mimi = MimiModel.from_pretrained(cfg.mimi.model_id, token=hf_token()).to(device).eval()

    raw_bunches, _ = list_config_parquets(cfg.repos.data, CONFIG_RAW, token)
    if not raw_bunches:
        raise RuntimeError(f"{cfg.repos.data} has no `raw` bunches — run fetch first.")
    led = _load(cfg)
    done = set(led.d["raw_files_done"])
    todo = [f for f in raw_bunches if f not in done]
    log.info("%d/%d raw bunches done, %d LEFT", len(done), len(raw_bunches), len(todo))
    if not todo:
        log.info("nothing to encode."); return cfg.repos.data

    pacer = CommitPacer(float(cfg.data.commit_min_interval_s))
    # even rows/bunch so final mimi file count ~= mimi_target_shards. Use the true
    # total clip count from the data ledger (falls back to a rough estimate).
    data_led = Ledger(os.path.join(cfg.paths.ledger_dir, "data.json"), repo_id=cfg.repos.data,
                      path_in_repo="ledger/data.json", default={})
    data_led.sync_from_hub()
    total_clips = int((data_led.d.get("totals") or {}).get("clips", 0)) or \
        int(float(cfg.data.target_hours) * 3600 / 10.0)     # ~10s/clip fallback
    rows_per_bunch = max(int(cfg.data.shard_rows),
                         total_clips // int(cfg.data.mimi_target_shards) or int(cfg.data.shard_rows))
    bunch_idx = max(int(led.d["shards"].get("hub_bunches", 0)),
                    next_bunch_index([f"{CONFIG_MIMI}/data/{b}" for b in led.d["shards"].get("bunch_files", [])]))

    pending: list[str] = []
    pending_rows = 0

    def _upload_wave(force=False):
        nonlocal pending, pending_rows, bunch_idx
        if not pending or (not force and pending_rows < rows_per_bunch):
            return
        bunches = compact_to_bunches(pending, P["bunches"], rows_per_bunch, start_index=bunch_idx,
                                     soft_gb=float(cfg.data.bunch_soft_gb))
        if cfg.mimi.push and bunches:
            led.save()
            upload_bunches(bunches, cfg.repos.data, CONFIG_MIMI, pacer, drop_local=True,
                           extra_file=(P["ledger"], "ledger/mimi.json"))
            bunch_idx += len(bunches)
            led.d["shards"]["hub_bunches"] = bunch_idx
            led.d["shards"]["bunch_files"] = [f"bunch_{i:05d}.parquet" for i in range(bunch_idx)]
        for p in pending:
            try:
                os.remove(p)
            except OSError:
                pass
        pending, pending_rows = [], 0
        led.save()

    def on_flush(path, idx, nrows):
        nonlocal pending, pending_rows
        pending.append(path)
        pending_rows += nrows
        led.d["next_shard_index"] = idx + 1
        led.save()
        if cfg.mimi.push:
            _upload_wave()

    writer = ShardWriter(P["shards"], MIMI_SCHEMA, int(cfg.data.shard_rows),
                         start_index=int(led.d.get("next_shard_index", 0)), on_flush=on_flush)
    for _, p in list_local_shards(P["shards"]):
        if p not in pending:
            pending.append(p); pending_rows += int(cfg.data.shard_rows)

    @torch.inference_mode()
    def _encode_batch(arrays, metas):
        if not arrays:
            return
        maxlen = max(a.shape[0] for a in arrays)
        iv = torch.zeros(len(arrays), 1, maxlen, dtype=torch.float32)
        for i, a in enumerate(arrays):
            iv[i, 0, : a.shape[0]] = torch.from_numpy(a)
        iv = iv.to(device)
        try:
            if device == "cuda":
                with torch.autocast("cuda", dtype=dtype):
                    codes = mimi.encode(iv, num_quantizers=n_cb).audio_codes
            else:
                codes = mimi.encode(iv, num_quantizers=n_cb).audio_codes
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and len(arrays) > 1:
                if device == "cuda":
                    torch.cuda.empty_cache()
                m = len(arrays) // 2
                _encode_batch(arrays[:m], metas[:m]); _encode_batch(arrays[m:], metas[m:])
                return
            raise
        codes = codes.to("cpu").numpy()          # [B, n_cb, T]
        for j, r in enumerate(metas):
            nf = min(codes.shape[2], _frames_of(r["duration"]))
            cb = codes[j, :n_cb, :nf].astype(np.int32)
            writer.add({
                "id": r["id"], "source": r["source"], "text": r["text"],
                "duration": float(r["duration"]), "split": r["split"],
                "num_frames": int(nf), "num_codebooks": int(n_cb),
                "codes": [cb[c].tolist() for c in range(n_cb)],
            })

    import pyarrow.parquet as pq
    processed_h = float(led.d["encoded_hours"])
    for bi, af in enumerate(todo):
        tmp = tempfile.mkdtemp(dir=P["dl"])
        try:
            local = hf_hub_download(cfg.repos.data, af, repo_type="dataset", token=token,
                                    local_dir=tmp)
            pf = pq.ParquetFile(local)
            clips = []
            for b in pf.iter_batches(batch_size=64,
                                     columns=["id", "source", "text", "duration",
                                              "split", "sr", "audio_format", "audio_bytes"]):
                for rec in b.to_pylist():
                    try:
                        arr, _sr = decode_bytes(rec["audio_bytes"])
                        clips.append((arr, rec))
                    except Exception as e:
                        log.debug("decode fail %s: %s", rec.get("id"), e)
            clips.sort(key=lambda c: len(c[0]))     # length-sort -> minimal padding
            arrays, metas, frames = [], [], 0
            for arr, rec in tqdm(clips, desc=f"encode {os.path.basename(af)} [{bi+1}/{len(todo)}]",
                                 leave=False):
                nf = _frames_of(rec["duration"])
                if arrays and frames + nf > budget:
                    _encode_batch(arrays, metas)
                    processed_h += sum(m["duration"] for m in metas) / 3600.0
                    led.d["clips"] += len(metas)
                    led.d["encoded_hours"] = round(processed_h, 4)
                    arrays, metas, frames = [], [], 0
                arrays.append(arr); metas.append(rec); frames += nf
            if arrays:
                _encode_batch(arrays, metas)
                processed_h += sum(m["duration"] for m in metas) / 3600.0
                led.d["clips"] += len(metas)
                led.d["encoded_hours"] = round(processed_h, 4)
            led.d["raw_files_done"] = sorted(set(led.d["raw_files_done"]) | {af})
            led.save()
            log.info("bunch %s done | encoded=%.1fh clips=%d", os.path.basename(af),
                     led.d["encoded_hours"], led.d["clips"])
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    writer.close()
    if cfg.mimi.push:
        _upload_wave(force=True)
    led.push("encode: update mimi ledger")
    log.info("encode DONE | encoded=%.1fh clips=%d hub_bunches=%d -> %s [mimi]",
             led.d["encoded_hours"], led.d["clips"], led.d["shards"]["hub_bunches"], cfg.repos.data)
    return cfg.repos.data
