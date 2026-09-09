"""Stage 2 (FC track) — FastConformer-encode the `raw` config into `fc` features.

Same pipelined, resumable, multi-GPU design as data/encode.py (Mimi), but the GPU
runs the FastConformer encoder instead of Mimi and emits CONTINUOUS features
(float16 [T, D]) rather than discrete codes:

  [downloader] prefetch raw bunches (bounded by disk)
     -> [decoder pool] FLAC-decode into length-sorted, frame-budget batches
        -> [GPU pool] FastConformer-encode across ALL GPUs (round-robin, OOM auto-split)
           -> [uploader] pack `fc` shards into bunches, commit (paced), delete locally

Resumable via the fc ledger (which raw bunches are done). Reads ONLY from the Hub.
Run inside tmux; a killed run resumes at the next unfinished raw bunch. This is a
distinct, parallelizable step: encode once, then train (fc_train) many times.
"""
from __future__ import annotations

import math
import os
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..audio import decode_bytes
from ..constants import CONFIG_FC, CONFIG_RAW, FC_FRAME_RATE
from ..env import ensure_repo, hf_token, log, require_token
from ..hub import CommitPacer, list_config_parquets, next_bunch_index
from ..ledger import Ledger, new_fc_ledger
from ..modeling.fc_encoder import FastConformerEncoder
from .bunch import compact_to_bunches, upload_bunches
from .shards import FC_SCHEMA, ShardWriter, list_local_shards

_DONE = object()


def _paths(cfg):
    m = cfg.paths.fc_dir
    return {"shards": os.path.join(m, "shards"), "bunches": os.path.join(m, "bunches"),
            "dl": os.path.join(m, "dl"), "ledger": os.path.join(cfg.paths.ledger_dir, "fc.json")}


def _frames_of(dur: float) -> int:
    return max(1, int(math.ceil(dur * FC_FRAME_RATE)))


def _devices():
    import torch
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def _load(cfg):
    led = Ledger(_paths(cfg)["ledger"], repo_id=cfg.repos.data,
                 path_in_repo="ledger/fc.json", default=new_fc_ledger(cfg))
    led.sync_from_hub(merge=lambda l, h: h if len(h.get("raw_files_done", [])) >
                      len(l.get("raw_files_done", [])) else l)
    return led


def status(cfg) -> dict:
    ensure_repo(cfg.repos.data, "dataset")
    raw, _ = list_config_parquets(cfg.repos.data, CONFIG_RAW, require_token())
    led = _load(cfg)
    done = set(led.d["raw_files_done"])
    left = [f for f in raw if f not in done]
    log.info("fc-encode: %d/%d raw bunches done, %d LEFT | encoded=%.1fh clips=%d fc_bunches=%d",
             len(done), len(raw), len(left), led.d["encoded_hours"], led.d["clips"],
             led.d["shards"]["hub_bunches"])
    return {"total": len(raw), "done": len(done), "left": len(left)}


def encode(cfg) -> str:
    import torch
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ensure_repo(cfg.repos.data, "dataset")
    token = require_token()
    P = _paths(cfg)
    for d in P.values():
        if not d.endswith(".json"):
            os.makedirs(d, exist_ok=True)

    devices = _devices()
    nd = len(devices)
    use_cuda = devices[0].startswith("cuda")
    dtype = torch.float16 if use_cuda else torch.float32
    budget = int(cfg.fc.batch_max_frames)
    budget_total = budget * nd
    log.info("FastConformer encode on %s | batch=%d frames/GPU", devices, budget)
    models = [FastConformerEncoder.load(cfg.base.encoder_id, dv, dtype, trainable=False)
              for dv in devices]
    feat_dim = models[0].feat_dim
    led = _load(cfg)
    led.d["feat_dim"] = feat_dim

    raw_bunches, _ = list_config_parquets(cfg.repos.data, CONFIG_RAW, token)
    if not raw_bunches:
        raise RuntimeError(f"{cfg.repos.data} has no `raw` bunches — run fetch first.")
    done = set(led.d["raw_files_done"])
    todo = [f for f in raw_bunches if f not in done]
    log.info("%d/%d raw bunches done, %d LEFT | feat_dim=%d", len(done), len(raw_bunches),
             len(todo), feat_dim)
    if not todo:
        log.info("nothing to encode."); return cfg.repos.data

    pacer = CommitPacer(float(cfg.data.commit_min_interval_s))
    rows_per_bunch = max(int(cfg.data.shard_rows), int(getattr(cfg.fc, "fc_rows", 20000)))
    bunch_idx = max(int(led.d["shards"].get("hub_bunches", 0)),
                    next_bunch_index([f"{CONFIG_FC}/data/{b}" for b in led.d["shards"].get("bunch_files", [])]))
    pending: list[str] = []
    pending_rows = 0
    lock = threading.Lock()

    def _upload_wave(force=False):
        nonlocal pending, pending_rows, bunch_idx
        if not pending or (not force and pending_rows < rows_per_bunch):
            return
        take, pending, pending_rows = pending, [], 0
        bunches = compact_to_bunches(take, P["bunches"], rows_per_bunch, start_index=bunch_idx,
                                     soft_gb=float(cfg.data.bunch_soft_gb))
        if cfg.fc.push and bunches:
            led.save()
            upload_bunches(bunches, cfg.repos.data, CONFIG_FC, pacer, drop_local=True,
                           extra_file=(P["ledger"], "ledger/fc.json"))
            bunch_idx += len(bunches)
            led.d["shards"]["hub_bunches"] = bunch_idx
            led.d["shards"]["bunch_files"] = [f"bunch_{i:05d}.parquet" for i in range(bunch_idx)]
        for p in take:
            try:
                os.remove(p)
            except OSError:
                pass
        led.save()

    def on_flush(path, idx, nrows):
        nonlocal pending_rows
        pending.append(path)
        pending_rows += nrows
        led.d["next_shard_index"] = idx + 1

    writer = ShardWriter(P["shards"], FC_SCHEMA, int(cfg.data.shard_rows),
                         start_index=int(led.d.get("next_shard_index", 0)), on_flush=on_flush)
    for _, p in list_local_shards(P["shards"]):
        if p not in pending:
            pending.append(p); pending_rows += int(cfg.data.shard_rows)

    decode_pool = ThreadPoolExecutor(max_workers=int(getattr(cfg.fc, "decode_workers", 16)))
    gpu_pool = ThreadPoolExecutor(max_workers=nd)

    # ---- GPU encode (multi-GPU, OOM auto-split) ----
    def _part(gi, arrays, srs, metas):
        if not arrays:
            return []
        model = models[gi]
        try:
            feats = model.encode_arrays(arrays, srs, devices[gi], use_cuda)
        except Exception as e:
            if use_cuda:
                torch.cuda.empty_cache()
            if _oom(e) and len(arrays) > 1:
                m = len(arrays) // 2
                return (_part(gi, arrays[:m], srs[:m], metas[:m]) +
                        _part(gi, arrays[m:], srs[m:], metas[m:]))
            if _oom(e):
                return []
            raise
        rows = []
        for j, r in enumerate(metas):
            f = feats[j]                                        # [T, D] float16
            nf = min(f.shape[0], _frames_of(r["duration"]))
            f = np.ascontiguousarray(f[:nf], dtype=np.float16)
            rows.append({"id": r["id"], "source": r["source"], "text": r["text"],
                         "duration": float(r["duration"]), "split": r["split"],
                         "num_frames": int(nf), "feat_dim": int(f.shape[1]),
                         "feats": f.tobytes()})
        return rows

    def gpu_encode(arrays, srs, metas):
        step = math.ceil(len(arrays) / nd)
        futs = [gpu_pool.submit(_part, min(k, nd - 1), arrays[i:i + step], srs[i:i + step],
                                metas[i:i + step])
                for k, i in enumerate(range(0, len(arrays), step))]
        with lock:
            for f in futs:
                for row in f.result():
                    writer.add(row)

    # ---- downloader thread ----
    dlq: queue.Queue = queue.Queue(maxsize=int(getattr(cfg.fc, "prefetch", 1)))

    def downloader():
        for af in todo:
            d = tempfile.mkdtemp(dir=P["dl"])
            try:
                local = hf_hub_download(cfg.repos.data, af, repo_type="dataset", token=token, local_dir=d)
                dlq.put((af, d, local))
            except Exception as e:
                shutil.rmtree(d, ignore_errors=True)
                log.warning("download %s failed: %s", af, e)
        dlq.put(_DONE)

    # ---- decoder thread ----
    batchq: queue.Queue = queue.Queue(maxsize=max(2, 2 * nd))

    def _decode(rec):
        try:
            a, sr = decode_bytes(rec["audio_bytes"])
            return (np.asarray(a, np.float32) if a is not None else None), sr, rec
        except Exception:
            return None, None, rec

    chunk_clips = int(getattr(cfg.fc, "decode_chunk", 3000))

    def _emit(clips):
        clips.sort(key=lambda c: len(c[0]))            # length-sort within chunk -> less padding
        arrays, srs, metas, frames = [], [], [], 0
        for arr, sr, rec in clips:
            nf = _frames_of(rec["duration"])
            if arrays and frames + nf > budget_total:
                batchq.put(("batch", arrays, srs, metas)); arrays, srs, metas, frames = [], [], [], 0
            arrays.append(arr); srs.append(sr); metas.append(rec); frames += nf
        if arrays:
            batchq.put(("batch", arrays, srs, metas))

    def decoder():
        cols = ["id", "source", "text", "duration", "split", "audio_bytes"]
        while True:
            item = dlq.get()
            if item is _DONE:
                break
            af, d, local = item
            try:
                pf = pq.ParquetFile(local)
                chunk = []
                for b in pf.iter_batches(batch_size=64, columns=cols):
                    for arr, sr, rec in decode_pool.map(_decode, b.to_pylist()):
                        if arr is not None and rec.get("duration"):
                            chunk.append((arr, sr, rec))
                            if len(chunk) >= chunk_clips:
                                _emit(chunk); chunk = []
                if chunk:
                    _emit(chunk)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            batchq.put(("file", af))
        batchq.put(_DONE)

    threading.Thread(target=downloader, daemon=True).start()
    threading.Thread(target=decoder, daemon=True).start()

    # ---- main GPU consumer ----
    processed, base, total = 0, len(done), len(raw_bunches)
    hours = float(led.d["encoded_hours"])
    t0 = time.time()
    while True:
        item = batchq.get()
        if item is _DONE:
            break
        if item[0] == "batch":
            _, arrays, srs, metas = item
            gpu_encode(arrays, srs, metas)
            hours += sum(m["duration"] for m in metas) / 3600.0
            led.d["clips"] += len(metas)
            led.d["encoded_hours"] = round(hours, 4)
        else:                                          # a raw bunch finished
            af = item[1]
            led.d["raw_files_done"] = sorted(set(led.d["raw_files_done"]) | {af})
            led.save()
            if use_cuda:
                torch.cuda.empty_cache()
            _upload_wave()
            processed += 1
            el = max(1e-6, time.time() - t0)
            rate = processed / (el / 3600)
            eta = (len(todo) - processed) / rate if rate else 0
            log.info("bunch %s done | %d/%d | %.1f h encoded | %.1f bunches/h | ETA %.1f h",
                     os.path.basename(af), base + processed, total, led.d["encoded_hours"], rate, eta)

    writer.close()
    _upload_wave(force=True)
    for pl in (gpu_pool, decode_pool):
        pl.shutdown(wait=True)
    led.push("fc-encode: update fc ledger")
    log.info("fc-encode DONE | %.1f h, %d clips, %d fc bunches -> %s [fc]",
             led.d["encoded_hours"], led.d["clips"], led.d["shards"]["hub_bunches"], cfg.repos.data)
    return cfg.repos.data


def _oom(e):
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
