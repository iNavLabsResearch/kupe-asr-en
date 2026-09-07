"""Stage 1 — collect English audio into the `raw` config on the Hub.

Stream each registered source (bounded download: we stop at each source's hour
cap), resample to 24 kHz mono, compress (flac/opus), dedup, assign a fixed
train/val/test split, and write local parquet shards. Shards are rolled into big
`bunch_*.parquet` files and uploaded so the Hub only ever holds ~`raw_target_shards`
files and local disk stays bounded to about one bunch.

Everything is tracked in the data ledger (per-source hours, files_done, shards,
bytes) so a killed job resumes exactly where it stopped. Fingerprints in a sidecar
give exact-once dedup across restarts and across sources.
"""
from __future__ import annotations

import gc
import hashlib
import os
import random

import numpy as np
from tqdm import tqdm

from ..audio import encode_bytes, make_silence, to_mono_24k
from ..constants import (CONFIG_RAW, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL)
from ..env import ensure_repo, log, require_token
from ..hub import CommitPacer, next_bunch_index
from ..ledger import Ledger, SeenSet, new_data_ledger
from ..text import is_probably_valid, normalize
from .bunch import compact_to_bunches, upload_bunches
from .shards import RAW_SCHEMA, ShardWriter, list_local_shards
from .sources import SkipSource, get_source, iter_examples


def _free_memory() -> None:
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def clip_fp(source: str, text_norm: str, dur: float) -> str:
    h = hashlib.sha1(f"{source}|{text_norm}|{dur:.2f}".encode("utf-8")).hexdigest()
    return h[:16]


def _paths(cfg):
    raw = cfg.paths.raw_dir
    return {
        "shards": os.path.join(raw, "shards"),
        "bunches": os.path.join(raw, "bunches"),
        "ledger": os.path.join(cfg.paths.ledger_dir, "data.json"),
        "seen": os.path.join(cfg.paths.ledger_dir, "seen_raw.txt"),
    }


def _rows_per_bunch(cfg) -> int:
    """Rows per upload wave. Small = fast, frequent, crash-safe uploads (each bunch
    logs when written); a killed run then loses at most one small in-flight bunch."""
    return max(int(cfg.data.shard_rows), int(getattr(cfg.data, "wave_rows", 6000)))


def _load(cfg):
    ledger = Ledger(_paths(cfg)["ledger"], repo_id=cfg.repos.data,
                    path_in_repo="ledger/data.json", default=new_data_ledger(cfg))
    ledger.sync_from_hub()
    # make sure every configured source with cap>0 has a ledger entry
    caps = cfg.data.source_caps.to_dict() if hasattr(cfg.data.source_caps, "to_dict") \
        else dict(cfg.data.source_caps)
    for name, cap in caps.items():
        if float(cap) > 0:
            ledger.d["sources"].setdefault(name, {"cap_h": float(cap), "kept_h": 0.0,
                                                  "clips": 0, "files_done": 0, "status": "pending"})
            ledger.d["sources"][name]["cap_h"] = float(cap)
    return ledger


def status(cfg) -> dict:
    ensure_repo(cfg.repos.data, "dataset")
    ledger = _load(cfg)
    d = ledger.d
    t = d["totals"]
    log.info("=== data ledger (%s) ===", cfg.repos.data)
    for name, s in d["sources"].items():
        log.info("  %-16s %6.1f/%6.1fh  clips=%-8d files_done=%-4d  %s",
                 name, s["kept_h"], s["cap_h"], s["clips"], s["files_done"], s["status"])
    log.info("  TOTAL kept=%.1fh (train=%.1f val=%.1f test=%.1f) clips=%d  target=%.0fh min=%.0fh",
             t["kept_h"], t["train_h"], t["val_h"], t["test_h"], t["clips"],
             d["target_hours"], d["min_hours"])
    sil = d.get("silence") or {}
    log.info("  SILENCE clips=%d (%.2fh) labeled \"\" for anti-hallucination",
             int(sil.get("clips", 0)), float(sil.get("hours", 0.0)))
    log.info("  hub bunches=%d  (%.1f%% of target)", d["shards"]["hub_bunches"],
             100.0 * t["kept_h"] / max(1e-9, d["target_hours"]))
    if t["kept_h"] < d["min_hours"]:
        log.warning("  below min_hours (%.0f) — training will REFUSE to start", d["min_hours"])
    return d


def fetch(cfg, *, reset: bool = False) -> str:
    require_token()
    ensure_repo(cfg.repos.data, "dataset")
    P = _paths(cfg)
    os.makedirs(P["shards"], exist_ok=True)
    os.makedirs(P["bunches"], exist_ok=True)

    ledger = _load(cfg)
    if reset:
        log.warning("--reset: zeroing ledger hour counts (Hub parquet not deleted)")
        ledger.d = new_data_ledger(cfg)
    seen = SeenSet(P["seen"])
    rng = random.Random(cfg.seed)

    d = ledger.d
    t = d["totals"]
    val_budget = float(cfg.data.val_hours) * 3600.0
    test_budget = float(cfg.data.test_hours) * 3600.0
    rows_per_bunch = _rows_per_bunch(cfg)
    pacer = CommitPacer(float(cfg.data.commit_min_interval_s))

    start_idx = int(d.get("next_shard_index", 0))
    bunch_idx = max(int(d["shards"].get("hub_bunches", 0)),
                    next_bunch_index([f for f in d["shards"].get("bunch_files", [])]))

    # ---- rolling bunch uploader: pack+push whenever enough rows accumulate ----
    pending: list[str] = []
    pending_rows = 0

    def _upload_wave(force=False):
        nonlocal pending, pending_rows, bunch_idx
        if not pending or (not force and pending_rows < rows_per_bunch):
            return
        bunches = compact_to_bunches(pending, P["bunches"], rows_per_bunch,
                                     start_index=bunch_idx, soft_gb=float(cfg.data.bunch_soft_gb))
        if cfg.data.push and bunches:
            ledger.save()
            upload_bunches(bunches, cfg.repos.data, CONFIG_RAW, pacer, drop_local=True,
                           extra_file=(P["ledger"], "ledger/data.json"))
            bunch_idx += len(bunches)
            d["shards"]["hub_bunches"] = bunch_idx
            d["shards"]["bunch_files"] = [f"bunch_{i:05d}.parquet" for i in range(bunch_idx)]
        for p in pending:                        # tiny shards now packed -> remove
            try:
                os.remove(p)
            except OSError:
                pass
        pending, pending_rows = [], 0
        ledger.save()
        _free_memory()

    def on_flush(path, idx, nrows):
        nonlocal pending, pending_rows
        pending.append(path)
        pending_rows += nrows
        d["next_shard_index"] = idx + 1
        d["shards"]["local"] = d["shards"].get("local", 0) + 1
        ledger.save()
        if cfg.data.push:
            _upload_wave()

    writer = ShardWriter(P["shards"], RAW_SCHEMA, int(cfg.data.shard_rows),
                         start_index=start_idx, on_flush=on_flush)

    # pick up any local shards left by a previous crash
    for _, p in list_local_shards(P["shards"]):
        if p not in pending:
            pending.append(p)
            pending_rows += 1  # unknown; treat as small, force-flush handles it

    for name, s in d["sources"].items():
        cap_s = float(s["cap_h"]) * 3600.0
        has_clips = int(s.get("clips", 0)) > 0
        if cap_s <= 0:
            continue
        if s["kept_h"] * 3600.0 >= cap_s:
            s["status"] = "done"
            continue
        # skip only sources that truly finished WITH data; a 'done' source that
        # produced 0 clips (e.g. a gate that was later granted) is retried.
        if s["status"] == "done" and has_clips:
            continue
        try:
            src = get_source(name)
        except KeyError as e:
            log.warning("skip %s: %s", name, e)
            continue

        s["status"] = "in_progress"
        ledger.save()
        got_s = float(s["kept_h"]) * 3600.0
        clips = int(s["clips"])
        files_done = int(s["files_done"])
        if clips == 0:
            files_done = 0        # never yielded anything -> re-read from file 0

        pbar = tqdm(total=cap_s / 3600.0, initial=got_s / 3600.0, unit="h",
                    desc=f"fetch {name}",
                    bar_format="{l_bar}{bar}| {n:.2f}/{total:.1f}h [{elapsed}<{remaining}] {postfix}")
        try:
            it = iter_examples(src, require_token(), start_file=files_done)
        except SkipSource as e:
            log.warning("skip %s: %s", name, e)
            s["status"] = "unavailable"
            ledger.save()
            pbar.close()
            continue

        dups = 0
        while got_s < cap_s:
            try:
                arr, sr, text, file_idx = next(it)
            except StopIteration:
                break
            except Exception as e:
                log.debug("next() %s: %s", name, e)
                continue

            if arr is None:                       # file finished -> save resume point
                files_done = int(file_idx)
                s["files_done"] = files_done
                ledger.save()
                _free_memory()
                continue

            if not is_probably_valid(text):
                continue
            try:
                wav = to_mono_24k(arr, sr, int(cfg.data.target_sr))
            except Exception as e:
                log.debug("resample fail: %s", e)
                continue
            dur = len(wav) / int(cfg.data.target_sr)
            if dur < cfg.data.min_dur or dur > cfg.data.max_dur:
                continue

            text_n = normalize(text)
            fp = clip_fp(name, text_n, dur)
            if fp in seen:
                dups += 1
                if dups % 1000 == 0:
                    log.info("resume %s: skipped %d already-seen clips", name, dups)
                continue

            # split routing: sprinkle val/test across the whole stream until full
            if t["val_h"] * 3600.0 < val_budget and rng.random() < 0.02:
                split = SPLIT_VAL
            elif t["test_h"] * 3600.0 < test_budget and rng.random() < 0.02:
                split = SPLIT_TEST
            else:
                split = SPLIT_TRAIN

            try:
                ab = encode_bytes(wav, int(cfg.data.target_sr), cfg.data.audio_format,
                                  int(getattr(cfg.data, "opus_bitrate", 24000)))
            except Exception as e:
                log.debug("encode_bytes fail: %s", e)
                continue

            writer.add({
                "id": f"en-{name}-{d['next_clip_index']:09d}",
                "source": name, "text": text_n, "duration": float(dur),
                "split": split, "sr": int(cfg.data.target_sr),
                "audio_format": cfg.data.audio_format, "audio_bytes": ab,
            })
            seen.add(fp)
            clips += 1
            d["next_clip_index"] += 1
            got_s += dur
            t["kept_h"] = round(t["kept_h"] + dur / 3600.0, 4)
            t[f"{split}_h"] = round(t.get(f"{split}_h", 0.0) + dur / 3600.0, 4)
            t["clips"] += 1
            t["bytes"] += len(ab)
            s["kept_h"] = round(got_s / 3600.0, 4)
            s["clips"] = clips
            pbar.update(dur / 3600.0)
            if clips % 500 == 0:
                pbar.set_postfix_str(f"{clips} clips")
                seen.flush()
                ledger.save()
        pbar.close()
        s["files_done"] = files_done
        # a 0-clip source stays retryable (gate may be granted later); don't lock it 'done'
        s["status"] = "done" if int(s["clips"]) > 0 else "pending"
        seen.flush()
        ledger.save()
        log.info("source %s %s: %.1f h, %d clips", name, s["status"], s["kept_h"], s["clips"])
        if s["clips"] == 0:
            log.warning("source %s produced 0 clips — check gate/access (will retry next run)", name)

    # === anti-hallucination: inject silence/ambient clips labeled "" ===========
    sil = d.setdefault("silence", {"clips": 0, "hours": 0.0, "done": False})
    frac = float(getattr(cfg.data, "silence_frac", 0.0))
    if frac > 0 and not sil.get("done"):
        want = int(round(frac * max(1, t["clips"])))
        todo_sil = max(0, want - int(sil["clips"]))
        if todo_sil > 0:
            log.info("injecting %d silence clips (%.0f%% of %d real clips) labeled \"\"",
                     todo_sil, frac * 100, t["clips"])
            grng = np.random.default_rng(cfg.seed + 99)
            val_cap = int(getattr(cfg.data, "silence_val_cap", 0))
            sr = int(cfg.data.target_sr)
            lo, hi = float(cfg.data.silence_min_dur), float(cfg.data.silence_max_dur)
            for i in range(todo_sil):
                dur = float(grng.uniform(lo, hi))
                kind = "ambient" if grng.random() < 0.4 else "pure"
                wav = make_silence(dur, sr, kind, grng)
                # a few to val so we can MEASURE silence->empty; the rest train
                split = SPLIT_VAL if int(sil["clips"]) < val_cap else SPLIT_TRAIN
                try:
                    ab = encode_bytes(wav, sr, cfg.data.audio_format,
                                      int(getattr(cfg.data, "opus_bitrate", 24000)))
                except Exception as e:
                    log.debug("silence encode fail: %s", e)
                    continue
                writer.add({
                    "id": f"en-silence-{d['next_clip_index']:09d}",
                    "source": "silence", "text": "", "duration": dur,
                    "split": split, "sr": sr, "audio_format": cfg.data.audio_format,
                    "audio_bytes": ab,
                })
                d["next_clip_index"] += 1
                sil["clips"] += 1
                sil["hours"] = round(sil["hours"] + dur / 3600.0, 4)
                t["clips"] += 1
                t[f"{split}_h"] = round(t.get(f"{split}_h", 0.0) + dur / 3600.0, 4)
                if (i + 1) % 500 == 0:
                    ledger.save()
            sil["done"] = True
            ledger.save()
            log.info("silence injection done: %d clips, %.2f h", sil["clips"], sil["hours"])

    writer.close()
    if cfg.data.push:
        _upload_wave(force=True)
    ledger.push("fetch: update data ledger")

    # ---- target enforcement (rule: don't proceed on a dataset that missed target) ----
    if t["kept_h"] < d["min_hours"]:
        log.error("COLLECTED %.1f h < min_hours %.0f h. Fix source access/caps and re-run "
                  "before encoding/training.", t["kept_h"], d["min_hours"])
    elif t["kept_h"] < d["target_hours"]:
        log.warning("collected %.1f h (target %.0f h) — above min, below target.",
                    t["kept_h"], d["target_hours"])
    else:
        log.info("collected %.1f h (>= target %.0f h) ✓", t["kept_h"], d["target_hours"])
    log.info("fetch done | total=%.1fh clips=%d hub_bunches=%d | repo=%s [raw]",
             t["kept_h"], t["clips"], d["shards"]["hub_bunches"], cfg.repos.data)
    return cfg.repos.data
