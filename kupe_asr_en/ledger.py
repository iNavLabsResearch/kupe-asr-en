"""Ledgers: durable JSON state for every stage so nothing is ever recomputed.

There are four ledgers, each a plain JSON file mirrored to the Hub so any box can
resume exactly where another left off:

  data.json   (data repo)  raw-hours collected per source, shards, disk bytes
  mimi.json   (data repo)  encoded-hours, which raw files are done, mimi shards
  runs.json   (runs repo)  one entry per training run (config hash, status, metrics)
  evals.json  (runs repo)  one entry per evaluation (WER/CER per split)

Design:
  * a Ledger is a dict-backed JSON store with ATOMIC writes (temp file + rename)
    so a crash mid-write can never corrupt the file;
  * `sync_from_hub` merges the Hub copy into the local copy before work starts;
  * fingerprints (millions of them) live in a sidecar text file, never in JSON.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

from .env import hf_token, log, require_token


# --------------------------------------------------------------------------- io
def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)          # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("ledger read failed (%s): %s", path, e)
        return None


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------ Ledger class
class Ledger:
    """A JSON store with a local path and an optional Hub mirror location."""

    def __init__(self, local_path: str, repo_id: str | None = None,
                 repo_type: str = "dataset", path_in_repo: str | None = None,
                 default: dict | None = None):
        self.local_path = local_path
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.path_in_repo = path_in_repo or os.path.basename(local_path)
        self.d: dict = read_json(local_path) or dict(default or {})

    # ---- persistence ----
    def save(self) -> None:
        self.d["updated"] = iso_now()
        atomic_write_json(self.local_path, self.d)

    def push(self, message: str = "update ledger") -> None:
        """Commit this ledger to the Hub (one small file = one cheap commit)."""
        if not self.repo_id:
            return
        from huggingface_hub import upload_file
        self.save()
        try:
            upload_file(path_or_fileobj=self.local_path, path_in_repo=self.path_in_repo,
                        repo_id=self.repo_id, repo_type=self.repo_type,
                        token=require_token(), commit_message=message)
        except Exception as e:
            log.warning("ledger push failed (%s): %s", self.path_in_repo, e)

    def sync_from_hub(self, merge=None) -> "Ledger":
        """Pull the Hub copy and merge it into the local dict (for cross-box resume).

        `merge(local_dict, hub_dict) -> merged_dict` customises the merge; the
        default keeps whichever copy has more collected hours.
        """
        if not self.repo_id:
            return self
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(self.repo_id, self.path_in_repo, repo_type=self.repo_type,
                                token=hf_token())
            hub = read_json(p) or {}
        except Exception:
            hub = {}
        if not hub:
            return self
        self.d = (merge or _merge_by_hours)(self.d, hub)
        self.save()
        return self


def _merge_by_hours(local: dict, hub: dict) -> dict:
    """Default merge: take the copy with more total kept hours; union files_done."""
    lk = float((local.get("totals") or {}).get("kept_h", 0))
    hk = float((hub.get("totals") or {}).get("kept_h", 0))
    base = dict(hub) if hk > lk else dict(local)
    other = local if base is not local and base is hub else hub
    # union per-source files_done so neither box re-reads finished parquet files
    b_src = base.setdefault("sources", {})
    for name, o in (other.get("sources") or {}).items():
        b = b_src.setdefault(name, {})
        b["files_done"] = max(int(b.get("files_done", 0)), int(o.get("files_done", 0)))
    return base


# ---------------------------------------------------------------- fingerprints
class SeenSet:
    """Exact-once clip fingerprints, backed by an append-only text sidecar file."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.s: set[str] = set()
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                self.s = {line.strip() for line in f if line.strip()}
        self._fh = open(path, "a", encoding="utf-8")

    def __contains__(self, fp: str) -> bool:
        return fp in self.s

    def add(self, fp: str) -> None:
        if fp not in self.s:
            self.s.add(fp)
            self._fh.write(fp + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def __len__(self) -> int:
        return len(self.s)


# ------------------------------------------------------------ default schemas
def new_data_ledger(cfg) -> dict:
    caps = cfg.data.source_caps.to_dict() if hasattr(cfg.data.source_caps, "to_dict") \
        else dict(cfg.data.source_caps)
    return {
        "project": cfg.project, "stage": "raw", "languages": list(cfg.data.languages),
        "target_hours": float(cfg.data.target_hours), "min_hours": float(cfg.data.min_hours),
        "audio_format": cfg.data.audio_format,
        "sources": {name: {"cap_h": float(cap), "kept_h": 0.0, "clips": 0,
                           "files_done": 0, "status": "pending"}
                    for name, cap in caps.items() if float(cap) > 0},
        "totals": {"kept_h": 0.0, "train_h": 0.0, "val_h": 0.0, "test_h": 0.0,
                   "clips": 0, "bytes": 0},
        "silence": {"clips": 0, "hours": 0.0, "done": False},
        "shards": {"local": 0, "hub_bunches": 0, "bunch_files": []},
        "next_shard_index": 0, "next_clip_index": 0,
        "updated": iso_now(),
    }


def new_mimi_ledger(cfg) -> dict:
    return {
        "project": cfg.project, "stage": "mimi",
        "num_codebooks": int(cfg.mimi.num_codebooks),
        "raw_files_done": [], "encoded_hours": 0.0, "clips": 0,
        "shards": {"local": 0, "hub_bunches": 0, "bunch_files": []},
        "next_shard_index": 0, "next_bunch_index": 0,
        "updated": iso_now(),
    }
