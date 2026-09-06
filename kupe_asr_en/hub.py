"""Hub I/O with the free-tier limits baked in (rule: never trip HF rate limits).

Limits we respect:
  * 128 commits / hour            -> `CommitPacer` enforces a min gap between commits
    and, because we pack local shards into ~20-25 big "bunch" parquet files, a
    whole dataset is only ~25 commits total.
  * 1000 download requests / 5 min -> we only ever download the handful of bunch
    files (via snapshot_download with a tight allow_patterns), never thousands
    of tiny shards.

`commit_with_backoff` also retries 429/5xx with exponential backoff so a transient
limit hit pauses instead of crashing a multi-hour job.
"""
from __future__ import annotations

import time

from .env import log, require_token


class CommitPacer:
    """Sleeps so consecutive commits are at least `min_interval_s` apart."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0

    def wait(self) -> None:
        gap = time.time() - self._last
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        self._last = time.time()


def commit_with_backoff(fn, what: str, retries: int = 6, base: float = 5.0):
    """Run a Hub-committing callable, retrying rate limits / transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = ("429" in msg or "too many requests" in msg or "rate" in msg
                         or "500" in msg or "502" in msg or "503" in msg or "timeout" in msg)
            if not transient or attempt == retries - 1:
                raise
            wait = base * (2 ** attempt)
            log.warning("%s hit '%s' -> retry %d/%d in %.0fs", what, str(e)[:80],
                        attempt + 1, retries, wait)
            time.sleep(wait)


def list_config_parquets(repo_id: str, config_name: str, token: str | None = None):
    """Return (bunch_files, shard_files) parquet paths under `<config>/data/`."""
    from huggingface_hub import HfApi
    files = HfApi(token=token or require_token()).list_repo_files(repo_id, repo_type="dataset")
    pref = f"{config_name}/data/"
    pq = [f for f in files if f.startswith(pref) and f.endswith(".parquet")]
    bunches = sorted(f for f in pq if "/bunch_" in f)
    shards = sorted(f for f in pq if "/shard_" in f)
    return bunches, shards


def next_bunch_index(bunch_files: list[str]) -> int:
    mx = -1
    for f in bunch_files:
        base = f.rsplit("/", 1)[-1]
        if base.startswith("bunch_") and base.endswith(".parquet"):
            try:
                mx = max(mx, int(base[len("bunch_"):-len(".parquet")]))
            except ValueError:
                pass
    return mx + 1


def download_bunches(repo_id: str, config_name: str, token: str | None = None) -> list[str]:
    """Download only the bunch parquets of a config; return local file paths."""
    import glob
    import os

    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    token = token or require_token()
    bunches, shards = list_config_parquets(repo_id, config_name, token)
    if not bunches and shards:
        raise RuntimeError(
            f"{repo_id} [{config_name}] has {len(shards)} un-bunched tiny parquet files.\n"
            "Downloading them will trip the 1000-request/5-min limit. Re-run the compact "
            "step of the owning stage so the Hub only holds bunch_*.parquet.")
    if not bunches:
        raise FileNotFoundError(f"{repo_id} has no parquet under {config_name}/data/")
    local = snapshot_download(repo_id, repo_type="dataset", token=token,
                              allow_patterns=[f"{config_name}/data/bunch_*.parquet"],
                              max_workers=min(8, len(bunches)))
    found = sorted(glob.glob(f"{local}/{config_name}/data/bunch_*.parquet"))
    log.info("downloaded %d %s bunch parquet(s) from %s", len(found), config_name, repo_id)
    return found
