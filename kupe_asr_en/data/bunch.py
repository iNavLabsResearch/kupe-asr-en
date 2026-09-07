"""Pack many tiny local shards into ~N big `bunch_*.parquet` files and upload them.

Why: the Hub free tier allows 128 commits/hour and 1000 download requests / 5 min.
Thousands of tiny shards would blow both. So we compact to ~`target_shards` files
(default 24) and upload them one paced commit each — a whole dataset is ~25 commits.

Compaction streams row-groups through a ParquetWriter, so writing a multi-GB bunch
never loads it fully into memory.
"""
from __future__ import annotations

import os

import pyarrow.parquet as pq

from ..env import log, require_token
from ..hub import CommitPacer, commit_with_backoff
from .shards import list_local_shards


def _total_rows(shard_paths: list[str]) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in shard_paths)


def _readable_shards(shard_paths: list[str]) -> list[str]:
    """Drop (and delete) any shard parquet left corrupt by a crash mid-write, so
    compaction never dies on a truncated file. Lost clips are re-fetchable / silence
    is regenerable, so removing a bad shard is safe."""
    good = []
    for p in shard_paths:
        try:
            pq.ParquetFile(p).metadata.num_rows
            good.append(p)
        except Exception as e:
            log.warning("dropping corrupt shard %s (%s)", os.path.basename(p), e)
            try:
                os.remove(p)
            except OSError:
                pass
    return good


def compact_to_bunches(shard_paths: list[str], out_dir: str, target_shards: int,
                       start_index: int = 0, batch_rows: int = 512,
                       soft_gb: float = 6.0) -> list[str]:
    """Stream `shard_paths` into ~`target_shards` bunch parquet files in `out_dir`.

    Returns the list of written bunch file paths. Rows are distributed evenly by
    count; a warning fires if a bunch exceeds `soft_gb` (raise target_shards or use
    audio_format=opus if so).
    """
    shard_paths = _readable_shards(shard_paths)
    if not shard_paths:
        return []
    os.makedirs(out_dir, exist_ok=True)
    total = _total_rows(shard_paths)
    per_bunch = max(1, -(-total // max(1, target_shards)))   # ceil
    schema = pq.ParquetFile(shard_paths[0]).schema_arrow

    written: list[str] = []
    idx = start_index
    writer = None
    in_bunch = 0

    def _open(i):
        path = os.path.join(out_dir, f"bunch_{i:05d}.parquet")
        return pq.ParquetWriter(path, schema, compression="zstd"), path

    for sp in shard_paths:
        pf = pq.ParquetFile(sp)
        for batch in pf.iter_batches(batch_size=batch_rows):
            if writer is None:
                writer, cur = _open(idx)
                in_bunch = 0
            writer.write_batch(batch)
            in_bunch += batch.num_rows
            if in_bunch >= per_bunch:
                writer.close()
                sz = os.path.getsize(cur) / 1e9
                if sz > soft_gb:
                    log.warning("bunch %s is %.1f GB (> soft cap %.1f GB) — consider "
                                "raising target_shards or audio_format=opus",
                                os.path.basename(cur), sz, soft_gb)
                log.info("wrote %s (%d rows, %.2f GB)", os.path.basename(cur), in_bunch, sz)
                written.append(cur)
                idx += 1
                writer = None
    if writer is not None:
        writer.close()
        written.append(cur)
        log.info("wrote %s (%d rows, %.2f GB)", os.path.basename(cur), in_bunch,
                 os.path.getsize(cur) / 1e9)
    return written


def upload_bunches(bunch_paths: list[str], repo_id: str, config_name: str,
                   pacer: CommitPacer, drop_local: bool = True,
                   extra_file: tuple[str, str] | None = None) -> int:
    """Commit each bunch to `<config>/data/` (one paced commit each). Returns count.

    `extra_file=(local_path, path_in_repo)` piggybacks a ledger snapshot onto the
    FIRST commit so state and data advance together.
    """
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=require_token())
    n = 0
    for k, bp in enumerate(bunch_paths):
        name = os.path.basename(bp)
        ops = [CommitOperationAdd(f"{config_name}/data/{name}", bp)]
        if k == 0 and extra_file:
            ops.append(CommitOperationAdd(extra_file[1], extra_file[0]))
        pacer.wait()
        commit_with_backoff(
            lambda ops=ops, name=name: api.create_commit(
                repo_id=repo_id, repo_type="dataset", operations=ops,
                commit_message=f"{config_name}: {name}"),
            f"{config_name} bunch commit")
        log.info("uploaded %s -> %s [%s]", name, repo_id, config_name)
        if drop_local:
            try:
                os.remove(bp)
            except OSError:
                pass
        n += 1
    return n


def dataset_from_parquets(paths: list[str]):
    """Build a memory-mapped HF Dataset from parquet files."""
    from datasets import Dataset, concatenate_datasets
    if not paths:
        raise RuntimeError("no parquet files to load")
    return concatenate_datasets([Dataset.from_parquet(p) for p in paths])


def prune_uploaded_shards(shards_dir: str) -> None:
    """Delete local tiny shards once compaction+upload succeeded."""
    for _, p in list_local_shards(shards_dir):
        try:
            os.remove(p)
        except OSError:
            pass
