"""Bounded-memory local parquet shard writer.

Rows accumulate in RAM and flush to `shard_{idx:05d}.parquet` every `shard_rows`
rows, so peak memory is ~one shard. These tiny shards are LOCAL only — they get
packed into a handful of big `bunch_*.parquet` files (see bunch.py) before ever
touching the Hub, which is what keeps us far under the 128-commit/hour limit.
"""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

from ..constants import (CONFIG_FC, CONFIG_MIMI, CONFIG_RAW, SPLIT_TEST,
                         SPLIT_TRAIN, SPLIT_VAL)
from ..env import log

# ------------------------------------------------------------------- schemas
RAW_SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("text", pa.string()),
    ("duration", pa.float32()), ("split", pa.string()),
    ("sr", pa.int32()), ("audio_format", pa.string()), ("audio_bytes", pa.binary()),
])

MIMI_SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("text", pa.string()),
    ("duration", pa.float32()), ("split", pa.string()),
    ("num_frames", pa.int32()), ("num_codebooks", pa.int32()),
    ("codes", pa.list_(pa.list_(pa.int32()))),   # codebook-major [num_cb][num_frames]
])

# FastConformer encoder features. `feats` holds a float16 [num_frames, feat_dim]
# array as raw little-endian bytes (compact + robust; parquet float16 support is
# spotty). Reader = np.frombuffer(feats, "<f2").reshape(num_frames, feat_dim).
FC_SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("text", pa.string()),
    ("duration", pa.float32()), ("split", pa.string()),
    ("num_frames", pa.int32()), ("feat_dim", pa.int32()), ("feats", pa.binary()),
])

SCHEMAS = {CONFIG_RAW: RAW_SCHEMA, CONFIG_MIMI: MIMI_SCHEMA, CONFIG_FC: FC_SCHEMA}
VALID_SPLITS = {SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST}


class ShardWriter:
    def __init__(self, out_dir: str, schema: pa.Schema, shard_rows: int,
                 start_index: int = 0, on_flush=None, note_fn=None):
        self.out_dir = out_dir
        self.schema = schema
        self.shard_rows = int(shard_rows)
        self._next_idx = int(start_index)
        self.on_flush = on_flush          # (path, idx, nrows) -> None
        self.note_fn = note_fn            # () -> str, appended to each flush log line
        self.rows: list[dict] = []
        os.makedirs(out_dir, exist_ok=True)

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.shard_rows:
            self.flush()

    def _take_index(self) -> int:
        while True:
            p = os.path.join(self.out_dir, f"shard_{self._next_idx:05d}.parquet")
            if not os.path.exists(p):
                idx = self._next_idx
                self._next_idx += 1
                return idx
            self._next_idx += 1

    def flush(self) -> None:
        if not self.rows:
            return
        idx = self._take_index()
        path = os.path.join(self.out_dir, f"shard_{idx:05d}.parquet")
        n = len(self.rows)
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        pq.write_table(table, path, compression="zstd")
        self.rows = []
        note = ""
        if self.note_fn:
            try:
                note = self.note_fn() or ""
            except Exception:
                note = ""
        log.info("flushed %s (%d rows)%s", os.path.basename(path), n, note)
        if self.on_flush:
            self.on_flush(path, idx, n)

    def close(self) -> None:
        self.flush()


def list_local_shards(shards_dir: str) -> list[tuple[int, str]]:
    if not os.path.isdir(shards_dir):
        return []
    out = []
    for name in os.listdir(shards_dir):
        if name.startswith("shard_") and name.endswith(".parquet"):
            try:
                out.append((int(name[len("shard_"):-len(".parquet")]),
                            os.path.join(shards_dir, name)))
            except ValueError:
                pass
    return sorted(out)
