"""English ASR source registry — the factory (rule: adding a dataset is one entry).

To add a new English source, append a `Source(...)` to REGISTRY. If the dataset
has a normal `audio`+`text` parquet schema, no code is needed — column names are
auto-detected. If its schema is exotic, subclass `Source` and override
`extract(record) -> (array, sr, text)`; everything else (file listing, resume,
streaming, dedup) is inherited.

All sources were fact-checked on the Hub, 2026-09:
  openslr/librispeech_asr    ungated   all/train.clean.100|360, all/train.other.500
  MLCommons/peoples_speech   ungated   clean/train-*, dirty/train-*   (CC-BY)
  ai4bharat/Svarah           gated:auto data/*                        (Indian-accent EN)
  speechcolab/gigaspeech     gated      <config>/*                    (off by default)

A gated/unaccepted source is logged and skipped, never fatal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from ..audio import decode_audio
from ..env import log

_TEXT_CANDIDATES = [
    "text", "sentence", "transcript", "transcription", "normalized_text",
    "normalized", "verbatim", "raw_text", "target", "clean_text",
]


class SkipSource(Exception):
    """(source) cannot be read right now — caller logs and moves on."""


@dataclass
class Source:
    name: str
    hf_id: str
    include: list[str]                       # a parquet path is used if any substring matches
    exclude: list[str] = field(default_factory=list)
    audio_col: str | None = None             # None => auto-detect from schema
    text_col: str | None = None
    trust_remote_code: bool = False
    note: str = ""

    # ---- override point for exotic schemas -------------------------------
    def extract(self, record: dict, audio_col: str, text_col: str):
        """record -> (float32 array, sr, text). Override for non-standard rows."""
        text = record.get(text_col)
        if not text or not str(text).strip():
            return None, None, None
        arr, sr = decode_audio(record.get(audio_col))
        if arr is None:
            return None, None, None
        return arr, sr, str(text).strip()


# --------------------------------------------------------------------------- registry
REGISTRY: dict[str, Source] = {
    "librispeech": Source(
        name="librispeech", hf_id="openslr/librispeech_asr",
        include=["train.clean.100/", "train.clean.360/", "train.other.500/"],
        audio_col="audio", text_col="text",
        note="clean+other read English, ~960h, ungated",
    ),
    "peoples_speech": Source(
        name="peoples_speech", hf_id="MLCommons/peoples_speech",
        include=["clean/train-", "dirty/train-"], audio_col="audio", text_col="text",
        note="diverse English, CC-BY, ungated (subset via cap)",
    ),
    "svarah": Source(
        name="svarah", hf_id="ai4bharat/Svarah",
        include=["data/"], note="Indian-accented English, ~9.6h, gated:auto",
    ),
    "gigaspeech": Source(
        name="gigaspeech", hf_id="speechcolab/gigaspeech",
        include=["/train"], exclude=["dev", "test"], trust_remote_code=True,
        note="multi-domain English, gated (accept the license, set cap>0)",
    ),
}


def get_source(name: str) -> Source:
    if name not in REGISTRY:
        raise KeyError(f"unknown source '{name}' — add it to sources.REGISTRY")
    return REGISTRY[name]


# ------------------------------------------------------------------ schema helpers
def _cols_from_schema(schema, src: Source) -> tuple[str | None, str | None]:
    import pyarrow as pa
    lower = {n.lower(): n for n in schema.names}
    text = src.text_col or next((lower[c] for c in _TEXT_CANDIDATES if c in lower), None)
    audio = src.audio_col
    if audio is None:
        for n in schema.names:
            t = schema.field(n).type
            if n == "audio" or (pa.types.is_struct(t) and
                                any(t.field(i).name == "bytes" for i in range(t.num_fields))):
                audio = n
                break
    return audio, text


def list_parquet_files(src: Source, token: str) -> list[str]:
    from huggingface_hub import HfApi
    files = HfApi(token=token).list_repo_files(src.hf_id, repo_type="dataset")
    out = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        if src.include and not any(inc in f for inc in src.include):
            continue
        if src.exclude and any(exc in f for exc in src.exclude):
            continue
        out.append(f)
    return sorted(out)


def iter_examples(src: Source, token: str, start_file: int = 0) -> Iterator[tuple]:
    """Yield (array, sr, text, file_idx) per clip and a (None,None,None,i+1) marker
    after each parquet file is fully consumed (so resume is per-file, O(1) skip).

    One file at a time is downloaded to disk and deleted after use -> bounded RAM.
    """
    import os
    import shutil
    import tempfile

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    dl_root = os.environ.get("KUPE_DL_DIR") or os.path.join(os.getcwd(), ".kupe_dl")
    os.makedirs(dl_root, exist_ok=True)

    try:
        files = list_parquet_files(src, token)
    except Exception as e:
        raise SkipSource(f"{src.name}: cannot list files ({e})")
    if not files:
        raise SkipSource(f"{src.name}: no parquet matched include={src.include}")
    log.info("open %s [%s]: %d parquet files, start at %d", src.name, src.hf_id,
             len(files), start_file)

    for i in range(start_file, len(files)):
        tmp = tempfile.mkdtemp(dir=dl_root)
        try:
            try:
                local = hf_hub_download(src.hf_id, files[i], repo_type="dataset",
                                        token=token, local_dir=tmp)
            except Exception as e:
                log.warning("download %s failed: %s", files[i], e)
                yield None, None, None, i + 1
                continue
            try:
                pf = pq.ParquetFile(local)
                audio_col, text_col = _cols_from_schema(pf.schema_arrow, src)
                if not audio_col or not text_col:
                    log.warning("cols not found in %s (audio=%s text=%s)",
                                files[i], audio_col, text_col)
                else:
                    for batch in pf.iter_batches(batch_size=32, columns=[audio_col, text_col]):
                        for rec in batch.to_pylist():
                            try:
                                arr, sr, text = src.extract(rec, audio_col, text_col)
                                if arr is not None:
                                    yield arr, sr, text, i
                            except Exception as e:
                                log.debug("row skip %s: %s", files[i], e)
            except Exception as e:
                log.warning("read %s failed: %s", files[i], e)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        yield None, None, None, i + 1
