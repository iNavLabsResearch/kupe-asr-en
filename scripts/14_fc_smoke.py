#!/usr/bin/env python
"""Offline smoke test for the FastConformer + Nandi track (no downloads, no GPU).

Validates the parts that don't need the real encoder/decoder weights:
  1. FC_SCHEMA parquet round-trip (feats stored as float16 bytes).
  2. FcCollator: rows -> padded feats/text tensors.
  3. fc_evaluate._feats_batch reshape.
  4. FeatureFrontend forward shapes (mlp + linear).
  5. FastConformerASR.forward + build_prefix sequence assembly, with a TINY stub
     decoder that mimics the HF causal-LM interface (embeds + CE loss).

Run: python scripts/14_fc_smoke.py
"""
import _bootstrap  # noqa: F401
import io
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from kupe_asr_en.data.shards import FC_SCHEMA
from kupe_asr_en.fc_collate import FcCollator
from kupe_asr_en.fc_evaluate import _feats_batch
from kupe_asr_en.modeling.fc_frontend import FeatureFrontend
from kupe_asr_en.modeling.fc_asr_model import FastConformerASR

D = 512          # FastConformer feat_dim
E = 320          # decoder input embed dim (stub)
VOCAB = 200


def _fake_rows(n=5):
    rows = []
    for i in range(n):
        T = np.random.randint(5, 40)
        feats = np.random.randn(T, D).astype(np.float16)
        rows.append({"id": f"c{i}", "source": "librispeech", "text": "hello world",
                     "duration": T / 12.5, "split": "train",
                     "num_frames": T, "feat_dim": D, "feats": feats.tobytes()})
    return rows


def test_schema_roundtrip():
    rows = _fake_rows(3)
    tbl = pa.Table.from_pylist(rows, schema=FC_SCHEMA)
    buf = io.BytesIO()
    pq.write_table(tbl, buf, compression="zstd")
    back = pq.read_table(io.BytesIO(buf.getvalue())).to_pylist()
    r = back[0]
    f = np.frombuffer(r["feats"], dtype="<f2").reshape(-1, r["feat_dim"])
    assert f.shape == (rows[0]["num_frames"], D), f.shape
    print("  [1] FC_SCHEMA parquet round-trip OK  (feats %s)" % (f.shape,))


class _StubTok:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, add_special_tokens=False):
        ids = [3 + (ord(c) % (VOCAB - 5)) for c in text.split()[0]][:8]
        return SimpleNamespace(input_ids=ids)


def test_collator():
    tok = _StubTok()
    coll = FcCollator(tok, max_audio_frames=750, max_text_tokens=200,
                      bos_id=1, eos_id=2, pad_id=0)
    batch = coll(_fake_rows(5))
    assert batch["feats"].dtype == torch.float32
    assert batch["feats"].shape[0] == 5 and batch["feats"].shape[2] == D
    assert batch["text_ids"].shape == batch["labels"].shape
    assert batch["num_frames"].shape == (5,)
    print("  [2] FcCollator OK  (feats %s, text_ids %s)" %
          (tuple(batch["feats"].shape), tuple(batch["text_ids"].shape)))
    return batch


def test_feats_batch():
    feats, nf = _feats_batch(_fake_rows(4), max_audio_frames=750)
    assert feats.shape[0] == 4 and feats.shape[2] == D and nf.shape == (4,)
    print("  [3] _feats_batch reshape OK  (feats %s)" % (tuple(feats.shape),))


def test_frontend():
    for proj in ("mlp", "linear"):
        fe = FeatureFrontend(D, E, projector=proj)
        out = fe(torch.randn(3, 17, D))
        assert out.shape == (3, 17, E), (proj, out.shape)
    print("  [4] FeatureFrontend forward OK  (mlp + linear -> [B,T,%d])" % E)


class _StubDecoder(nn.Module):
    """Minimal HF-causal-LM stand-in: input embeddings + a CE loss head."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, E)
        self.head = nn.Linear(E, VOCAB)
        self.config = SimpleNamespace(hidden_size=E, vocab_size=VOCAB)

    def get_input_embeddings(self):
        return self.embed

    @property
    def device(self):
        return self.embed.weight.device

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **_):
        logits = self.head(inputs_embeds)                    # [B, L, V]
        loss = None
        if labels is not None:
            sl = logits[:, :-1].reshape(-1, VOCAB)
            tl = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(sl, tl, ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)


def test_model_forward(batch):
    dec = _StubDecoder()
    fe = FeatureFrontend(D, E, projector="mlp")
    model = FastConformerASR(dec, fe, bos_id=1, eos_id=2, max_audio_frames=750)
    out = model(text_ids=batch["text_ids"], text_lengths=batch["text_lengths"],
                feats=batch["feats"], num_frames=batch["num_frames"], labels=batch["labels"])
    assert out.loss is not None and torch.isfinite(out.loss), out.loss
    out.loss.backward()                                      # grads must flow to projector + markers
    assert model.frontend.proj[0].weight.grad is not None
    assert model.frontend.audio_bos.grad is not None
    prefixes = model.build_prefix(feats=batch["feats"][:2], num_frames=batch["num_frames"][:2])
    assert len(prefixes) == 2 and prefixes[0].shape[1] == E
    print("  [5] FastConformerASR forward+backward+build_prefix OK  (loss=%.3f)" % float(out.loss))


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    print("FastConformer+Nandi smoke test")
    test_schema_roundtrip()
    b = test_collator()
    test_feats_batch()
    test_frontend()
    test_model_forward(b)
    print("ALL SMOKE CHECKS PASSED ✅")
