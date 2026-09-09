"""Evaluation for the FastConformer + Nandi track — greedy-decode from features.

Mirrors evaluate.py:run_eval but builds continuous-feature batches (or waveform
batches, for a fine-tuned encoder). Reuses _wer_cer / normalize / save_report so
the report format is identical to the Mimi track.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .evaluate import _wer_cer, render_markdown, save_report  # noqa: F401 (re-exported)
from .text import normalize


def _feats_batch(rows, max_audio_frames):
    fl, nf = [], []
    for r in rows:
        d = int(r["feat_dim"])
        f = np.frombuffer(r["feats"], dtype="<f2").reshape(-1, d)
        f = np.asarray(f[:max_audio_frames], dtype=np.float32)
        fl.append(f); nf.append(f.shape[0])
    D = fl[0].shape[1]
    T = max(f.shape[0] for f in fl)
    out = np.zeros((len(rows), T, D), dtype=np.float32)
    for i, f in enumerate(fl):
        out[i, : f.shape[0]] = f
    return torch.from_numpy(out), torch.tensor(nf, dtype=torch.long)


@torch.inference_mode()
def _generate(model, tok, rows, device, max_audio_frames, max_new_tokens, batch_size):
    hyps = []
    for i in tqdm(range(0, len(rows), batch_size), desc="generate", leave=False):
        chunk = rows[i:i + batch_size]
        feats, nframes = _feats_batch(chunk, max_audio_frames)
        prefixes = model.build_prefix(feats=feats, num_frames=nframes)
        Lmax = max(p.shape[0] for p in prefixes)
        E = prefixes[0].shape[1]
        inp = torch.zeros(len(prefixes), Lmax, E, dtype=prefixes[0].dtype, device=device)
        attn = torch.zeros(len(prefixes), Lmax, dtype=torch.long, device=device)
        for b, p in enumerate(prefixes):                      # LEFT-pad for generation
            L = p.shape[0]
            inp[b, Lmax - L:] = p
            attn[b, Lmax - L:] = 1
        gen = model.lm.generate(
            inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=model.eos_id,
            pad_token_id=tok.pad_token_id)
        for b in range(len(prefixes)):
            ids = gen[b].tolist()
            if model.eos_id in ids:
                ids = ids[: ids.index(model.eos_id)]
            hyps.append(normalize(tok.decode(ids, skip_special_tokens=True)))
    return hyps


def run_eval(model, tok, ds, device, *, max_audio_frames, max_samples,
             max_new_tokens, batch_size, split_name="val") -> dict:
    model.eval()
    n = min(max_samples, ds.num_rows)
    rows = ds.shuffle(seed=0).select(range(n)) if ds.num_rows > n else ds
    rows = [rows[i] for i in range(rows.num_rows)]
    refs = [normalize(r["text"]) for r in rows]
    hyps = _generate(model, tok, rows, device, max_audio_frames, max_new_tokens, batch_size)

    sp_refs, sp_hyps, sil_hyps = [], [], []
    for r, ref, hyp in zip(rows, refs, hyps):
        if ref.strip() == "" or r.get("source") == "silence":
            sil_hyps.append(hyp)
        else:
            sp_refs.append(ref); sp_hyps.append(hyp)
    rep = _wer_cer(sp_refs, sp_hyps)
    rep["split"] = split_name
    if sil_hyps:
        empty = sum(1 for h in sil_hyps if h.strip() == "")
        rep["silence_n"] = len(sil_hyps)
        rep["silence_empty_rate"] = round(empty / len(sil_hyps), 4)
    rep["samples"] = [{"ref": r, "hyp": h} for r, h in list(zip(sp_refs, sp_hyps))[:12]]
    return rep
