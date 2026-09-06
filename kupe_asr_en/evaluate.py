"""Evaluation: greedy-decode English text from Mimi codes and score WER/CER.

Used both as a live training callback (small subset) and as the end-of-run eval
over the held-out val/test splits. Produces a JSON + Markdown report.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .env import log
from .text import normalize


def _codes_batch(rows, num_codebooks, max_audio_frames):
    """Rows -> (codes Long[B,K,T], num_frames Long[B])."""
    cbs, nf = [], []
    for r in rows:
        cb = np.asarray(r["codes"], dtype=np.int64)
        if cb.ndim == 1:
            cb = cb[None, :]
        cb = cb[:num_codebooks, :max_audio_frames]
        cbs.append(cb)
        nf.append(cb.shape[1])
    K = max(c.shape[0] for c in cbs)
    T = max(c.shape[1] for c in cbs)
    out = np.zeros((len(rows), K, T), dtype=np.int64)
    for i, c in enumerate(cbs):
        out[i, : c.shape[0], : c.shape[1]] = c
    return torch.from_numpy(out), torch.tensor(nf, dtype=torch.long)


@torch.inference_mode()
def _generate(model, tok, rows, device, num_codebooks, max_audio_frames,
              max_new_tokens, batch_size):
    hyps = []
    for i in tqdm(range(0, len(rows), batch_size), desc="generate", leave=False):
        chunk = rows[i:i + batch_size]
        codes, nframes = _codes_batch(chunk, num_codebooks, max_audio_frames)
        prefixes = model.build_prefix(codes, nframes)              # list of [Lp, E]
        Lmax = max(p.shape[0] for p in prefixes)
        E = prefixes[0].shape[1]
        inp = torch.zeros(len(prefixes), Lmax, E, dtype=prefixes[0].dtype, device=device)
        attn = torch.zeros(len(prefixes), Lmax, dtype=torch.long, device=device)
        for b, p in enumerate(prefixes):                          # LEFT-pad for generation
            L = p.shape[0]
            inp[b, Lmax - L:] = p
            attn[b, Lmax - L:] = 1
        gen = model.lumma.generate(
            inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=model.eos_id,
            pad_token_id=tok.pad_token_id)
        for b in range(len(prefixes)):
            ids = gen[b].tolist()
            if model.eos_id in ids:
                ids = ids[: ids.index(model.eos_id)]
            hyps.append(normalize(tok.decode(ids, skip_special_tokens=True)))
    return hyps


def _wer_cer(refs, hyps):
    import jiwer
    if not refs:
        return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    return {"wer": float(jiwer.wer(refs, hyps)), "cer": float(jiwer.cer(refs, hyps)), "n": len(refs)}


def run_eval(model, tok, ds, device, *, num_codebooks, max_audio_frames,
             max_samples, max_new_tokens, batch_size, split_name="val") -> dict:
    model.eval()
    n = min(max_samples, ds.num_rows)
    rows = ds.shuffle(seed=0).select(range(n)) if ds.num_rows > n else ds
    rows = [rows[i] for i in range(rows.num_rows)]
    refs = [normalize(r["text"]) for r in rows]
    hyps = _generate(model, tok, rows, device, num_codebooks, max_audio_frames,
                     max_new_tokens, batch_size)

    # separate real speech from silence clips (empty reference) so empty refs don't
    # corrupt WER, and so we can score silence -> empty directly.
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


def render_markdown(reports: dict, title: str) -> str:
    lines = [f"# {title}", "", "| split | WER | CER | n | silence→empty |",
             "|---|---|---|---|---|"]
    for name, r in reports.items():
        if isinstance(r, dict) and "wer" in r:
            sil = (f"{r['silence_empty_rate']*100:.1f}% (n={r['silence_n']})"
                   if "silence_empty_rate" in r else "—")
            lines.append(f"| {name} | {r['wer']:.4f} | {r['cer']:.4f} | {r['n']} | {sil} |")
    for name, r in reports.items():
        if isinstance(r, dict) and r.get("samples"):
            lines += ["", f"## {name} samples", ""]
            for s in r["samples"]:
                lines.append(f"- ref: `{s['ref']}`\n  hyp: `{s['hyp']}`")
    return "\n".join(lines) + "\n"


def save_report(reports: dict, out_dir: str, title: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "eval_report.json")
    mp = os.path.join(out_dir, "eval_report.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(render_markdown(reports, title))
    log.info("eval report -> %s", out_dir)
    return jp, mp
