#!/usr/bin/env python
"""Multilingual base-model KNOWLEDGE check (11 languages) — for our own knowledge.

This is NOT part of Phase 1 (which is English only). It measures, per language,
how well Lumma's tokenizer + pretrained weights already handle each script, so we
know what to expect if/when we add Indic languages in a later phase.

Per language it measures: tokenizer fertility (tok/word), round-trip fidelity,
in-vocab script coverage, text-generation script-correctness + repetition, and
perplexity. Writes one JSON per language + an overall JSON into a reports folder.

    python scripts/08_indic_model_check.py
    python scripts/08_indic_model_check.py --model google/gemma-3-1b-pt --out-dir reports/gemma
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os

import numpy as np
import torch

# Unicode block per script (start, end) inclusive
SCRIPT_RANGES = {
    "latin": (0x0041, 0x024F),
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}


def _in_script(ch, script):
    lo, hi = SCRIPT_RANGES[script]
    return lo <= ord(ch) <= hi


def _grade_fertility(lang, f):
    if lang == "en":
        return "✅" if f < 1.5 else "⚠️" if f < 2.0 else "❌"
    return "✅" if f < 2.0 else "⚠️" if f < 3.0 else "❌"


def _grade_ppl(lang, p):
    if lang == "en":
        return "✅" if p < 50 else "⚠️" if p < 100 else "❌"
    return "✅" if p < 200 else "⚠️" if p < 500 else "❌"


def _vocab_script_counts(tok):
    """How many distinct chars of each script appear anywhere in the vocab."""
    try:
        vocab_text = tok.decode(list(range(len(tok))), skip_special_tokens=True)
    except Exception:
        vocab_text = ""
    chars = set(vocab_text)
    return {s: sum(1 for c in chars if _in_script(c, s)) for s in SCRIPT_RANGES}


def eval_language(model, tok, lang, meta, block):
    script = meta["script"]
    r = {"language": lang, "name": meta["name"], "script": script}

    # --- tokenizer: fertility + round-trip ---
    ferts, rt_fail, chars = [], 0, set()
    for text in block["samples"]:
        ids = tok.encode(text, add_special_tokens=False)
        dec = tok.decode(ids, skip_special_tokens=True).strip()
        if dec.split() != text.strip().split():
            rt_fail += 1
        w = text.split()
        if w:
            ferts.append(len(ids) / len(w))
        chars.update(c for c in text if c.strip())
    n = max(1, len(block["samples"]))
    r["avg_fertility"] = round(float(np.mean(ferts)), 3) if ferts else 0.0
    r["max_fertility"] = round(float(np.max(ferts)), 3) if ferts else 0.0
    r["roundtrip_rate"] = round(1 - rt_fail / n, 4)
    r["sample_unique_chars"] = len(chars)

    # --- generation: correct-script + repetition ---
    gen_ok, gen_rep, samples_out = [], [], []
    for prompt in block.get("generation_prompts", []):
        ids = tok.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.7,
                                 top_p=0.9, repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ok = any(_in_script(c, script) for c in gen if c.strip())
        wl = gen.split()
        rep = len(wl) > 4 and len({" ".join(wl[i:i+3]) for i in range(len(wl)-2)}) != len(wl) - 2
        gen_ok.append(ok); gen_rep.append(rep)
        samples_out.append({"prompt": prompt, "output": gen[:100], "correct_script": ok})
    r["gen_correct_script_rate"] = round(float(np.mean(gen_ok)), 3) if gen_ok else None
    r["gen_repetitive_rate"] = round(float(np.mean(gen_rep)), 3) if gen_rep else None
    r["gen_samples"] = samples_out

    # --- perplexity ---
    losses = []
    for text in block.get("perplexity_text", []):
        ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            losses.append(model(ids, labels=ids).loss.item())
    if losses:
        r["avg_loss"] = round(float(np.mean(losses)), 3)
        r["perplexity"] = round(float(np.exp(np.mean(losses))), 1)
    else:
        r["avg_loss"] = r["perplexity"] = None

    r["fertility_grade"] = _grade_fertility(lang, r["avg_fertility"])
    r["perplexity_grade"] = _grade_ppl(lang, r["perplexity"]) if r["perplexity"] else "—"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FrontiersMind/Lumma-0.6B-Base")
    ap.add_argument("--data", default="configs/base_check_indic.json")
    ap.add_argument("--out-dir", default="reports/base_model_check")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM
    from kupe_asr_en.modeling.asr_model import load_tokenizer

    cfg = json.load(open(args.data))
    langs = cfg["languages"]
    data = cfg["data"]
    os.makedirs(args.out_dir, exist_ok=True)

    print("#" * 64)
    print(f"# MULTILINGUAL BASE CHECK: {args.model}")
    print(f"# transformers {transformers.__version__} | torch {torch.__version__}")
    print("#" * 64)

    tok = load_tokenizer(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32).eval()

    # weight health once
    nan = inf = 0
    for _, p in model.named_parameters():
        nan += int(torch.isnan(p).any()); inf += int(torch.isinf(p).any())
    vocab_scripts = _vocab_script_counts(tok)

    overall = {"model": args.model, "transformers": transformers.__version__,
               "weights_healthy": (nan == 0 and inf == 0), "vocab_size": len(tok),
               "vocab_script_coverage": vocab_scripts, "per_language": {}}

    print(f"\nweights: NaN={nan} Inf={inf} | vocab={len(tok)}")
    print("vocab script coverage:", {k: v for k, v in vocab_scripts.items() if v})
    print("\n%-4s %-10s %6s %6s %9s %9s %10s" %
          ("code", "name", "fert", "rt%", "gen-scr%", "ppl", "grade"))
    print("-" * 64)

    for code, meta in langs.items():
        if code not in data:
            continue
        r = eval_language(model, tok, code, meta, data[code])
        overall["per_language"][code] = r
        json.dump(r, open(os.path.join(args.out_dir, f"lumma_{code}.json"), "w"),
                  indent=2, ensure_ascii=False)
        gs = f"{r['gen_correct_script_rate']*100:.0f}" if r["gen_correct_script_rate"] is not None else "-"
        pp = f"{r['perplexity']:.0f}" if r["perplexity"] else "-"
        print("%-4s %-10s %6.2f %5.0f%% %8s%% %9s   %s/%s" %
              (code, meta["name"][:10], r["avg_fertility"], r["roundtrip_rate"]*100,
               gs, pp, r["fertility_grade"], r["perplexity_grade"]))

    # ---- overall summary + verdict per language ----
    for code, r in overall["per_language"].items():
        usable = (r["roundtrip_rate"] > 0.95 and r["avg_fertility"] < 3.0)
        r["indic_usable_tokenizer"] = bool(usable)
    overall["summary"] = {
        "languages_tested": list(overall["per_language"].keys()),
        "tokenizer_usable": [c for c, r in overall["per_language"].items()
                             if r["indic_usable_tokenizer"]],
        "high_perplexity": [c for c, r in overall["per_language"].items()
                            if r["perplexity"] and r["perplexity"] > 500],
    }
    json.dump(overall, open(os.path.join(args.out_dir, "lumma_overall.json"), "w"),
              indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print("OVERALL")
    print("=" * 64)
    print("tokenizer usable (rt>95%, fert<3):", overall["summary"]["tokenizer_usable"])
    print("very high perplexity (>500, model barely knows it):",
          overall["summary"]["high_perplexity"] or "none")
    print(f"\nreports written to: {args.out_dir}/  (lumma_<lang>.json + lumma_overall.json)")


if __name__ == "__main__":
    main()
