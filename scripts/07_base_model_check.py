#!/usr/bin/env python
"""Base-model FIT CHECK — run BEFORE spending a dollar on training.

Answers: "is Lumma-0.6B-Base actually usable as an English ASR head with our
inputs_embeds design?" — in ~5-10 min on CPU. Writes a JSON report card + an
IN/OUT verdict.

    python scripts/07_base_model_check.py                       # Lumma, English
    python scripts/07_base_model_check.py --model google/gemma-3-1b-pt   # compare a baseline
    python scripts/07_base_model_check.py --data configs/base_check_en.json --out lumma_check.json

IMPORTANT — this checks OUR architecture, which feeds audio as `inputs_embeds`
and does NOT resize the vocabulary (Lumma's embeddings are factorized+tied, so a
resize is both unnecessary for English and risky). The two audio tests therefore
verify (a) the model accepts external `inputs_embeds`, and (b) the AudioFrontend +
Mimi path produces a finite, trainable loss — NOT that you can add vocab tokens.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np
import torch


# ------------------------------------------------------------------ helpers
def _load_json(p):
    with open(p) as f:
        return json.load(f)


def _grade(ok, warn=False):
    return "✅" if ok else ("⚠️" if warn else "❌")


# ================================================================ TEST 1
def test_weight_health(model):
    print("\n" + "=" * 60 + "\nTEST 1: WEIGHT HEALTH\n" + "=" * 60)
    r, nan, inf, total, norms = {}, 0, 0, 0, []
    for name, p in model.named_parameters():
        total += p.numel()
        if torch.isnan(p).any():
            nan += 1; print(f"  ❌ NaN in {name}")
        if torch.isinf(p).any():
            inf += 1; print(f"  ❌ Inf in {name}")
        norms.append(p.float().norm().item())
    norms = np.array(norms)
    collapsed = int((norms < 0.01).sum())
    r.update(total_params=total, nan_layers=nan, inf_layers=inf, collapsed_layers=collapsed,
             weight_norm_mean=float(norms.mean()), weight_norm_std=float(norms.std()),
             weights_healthy=(nan == 0 and inf == 0 and collapsed == 0))
    print(f"  params={total:,} NaN={nan} Inf={inf} collapsed={collapsed} "
          f"norm mean={norms.mean():.2f}±{norms.std():.2f}  {_grade(r['weights_healthy'])}")
    return r


# ================================================================ TEST 2
def test_tokenizer(tok, data):
    print("\n" + "=" * 60 + "\nTEST 2: TOKENIZER QUALITY (English)\n" + "=" * 60)
    r = {"vocab_size": len(tok)}
    samples = data.get("en_samples", [])
    ferts, rt_fail, chars = [], 0, set()
    for s in samples:
        text = s["text"]
        ids = tok.encode(text, add_special_tokens=False)
        dec = tok.decode(ids, skip_special_tokens=True).strip()
        # tolerate case/space normalisation differences
        if dec.lower().split() != text.lower().split():
            rt_fail += 1
            if rt_fail <= 3:
                print(f"  ⚠️ round-trip: '{text[:30]}' -> '{dec[:30]}'")
        w = text.split()
        if w:
            ferts.append(len(ids) / len(w))
        chars.update(c for c in text if c.strip())
    avg_f = float(np.mean(ferts)) if ferts else 0
    rt_rate = 1 - rt_fail / max(1, len(samples))
    r.update(en_avg_fertility=round(avg_f, 3), en_roundtrip_rate=round(rt_rate, 4),
             en_unique_chars=len(chars))
    print(f"  EN fertility={avg_f:.2f} tok/word  roundtrip={rt_rate:.1%}  chars={len(chars)}  "
          f"{_grade(avg_f < 1.6 and rt_rate > 0.99, warn=avg_f < 2.0)}")
    return r


# ================================================================ TEST 3
def test_generation(model, tok, data):
    print("\n" + "=" * 60 + "\nTEST 3: TEXT GENERATION (does it speak English?)\n" + "=" * 60)
    model.eval()
    res = []
    for pd in data.get("en_generation_prompts", []):
        ids = tok.encode(pd["prompt"], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.7,
                                 top_p=0.9, repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        latin = any(0x41 <= ord(c) <= 0x7A for c in gen)
        w = gen.split()
        rep = len(w) > 4 and len({" ".join(w[i:i+3]) for i in range(len(w)-2)}) != len(w) - 2
        res.append({"latin": latin, "rep": rep, "len": len(gen), "out": gen[:80]})
        print(f"  '{pd['prompt'][:32]}' -> {gen[:60]!r}")
    latin_rate = float(np.mean([x["latin"] for x in res])) if res else 0
    rep_rate = float(np.mean([x["rep"] for x in res])) if res else 0
    print(f"  correct_script={latin_rate:.0%}  repetitive={rep_rate:.0%}  "
          f"{_grade(latin_rate > 0.8 and rep_rate < 0.3, warn=latin_rate > 0.5)}")
    return {"en_correct_script_rate": round(latin_rate, 3), "en_repetitive_rate": round(rep_rate, 3)}


# ================================================================ TEST 4
def test_perplexity(model, tok, data):
    print("\n" + "=" * 60 + "\nTEST 4: PERPLEXITY (does it already know English?)\n" + "=" * 60)
    model.eval()
    losses = []
    for text in data.get("en_perplexity_text", []):
        ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            losses.append(model(ids, labels=ids).loss.item())
    avg = float(np.mean(losses)) if losses else float("nan")
    ppl = float(np.exp(avg)) if losses else float("nan")
    print(f"  EN loss={avg:.2f} perplexity={ppl:.1f}  "
          f"{_grade(ppl < 50, warn=ppl < 100)}  (lower = knows English better)")
    return {"en_avg_loss": round(avg, 3), "en_perplexity": round(ppl, 1)}


# ================================================================ TEST 5 (architecture-specific)
def test_inputs_embeds(model, tok):
    print("\n" + "=" * 60 + "\nTEST 5: inputs_embeds ACCEPTANCE (our audio path)\n" + "=" * 60)
    r = {}
    emb = model.get_input_embeddings()
    E = emb.weight.shape[1]
    r["embed_dim"] = int(E)
    print(f"  input embedding dim = {E}  (audio frontend must emit {E}-dim vectors)")
    ids = tok.encode("hello world this is a test", add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        text_e = emb(ids)
        audio_e = torch.randn(1, 24, E) * float(text_e.float().std())   # fake 24-frame audio prefix
        full = torch.cat([audio_e.to(text_e.dtype), text_e], dim=1)
        out = model(inputs_embeds=full)
        ok_fwd = torch.isfinite(out.logits).all().item() and out.logits.shape[-1] == emb.weight.shape[0]
        gen = model.generate(inputs_embeds=audio_e.to(text_e.dtype),
                             attention_mask=torch.ones(1, 24, dtype=torch.long),
                             max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id)
        ok_gen = gen.shape[1] > 0
    r.update(forward_ok=bool(ok_fwd), generate_ok=bool(ok_gen),
             inputs_embeds_ok=bool(ok_fwd and ok_gen))
    print(f"  forward(inputs_embeds) finite logits: {ok_fwd}  {_grade(ok_fwd)}")
    print(f"  generate(inputs_embeds) works: {ok_gen}  {_grade(ok_gen)}")
    return r


# ================================================================ TEST 6 (architecture-specific)
def test_frontend_fit(model, tok, mimi_id="kyutai/mimi", num_codebooks=8):
    print("\n" + "=" * 60 + "\nTEST 6: AudioFrontend + Mimi FIT CHECK\n" + "=" * 60)
    r = {}
    from kupe_asr_en.modeling.asr_model import LummaASR
    from kupe_asr_en.modeling.frontend import AudioFrontend

    # (a) Mimi encodes 8 codebooks?
    try:
        from transformers import MimiModel
        mimi = MimiModel.from_pretrained(mimi_id).eval()
        wav = torch.randn(1, 1, 24000 * 2)                     # 2s synthetic audio
        with torch.no_grad():
            codes = mimi.encode(wav, num_quantizers=num_codebooks).audio_codes   # [1,K,T]
        r["mimi_codebooks"] = int(codes.shape[1])
        r["mimi_ok"] = codes.shape[1] == num_codebooks
        print(f"  Mimi returns {codes.shape[1]} codebooks, {codes.shape[2]} frames  {_grade(r['mimi_ok'])}")
    except Exception as e:
        r["mimi_ok"] = False
        print(f"  ❌ Mimi encode failed: {e}")
        codes = torch.randint(0, 2048, (1, num_codebooks, 25))

    # (b) frontend per-frame embedding norms reasonable
    E = model.get_input_embeddings().weight.shape[1]
    fe = AudioFrontend(E, num_codebooks, "per_frame_sum", "mlp")
    with torch.no_grad():
        a = fe(codes)
    fn = a.float().norm(dim=-1)
    reasonable = 0.05 < float(fn.mean()) < 200
    r.update(frame_norm_mean=round(float(fn.mean()), 4), frame_emb_reasonable=bool(reasonable))
    print(f"  frame embedding norm mean={fn.mean():.3f}  {_grade(reasonable)}")

    # (c) full LummaASR forward on (audio, text) -> finite loss + grad flows to frontend
    asr = LummaASR(model, fe, tok.bos_token_id, tok.eos_token_id, 750)
    text_ids = tok.encode("this is a spoken sentence", add_special_tokens=False)
    if text_ids and text_ids[0] == tok.bos_token_id:
        text_ids = text_ids[1:]
    text_ids = torch.tensor([text_ids + [tok.eos_token_id]])
    batch = {"codes": codes, "num_frames": torch.tensor([codes.shape[2]]),
             "text_ids": text_ids, "text_lengths": torch.tensor([text_ids.shape[1]]),
             "labels": text_ids.clone()}
    asr.train()
    out = asr(**batch)
    loss = out.loss
    finite = bool(torch.isfinite(loss).item())
    loss.backward()
    gflow = sum(p.grad.abs().sum().item() for p in fe.parameters() if p.grad is not None) > 0
    r.update(asr_loss=round(float(loss.detach()), 3), asr_loss_finite=finite, frontend_grad_flows=bool(gflow))
    print(f"  LummaASR forward loss={float(loss.detach()):.3f} finite={finite}  grad→frontend={gflow}  "
          f"{_grade(finite and gflow)}")
    r["frontend_fit_ok"] = bool(finite and gflow and r.get("mimi_ok", False))
    return r


# ================================================================ RUNNER
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FrontiersMind/Lumma-0.6B-Base")
    ap.add_argument("--data", default="configs/base_check_en.json")
    ap.add_argument("--out", default="base_check_results.json")
    ap.add_argument("--codebooks", type=int, default=8)
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM
    from kupe_asr_en.modeling.asr_model import load_tokenizer

    print("#" * 60)
    print(f"# BASE MODEL FIT CHECK: {args.model}")
    print(f"# transformers {transformers.__version__} | torch {torch.__version__}")
    print("#" * 60)
    if not transformers.__version__.startswith("5."):
        print("⚠️  Lumma needs transformers==5.4.0; other bases may load on 4.x.")

    data = _load_json(args.data)
    results = {"model": args.model, "transformers": transformers.__version__}

    # load once, reuse
    try:
        tok = load_tokenizer(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.float32).eval()
    except Exception as e:
        print(f"\n❌ FATAL: model/tokenizer failed to load: {e}")
        results["load_error"] = str(e)
        json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
        return

    results["test1_weights"] = test_weight_health(model)
    results["test2_tokenizer"] = test_tokenizer(tok, data)
    results["test3_generation"] = test_generation(model, tok, data)
    results["test4_perplexity"] = test_perplexity(model, tok, data)
    results["test5_inputs_embeds"] = test_inputs_embeds(model, tok)
    results["test6_frontend_fit"] = test_frontend_fit(model, tok, num_codebooks=args.codebooks)

    # ---------- REPORT CARD ----------
    t1, t2, t3, t4, t5, t6 = (results[k] for k in
                              ["test1_weights", "test2_tokenizer", "test3_generation",
                               "test4_perplexity", "test5_inputs_embeds", "test6_frontend_fit"])
    critical = {
        "Weights healthy (no NaN/Inf/collapse)": t1["weights_healthy"],
        "EN round-trip > 99%": t2["en_roundtrip_rate"] > 0.99,
        "Accepts inputs_embeds (forward+generate)": t5["inputs_embeds_ok"],
        "Mimi returns 8 codebooks": t6.get("mimi_ok", False),
        "AudioFrontend produces trainable loss": t6.get("frontend_fit_ok", False),
    }
    warnings = {
        "EN fertility < 1.6 tok/word": t2["en_avg_fertility"] < 1.6,
        "EN perplexity < 50": t4["en_perplexity"] < 50,
        "Generates Latin script > 80%": t3["en_correct_script_rate"] > 0.8,
        "Frame embedding norm reasonable": t6.get("frame_emb_reasonable", False),
    }
    print("\n" + "=" * 60 + "\nREPORT CARD\n" + "=" * 60)
    print("CRITICAL (any ❌ = OUT):")
    for k, v in critical.items():
        print(f"  {_grade(v)} {k}")
    print("QUALITY (⚠️ = fine, just needs more training data/epochs):")
    for k, v in warnings.items():
        print(f"  {_grade(v, warn=True)} {k}")

    go = all(critical.values())
    verdict = "IN ✅ — proceed to Phase-1 training" if go else "OUT ❌ — fix/replace base before training"
    print("\n" + "#" * 60)
    print(f"# VERDICT: {verdict}")
    if not go:
        print("# failed critical:", ", ".join(k for k, v in critical.items() if not v))
    print("#" * 60)

    results["report_card"] = {"critical": critical, "quality": warnings,
                              "verdict": "IN" if go else "OUT"}
    json.dump(results, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
