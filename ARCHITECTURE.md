# kupe-asr-en — Architecture & Decisions (Phase 1)

This document records **why** the code is shaped the way it is, and answers the
hard questions directly, with the evidence from a real load test of Lumma (not
theory). Phase 1 is **English only**.

---

## 1. The pipeline in one picture

```
LibriSpeech / People's Speech / Svarah      (HF datasets, streamed)
        │  fetch: resample 24kHz mono, FLAC, dedup, split
        ▼
  data repo  ── raw config  (audio bytes + text)          ~24 bunch parquet
        │  encode: kyutai/mimi, c0..c7 @ 12.5 Hz
        ▼
  data repo  ── mimi config (codes[8][T] + text)          ~24 bunch parquet   ← training reads this
        │  train: AudioFrontend → inputs_embeds → Lumma-0.6B → English text
        ▼
  runs repo  ── weights + frontend + eval report + ledgers
        │
        ▼
  streaming mic inference (PARTIAL / PRE_HIT_LLM / END_OF_SPEECH)
```

Everything is driven by one config, tracked in JSON ledgers, resumable, and
Hub-rate-limit safe. Nothing trains on local-only data.

---

## 2. Why Lumma-0.6B-Base (and not SmolLM-350M / Gemma-3 / Qwen)

Requirement 20 fixes the base model to Lumma. It is also the **right** choice for
a *voice-agent* ASR head, and here is the honest reasoning, plus what we verified.

**What Lumma is** (from its model card + `config.json`, verified 2026-09):
`model_type: Lumma`, `LummaForCausalLM`, 600M params, hidden 1440, 30 layers, GQA
(16 q / 8 kv heads), **vocab 131,072**, **factorized tied embeddings (rank 512)**,
12,288 context, trained on **1T tokens of English + 11 Indic languages**, with a
"multilingual tokenizer optimized for Indic languages" (low fertility).

| Candidate | For an English→(later Indic) voice ASR head | Verdict |
|---|---|---|
| **Lumma-0.6B** | Tokenizer is **already multilingual/Indic-efficient** → fewer tokens per Indic word later → shorter targets, faster decode, better CER. 600M gives real capacity for spontaneous speech. Modern arch (GQA, QK-norm, SwiGLU, shared-KV) → cheap KV cache for streaming. | ✅ Best fit for the roadmap |
| SmolLM-350M | Smaller/faster, but **English-centric BPE**: Indic fertility is terrible, so the exact v2 plan (add HI/GU) would need a tokenizer swap = retrain from scratch. Also less capacity for noisy/spontaneous audio. | ❌ Dead-ends the roadmap |
| Gemma-3-270m | Proven (our previous repo used it) and easy, but **256k English-leaning vocab** and only 270M. Fine for a demo, weaker Indic tokenization and capacity. | ⚠️ OK stopgap, not the target |
| Qwen3 0.6–0.8B | Strong general model, decent multilingual, but its tokenizer is **not** Indic-optimized the way Lumma's is; Lumma was purpose-built for exactly this language set. | ⚠️ Viable, but Lumma is the better-matched base |

**Will Lumma give correct results?** Phase 1 is precisely the experiment that
answers this with a WER number. What we have *already proven* by loading it:

- ✅ Loads and runs on **`transformers==5.4.0`** (native `LummaTokenizer` +
  `LummaForCausalLM`). It **cannot** load on 4.x — the modeling/tokenizer files
  import 5.x-only APIs. This pin is load-bearing and is in `requirements.txt`.
- ✅ `forward(inputs_embeds=…)` and `generate(inputs_embeds=…)` both work → the
  "embed audio, decode text" design is valid on this model.
- ✅ Its input embedding is **512-dim** (the factorized space), not hidden 1440.
  The frontend emits 512-dim vectors; we read the width from the model at runtime
  so a checkpoint change can never silently break it.
- ✅ Mimi `encode(wav, num_quantizers=8)` returns `(1, 8, T)` codes → c0–c7 available.

So the base is confirmed *mechanically* sound; Phase-1 training tells us if it's
*quality* sound. That is a $10–20 experiment (see §9), not a leap of faith.

---

## 3. How audio enters the model (and why we DON'T resize the vocab)

Lumma has **factorized, tied** embeddings (rank 512). Growing that matrix to add
"audio tokens" (the approach our Gemma repo used) is fragile on a factorized tied
embedding and risks corrupting the LM. Phase 1 is English only, so **the output
vocabulary needs nothing new** — English text already lives in Lumma's 131k vocab.

Therefore audio is fed as **`inputs_embeds`** (the SALMONN / Qwen-Audio / LauraGPT
pattern), and the LM head is untouched:

```
[bos] [audio_bos]  a_0 … a_{L-1}  [audio_eos]  w_0 … w_{m-1} <eos>
|------------- prefix, label = -100 -----------|---- supervised text ----|
```

`a_*` are audio embeddings from the frontend; `audio_bos/eos` are two learned
input-only markers. Each sample is assembled **contiguously** and the batch is
right-padded, so RoPE positions stay correct (no padding in the middle). The only
new parameters are the codebook embeddings + a small projector + 2 markers.

---

## 4. The Phase-1 A/B: per-frame-sum (c0–c7) vs flatten (c0–c3)

This is the experiment the material asked for, isolated behind
`kupe_asr_en/modeling/frontend.py` and switchable with one flag.

| | `per_frame_sum` (default) | `flatten` |
|---|---|---|
| Vector / frame | 1 (Σ of c0–c7 embeddings) | K (one per codebook, interleaved) |
| Codebooks used | 8 (full acoustic detail) | usually 4 (c0–c3) |
| Input rate | 12.5 tok/s | 50 tok/s (K=4) |
| Seq len (5 s) | ~63 | ~250 |
| Attention cost | 1× | ~16× |
| CPU streaming | comfortable | marginal |
| Published for **ASR** | not specifically | ✅ (SpeechGPT/TWIST flatten) |

**Will the single summed vector actually work, honestly?**

- Summed-codebook embeddings are **proven for the harder task of audio
  *generation*** (VQ-VAE-2, SoundStorm, MusicGen). ASR needs *less* information
  than generation (you only need the text, not to reconstruct audio), so if the
  sum carries enough to regenerate audio, it carries enough to transcribe it.
  That is a strong prior, **not** a published ASR guarantee.
- The sum is **learned** (per-codebook embedding tables + a projector), so the
  model can weight codebooks; it is strictly more expressive than a fixed sum.
- **We do not bet the project on it.** Phase 1 trains *both* and compares WER.
  The decision rule (from the material, encoded in the config gate):
  - per_frame WER ≤ flatten WER + 1% → **use per_frame** (much faster inference)
  - within +1–3% → still likely worth per_frame for a CPU voice agent
  - > +3% → **use flatten c0-c3** (accept slower inference)

So the risk is *measured and hedged*, not assumed away. Cost of the hedge: one
extra short training run (§9).

---

## 5. Is 3× augmentation safe / will it overfit?

Short answer: **3× speed+noise augmentation is the industry standard and does not
cause overfitting** — it is the *opposite* of overfitting.

- Augmented copies have **different inputs** (a speed-perturbed clip → different
  Mimi codes) but the **same text**. The model learns that one sentence can sound
  many ways = *robustness/regularization*, exactly like dropout. Overfitting is
  memorizing the *same* input→output pair repeatedly; augmentation prevents that.
- It is the **default** in Kaldi (speed 0.9/1.0/1.1 = 3×, since 2016), ESPnet,
  NeMo, WeNet — thousands of production systems. Papers: Ko et al. 2015 (speed
  perturb), Park et al. 2019 (SpecAugment) show 10–20% *relative* WER reduction.
- It only hurts at extreme ratios (10×+ on a tiny vocabulary) or when the
  augmentation destroys intelligibility (speed 0.5×, SNR −10 dB). 3× with SNR
  5–20 dB and ±10% speed is safe.

**Phase-1 scope note:** Phase 1 keeps augmentation *light on purpose* — it uses
on-the-fly **SpecAugment** (time masking in the frontend; free, no data blow-up)
and leaves heavy 3× audio augmentation to the scale-up phase, so the sanity signal
isn't confounded. The config exposes `train.spec_time_mask`. When you do add 3×,
it is safe — the worry is unfounded.

### Do we store augmented+noised data and then encode it to `mimi`? (asked directly)

**Phase 1: no.** We do **not** upload augmented/noised variants and we do **not**
encode them. Reasons: (1) it triples storage + encode time; (2) it confounds the
one thing Phase 1 measures ("can Lumma learn English from clean Mimi codes?").
Phase 1's only augmentation is on-the-fly SpecAugment.

**Scale-up: yes, and here is exactly where it plugs in.** Because training reads
**Mimi codes, not audio**, you cannot acoustically augment at train time — the
augmentation must happen in the **waveform domain at fetch time, before encode**:

```
fetch:  clip ─┬─ original            ─┐
              ├─ + noise (SNR 5-20dB) ─┼─ store all as `raw`  (3× rows)
              └─ + speed (0.9/1.1)    ─┘
        + 8% pure-silence/ambient clips labeled ""   (anti-hallucination)
encode: Mimi-encode EVERY variant  ->  `mimi`  (3× rows; each variant = different codes)
train:  unchanged — it just sees more, acoustically-varied rows
```

**Exception — silence training IS live in Phase 1.** Teaching the model that
silence → *nothing* (not "the the the" / "thank you") is too important to defer, so
fetch injects `data.silence_frac` (default 5%) of pure-silence + low-level-ambient
clips **labeled `""`**. These flow through encode → `mimi` like any clip; the empty
target teaches an immediate `<eos>`. A few are held out (`silence_val_cap`) so eval
reports a **silence→empty rate** (target > 95%). This complements the inference-time
silence gate + repetition penalty — belt and suspenders against hallucination.

The rest of the distribution (from the plan) is in `configs/phase1.yaml → augment` (disabled):
per clip = original + 1 noise + 1 speed (=3×), plus telephone bandpass 15%, reverb
15%, and 8% silence/ambient clips with empty transcripts. Noise/reverb need a corpus
(DNS/MUSAN, RIR) — that fetch-time hook is documented but intentionally not built in
Phase 1. **It will not overfit** (see the argument above: different inputs, same
text = regularization; the only new params here, the audio frontend, benefit most).

---

## 6. Data: sources, hours, and target enforcement

Target ≈ **3,000 h English**; hard floor `min_hours = 2500`. Sources are a
**registry/factory** (`data/sources.py`) — add one `Source(...)` line for a new
dataset; exotic schemas override `extract()`.

| Source | cap h | why |
|---|---|---|
| LibriSpeech (ungated) | 960 | clean read-speech core, guaranteed |
| People's Speech (ungated) | 2100 | domain/acoustic diversity |
| Svarah (gated:auto) | 12 | Indian-accented English (prevents a common failure) |
| GigaSpeech (gated) | 0 | optional; set cap>0 after accepting the license |

**Enforcement (rule 6):** `05_check_data.py` and `train.py` both refuse to
proceed below `min_hours`. The data ledger logs exactly how many hours were kept
per source and how far below target we are — no silent under-training.

**Splits** are fixed at fetch time (a `split` column), sprinkled across the whole
stream so val/test aren't all one source: `val_hours=5`, `test_hours=5`, rest train.

---

## 7. Storage: FLAC, ~20–25 shards, and the honest disk trade-off

- `raw` is stored **FLAC (lossless)** by default so the encoder sees the audio
  exactly as the source did. Local tiny shards (1000 rows) are packed into
  **~`raw_target_shards` (24)** big `bunch_*.parquet` on the Hub, and the same for
  `mimi` (`mimi_target_shards`).
- **Honest trade-off:** 3,000 h of FLAC is ~350–400 GB. Split into 24 files that
  is ~15 GB/bunch — the compactor **warns** past `bunch_soft_gb`. If disk is tight,
  set `data.audio_format: opus` (≈10× smaller, ~35 GB total, negligible ASR-quality
  cost). The **`mimi`** config is tiny either way (~2–4 GB for 3,000 h), so 24
  mimi shards is comfortable. Fetch keeps only ~one bunch on disk at a time by
  uploading in waves.

---

## 8. Ledgers, resume, and rate limits

- **Four JSON ledgers** (rule 7): `data.json`, `mimi.json` (data repo),
  `runs.json`, `evals.json` (runs repo). Atomic writes, mirrored to the Hub, merged
  on resume → no box ever redoes finished work.
- **Resume everywhere:** fetch resumes per parquet file + fingerprint dedup;
  encode resumes per raw bunch; train resumes from local or Hub checkpoint.
- **Rate limits (rule 9):** 128 commits/hr respected by a `CommitPacer`
  (min interval) + bunching (~25 commits total per dataset) + exponential backoff
  on 429/5xx. Downloads only fetch the handful of bunch files (well under
  1000 req/5 min).

---

## 9. Phase-1 compute & cost

Data ≈ 3,000 h, c0–c7, 2 epochs. Mimi input is 12.5 tok/s (per_frame) so a
*single* epoch is only ~0.14 B audio tokens.

| Stage | H100×1 | RTX 4090 | notes |
|---|---|---|---|
| Encode (once) | ~1 h | ~3–5 h | dominated by download + Mimi |
| Train per_frame_sum (2 ep) | **~1.5 h** | ~4–5 h | short seqs |
| Train flatten c0-c3 (2 ep) | ~4–5 h | ~12–15 h | 4× longer seqs, lower batch |
| **Both (the hedge)** | **~6–7 h GPU** | ~16–20 h | ≈ **$10–20** on rented H100 |

Fail-fast and cheap: if per_frame is broken you know in ~1.5 h.

---

## 10. Expected WER / CER for THIS Phase-1 run

English only, ~3,000 h (LibriSpeech + People's Speech + Svarah), Lumma-0.6B,
`per_frame_sum` c0–c7, 2 epochs, no heavy augmentation. CER ≈ 0.4–0.5 × WER for
English. These are *expectations to check the run against*, not guarantees:

| Test set | WER (expected) | CER (expected) | Note |
|---|---|---|---|
| LibriSpeech **test-clean**-like | **8–12%** | 3–6% | clean read speech — the main sanity number |
| LibriSpeech **test-other**-like | 15–22% | 8–12% | noisier/accented; limited in 3k h |
| People's Speech (in-domain) | 12–20% | 7–11% | spontaneous, varied mics |
| Svarah (Indian-accent EN) | 18–28% | 10–16% | tiny (12 h); expect weak, it's a diversity probe |

Reference points: production English ASR on full data hits 4–6% WER; our thin 3k h
+ 600M model at 8–12% is exactly the "the approach works, now scale data" signal.

- **> 60% WER** → pipeline broken (frontend wiring, codes, tokenizer) — debug first.
- **15–60%** → learning but thin — more data/epochs, or try `flatten`.
- **≤ 15%** → PASS. Ship the approach to scale-up.

The `flatten` c0–c3 A/B variant should land within ~±1–3% WER of `per_frame_sum`;
if it beats per_frame by >3%, switch (accept the slower inference).

## 11. Phase-1 decision gate & risks

**Gate** (in `eval.wer_pass` / `eval.wer_broken`, logged as a verdict):
- test WER **≤ 15%** → PASS: pipeline + Lumma work → scale up (more data, Indic).
- 15–60% → PARTIAL: learning but thin → more data/epochs or try the other frontend.
- **≥ 60%** → BROKEN: debug wiring (frontend, codes, tokenizer) before anything else.

Realistic Phase-1 expectation on ~3,000 h, 600M, 2 epochs: **WER ~8–12%** on clean
test — enough to prove the approach.

| Risk | Likelihood | Mitigation |
|---|---|---|
| per_frame underperforms flatten | ~15% | A/B in Phase 1; flatten fallback |
| Lumma base is a poor ASR head | low (mechanically verified) | Phase-1 WER catches it; Gemma-3 fallback documented |
| transformers version drift | medium | pinned `==5.4.0`; explicit assert + warning |
| Big FLAC bunches vs disk | medium | `audio_format: opus`; wave-upload frees disk |
| Indic data scarcity (later) | high (future) | out of Phase-1 scope; TTS synthesis in a later phase |
| Silence hallucination | low | **silence-data training (`data.silence_frac`, live)** + inference silence gate + repetition penalty; eval reports silence→empty rate |

---

## 12. What Phase 1 deliberately excludes

No languages other than English. No language tokens / auto-LID. No heavy 3×
augmentation, no TTS synthetic data, no quantization/ONNX, no Indic. Those are
later phases — Phase 1 is the sanity proof, kept small so its signal is clean.
