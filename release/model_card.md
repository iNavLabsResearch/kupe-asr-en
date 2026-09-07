---
license: other
language:
  - en
tags:
  - automatic-speech-recognition
  - asr
  - speech
  - streaming
  - mimi
  - kupe
  - kupe-slm
pipeline_tag: automatic-speech-recognition
base_model: FrontiersMind/Lumma-0.6B-Base
metrics:
  - wer
  - cer
model-index:
  - name: Kupe-SLM-EN-600M
    results:
      - task:
          type: automatic-speech-recognition
        metrics:
          - type: wer
            value: 10.3
            name: Test WER (%)
          - type: cer
            value: 4.9
            name: Test CER (%)
---

# 🗣️ Kupe‑SLM‑EN 600M — *streaming English ASR that thinks in tokens*

**Kupe‑SLM‑EN** is a 600M‑parameter **Speech Language Model** for **real‑time English
speech‑to‑text**. It doesn't use a classic acoustic encoder — it **listens in discrete
audio tokens** (Kyutai **Mimi** codec, 12.5 Hz) and lets a language model, **Kupe‑LM**
(our fine‑tune of `FrontiersMind/Lumma‑0.6B‑Base`), simply *read the audio and write the
text*. That makes it tiny, fast on modest hardware, and naturally streaming with built‑in
end‑of‑turn signals.

| | |
|---|---|
| **Test WER** | **10.3 %** |
| **Test CER** | **4.9 %** |
| **Silence → empty** | **100 %** (no "the the the" hallucination) |
| **Backbone** | Kupe‑LM (Lumma‑0.6B, 600M) |
| **Audio frontend** | Mimi c0–c7, per‑frame summed → 512‑d `inputs_embeds` |
| **Input rate** | 12.5 tokens/s (≈63 tokens for 5 s of audio) |
| **Training** | full fine‑tune (not LoRA), 1× H100, ~3.5 h |

---

## ✨ Why it's different

- **Audio as embeddings, not new vocab.** Lumma uses *factorized, tied* embeddings, so
  instead of bolting 16k audio tokens onto the vocabulary, we feed audio as
  **`inputs_embeds`** (512‑dim — Lumma's factorized width). The 131k text vocabulary is
  untouched, which keeps the LM's language ability intact.
- **Per‑frame c0–c7.** All 8 Mimi codebooks are summed into **one vector per 80 ms frame**,
  so a 5 s clip is only ~63 input tokens → very short sequences → fast CPU/GPU inference.
- **Streaming turn flags for free.** `PRE_HIT_LLM` (prefetch your LLM) and `END_OF_SPEECH`
  (commit final) come straight from the model's own `P(<eos>)` — no second model.
- **Anti‑hallucination trained in.** 5 % of training clips are pure silence / ambient noise
  labeled `""`, so silence reliably decodes to **nothing** (measured 100 % on held‑out
  silence).

---

## 📊 Results

Held‑out **test** split (n = 500), greedy decoding:

| Metric | Score |
|---|---|
| **WER** | **10.3 %** |
| **CER** | **4.9 %** |
| Silence → empty | 100 % |

Live WER during training fell steadily: 41 % → 18 % → 12.8 % → **10.3 %** (final).
For context: production English ASR on far larger data sits at 4–6 %; **10 % from a 600M
model on 2.7k h is a strong Phase‑1 result** and usable for voice agents.

---

## 🎓 Training data (2,707 h English)

| Source | Hours | Clips |
|---|---|---|
| LibriSpeech (clean+other) | 599 | 172,657 |
| People's Speech (CC‑BY) | 2,100 | 543,069 |
| Svarah (Indian‑accented EN) | 8.6 | 5,227 |
| **Silence / ambient (labeled "")** | 35 | 36,315 |
| **Total** | **~2,742** | **793,583** |

All audio resampled to 24 kHz mono, stored as FLAC, then Mimi‑encoded (8 codebooks).
Encoded corpus used for training: **2,606 h / 716,968 clips** (a few % dropped on decode).
Splits: **train 2,733 h · val 5.1 h · test 5.0 h** (sprinkled across the whole stream).

---

## 🏋️ Training setup (1× H100 80 GB)

| Hyper‑parameter | Value |
|---|---|
| Objective | full fine‑tune (all params), causal LM cross‑entropy |
| Micro‑batch | 32 · grad‑accum 4 → **effective batch 128** |
| LR | 2e‑4, cosine decay, 3 % warmup |
| Weight decay / grad‑clip | 0.01 / 1.0 |
| Epochs / steps | 2 / 11,158 |
| Precision | bf16, TF32 matmuls, fused AdamW |
| Regularization | SpecAugment (time‑mask) on the audio embeddings |
| Wall‑clock | ~3.5 h on one H100 |
| Final eval_loss | 0.26 |

Everything is driven by one config; runs are resumable from Hub checkpoints.

---

## 🔤 Tokenizer efficiency (fertility)

Kupe‑LM inherits Lumma's **multilingual tokenizer** (131,072 vocab), which is unusually
efficient — low **fertility** (tokens per word) means shorter targets, faster decoding,
lower CER. Measured on natural‑language sentences:

| Language | Fertility (tok/word) | Round‑trip |
|---|---|---|
| **English** | **1.29** | 100 % |
| Hindi | 1.63 | 100 % |
| Gujarati | 1.60 | 100 % |
| Punjabi | 1.47 | 100 % |
| Bengali | 1.77 | 100 % |
| Marathi | 1.93 | 100 % |
| Telugu | 2.14 | 100 % |
| Kannada | 2.23 | 100 % |
| Tamil | 2.29 | 100 % |
| Odia | 2.33 | 100 % |
| Malayalam | 3.12 | 100 % |

EN at **1.29 tok/word** is excellent; 100 % round‑trip everywhere means no script is
corrupted by the tokenizer.

---

## 🌏 Multilingual outlook (this release is EN‑only)

This checkpoint is **English only**, but the backbone was pre‑trained on English + 11 Indic
languages, and a base‑model probe found it already **generates coherent Hindi & Gujarati**
and has **100 % tokenizer round‑trip** for all 11 scripts, with **no language showing
"unknown" perplexity**. So the path to Hindi/Gujarati/… is **more data, same pipeline** —
the model side is ready. (Indic WER expectations, per our plan: HI ~7–10 %, GU ~10–14 %
before synthetic data; better after.)

---

## 🧩 Architecture

```
audio 24kHz ─► Mimi encoder ─► codes c0..c7 @ 12.5Hz
                                   │  per‑frame sum of 8 learned codebook embeddings
                                   ▼
         [bos][audio_bos] a0 a1 … aT [audio_eos] w0 w1 … <eos>
                                   │  fed as 512‑d inputs_embeds
                                   ▼
                    Kupe‑LM (Lumma‑0.6B) ──► English text
```

- **Backbone:** `kupe-lm/` in this repo (fine‑tuned Lumma‑0.6B: 30 layers, hidden 1440,
  GQA, factorized tied embeddings rank 512, vocab 131,072).
- **Audio frontend:** `frontend.pt` — 8 codebook embedding tables + projector + 2 markers.
- **Config:** `asr_config.json` — frontend mode (`per_frame_sum`), codebooks (8), embed dim.

---

## 🚀 Inference

Requires **`transformers==5.4.0`** (Lumma/Kupe‑LM custom architecture) and the
[`kupe-asr-en`](https://github.com/iNavLabsResearch/kupe-asr-en) code.

**File / batch:**
```python
from kupe_asr_en.modeling.asr_model import LummaASR
model, tok = LummaASR.load("path/to/this/repo", device="cuda")   # downloads via snapshot too
# encode audio with kyutai/mimi -> codes[8,T], then model.build_prefix(...) + generate
```

**Streaming from a wav (turn flags):**
```python
from kupe_asr_en.stream import transcribe_file, END_OF_SPEECH
for ev in transcribe_file("model_dir", "audio.wav"):
    print(ev.type, ev.text)     # PARTIAL / PRE_HIT_LLM / END_OF_SPEECH
```

**Live mic:**
```bash
python scripts/06_mic_stream.py --model-dir <this repo>
```

Streaming emits **PARTIAL** (live), **PRE_HIT_LLM** (start your LLM early), and
**END_OF_SPEECH** (commit) — all from the model's own end‑of‑sequence probability, plus a
VAD silence fallback.

---

## 📦 Repos

- **Model (this):** `anuj-inavlabs/kupe-asr-en`
- **Data:** `anuj-inavlabs/kupe-asr-en-data` (raw + Mimi configs, ledgers)
- **Runs:** `anuj-inavlabs/kupe-asr-en-runs` (weights + eval reports per run)
- **Code:** https://github.com/iNavLabsResearch/kupe-asr-en

## ⚖️ License & credits

Weights derive from `FrontiersMind/Lumma‑0.6B‑Base` (respect its license) and
`kyutai/mimi`. Data: LibriSpeech, MLCommons People's Speech (CC‑BY), AI4Bharat Svarah.
Released as **`other`**. Built by **iNavLabs / Kupe**.
