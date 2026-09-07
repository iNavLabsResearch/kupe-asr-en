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
metrics:
  - wer
  - cer
model-index:
  - name: kupe-asr-en
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

<img src="https://huggingface.co/anuj-inavlabs/kupe-asr-en/resolve/main/kupe-light.png" height="44" alt="logo"/>

### Streaming English ASR — early **Phase‑1** checkpoint

A 670M streaming speech‑to‑text model: audio is turned into Mimi codec tokens and the
**Kupe‑SLM‑670M** language backbone reads them and writes English text. Small, fast, real‑time.

| Split | WER | CER |
|---|---|---|
| **Test** (held‑out, n=500) | **10.3 %** | **4.9 %** |
| Validation | 11.2 % | 5.4 % |
| Silence → empty output | 100 % | — |

> ⚠️ **Early Phase‑1 checkpoint.** This is a first sanity‑milestone model trained on ~2.6k h
> to prove the pipeline and architecture. It is **not** the final release — quality will
> improve substantially with more data, augmentation, and epochs in later phases.

---

## Overview

- **Task:** English automatic speech recognition (streaming + full‑file).
- **How it works:** speech → **Mimi** neural audio codec (8 codebooks @ 12.5 Hz) → the 8
  codebooks are summed into one embedding per 80 ms frame → fed to **Kupe‑SLM‑670M** as
  `inputs_embeds` → the model decodes English text. No separate acoustic encoder.
- **Streaming:** emits turn signals from the model's own end‑of‑sequence probability —
  `PRE_HIT_LLM` (prefetch your LLM) and `END_OF_SPEECH` (commit final).
- **Anti‑hallucination:** trained with 5 % silence/ambient clips labelled `""`, so silence
  decodes to **nothing** (measured 100 %).

## Results in context

Production English ASR on much larger corpora sits around 4–6 % WER. **10.3 % from a
Phase‑1 670M model on 2.6k h is a strong early result** and already usable for voice agents.
Live WER during training fell 41 % → 18 % → 12.8 % → **10.3 %**.

## Training data (~2.6k h English, encoded)

| Source | Hours | Clips |
|---|---|---|
| LibriSpeech (clean+other) | 599 | 172,657 |
| People's Speech (CC‑BY) | 2,100 | 543,069 |
| Svarah (Indian‑accented EN) | 8.6 | 5,227 |
| Silence / ambient (labelled "") | 35 | 36,315 |
| **Total encoded** | **~2,606** | **716,968** |

24 kHz mono, Mimi‑encoded (8 codebooks). Splits: train ~2,733 h · val 5.1 h · test 5.0 h.

## Training setup (1× H100 80 GB)

| | |
|---|---|
| Method | **full fine‑tune** (all params), causal‑LM cross‑entropy |
| Effective batch | 128 (micro‑batch 32 × grad‑accum 4) |
| LR / schedule | 2e‑4, cosine, 3 % warmup |
| Epochs / steps | 2 / 11,158 |
| Precision | bf16 + TF32, fused AdamW |
| Regularisation | SpecAugment (time masking) |
| Wall‑clock | ~3.5 h · final eval_loss 0.26 |

## Tokenizer efficiency (fertility)

The 131,072‑vocabulary tokenizer is very efficient — low **fertility** (tokens/word) means
shorter targets, faster decoding, lower CER.

| Language | Fertility (tok/word) | Round‑trip |
|---|---|---|
| **English** | **1.29** | 100 % |
| Hindi | 1.63 | 100 % |
| Gujarati | 1.60 | 100 % |
| Punjabi | 1.47 | 100 % |
| Bengali | 1.77 | 100 % |
| Marathi | 1.93 | 100 % |
| Telugu / Kannada / Tamil / Odia | 2.1–2.3 | 100 % |
| Malayalam | 3.12 | 100 % |

## Multilingual outlook (this checkpoint is EN‑only)

The backbone was pre‑trained on English + 11 Indic languages; a probe found it already
generates coherent Hindi & Gujarati and has **100 % tokenizer round‑trip across all 11
scripts**. So extending to Hindi/Gujarati/… is mainly a matter of **more data on the same
pipeline** — the model side is ready.

## Files

- `kupe-lm/` — the Kupe‑SLM‑670M backbone (weights + tokenizer).
- `frontend.pt` — audio frontend (codebook embeddings + projector).
- `asr_config.json` — frontend mode (`per_frame_sum`), codebooks (8), embed dim.

## Inference

Requires **`transformers==5.4.0`** and the
[`kupe-asr-en`](https://github.com/iNavLabsResearch/kupe-asr-en) code.

```python
from kupe_asr_en.modeling.asr_model import LummaASR   # loader class name; loads Kupe-SLM
model, tok = LummaASR.load("anuj-inavlabs/kupe-asr-en", device="cuda")
```

Streaming from a wav (turn flags):
```python
from kupe_asr_en.stream import transcribe_file
for ev in transcribe_file("model_dir", "audio.wav"):
    print(ev.type, ev.text)   # PARTIAL / PRE_HIT_LLM / END_OF_SPEECH
```

Live mic / Gradio demo (file + realtime, with RTF):
```bash
python scripts/06_mic_stream.py --model-dir <this repo>     # terminal mic
python scripts/10_gradio_kaggle.py                          # web UI (file + live mic)
```

## Related repos

- **Data:** `anuj-inavlabs/kupe-asr-en-data`
- **Runs:** `anuj-inavlabs/kupe-asr-en-runs`
- **Code:** https://github.com/iNavLabsResearch/kupe-asr-en

## License & credits

Speech‑LM backbone weights are used under their upstream license; audio codec: `kyutai/mimi`.
Data: LibriSpeech, MLCommons People's Speech (CC‑BY), AI4Bharat Svarah. Released as `other`.
Built by **iNavLabs**.
