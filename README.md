# kupe-asr-en — Phase 1

**English-only streaming ASR** on **Lumma-0.6B-Base + Mimi**.
Audio → Mimi codes (c0–c7 @ 12.5 Hz) → an audio frontend embeds them → **Lumma
decodes English text**. Phase 1 exists to answer one question with real numbers:

> **Does Lumma actually learn English ASR from Mimi tokens, and does the whole
> data→encode→train→eval→stream pipeline work end to end?**

It is deliberately not production quality. See [`ARCHITECTURE.md`](ARCHITECTURE.md)
for every design decision (why Lumma over SmolLM/Gemma/Qwen, why per-frame vs
flatten, why 3× augmentation is safe, expected WER, and the go/no-go gate).

---

## 0. Prerequisites (once)

**Accept the licenses** on the HF account whose token you use:
- Model: <https://huggingface.co/FrontiersMind/Lumma-0.6B-Base>
- Codec: <https://huggingface.co/kyutai/mimi>
- Data (gated ones you enable): <https://huggingface.co/datasets/ai4bharat/Svarah>,
  <https://huggingface.co/datasets/speechcolab/gigaspeech> (optional).
  LibriSpeech and People's Speech are ungated.

**Secrets** — copy `.env.example` → `.env` and fill it, or `export`:

```bash
export HF_TOKEN=hf_xxx
export HF_OWNER=your-hf-username
export WANDB_API_KEY=xxx
export WANDB_PROJECT=kupe-asr-en
```

**Install** — torch first (matched to your CUDA), then the rest. **Lumma requires
`transformers==5.4.0`** (pinned in requirements; 4.x cannot import it).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 1. Run the whole Phase-1 pipeline

Two repos are created on the Hub and used at every stage (nothing trains from
local-only data):

- `HF_OWNER/kupe-asr-en-data` — `raw` (audio) + `mimi` (codes) configs + ledgers.
- `HF_OWNER/kupe-asr-en-runs` — every run's weights, eval reports, and ledgers.
- `HF_OWNER/kupe-asr-en` — the latest/best model.

**Recommended: two boxes.** A cheap CPU box fetches + uploads `raw`; a GPU box
encodes `mimi` and trains. Everything is resumable and rate-limit-safe.

```bash
# --- step 0 (any box, ~10 min CPU): is Lumma FIT before you spend a dollar? ---
python scripts/07_base_model_check.py            # writes an IN/OUT report card

# --- box A (CPU, lots of disk): collect English audio ---
python scripts/00_create_repos.py
python scripts/01_fetch_data.py            # streams sources, uploads ~24 `raw` bunches
python scripts/05_check_data.py            # gate: refuses to proceed below min_hours

# --- box B (GPU): encode + train + eval ---
python scripts/02_encode_data.py           # downloads `raw`, Mimi-encodes c0-c7, pushes `mimi`
python scripts/03_train.py                 # trains, live WER, final val+test eval, pushes run
python scripts/04_evaluate.py --model-dir artifacts/runs/<run>/model --push
```

Single box (needs disk for a local `raw` copy too):

```bash
make all        # repos -> fetch -> check -> encode -> train
```

**The Phase-1 A/B experiment** (run both, compare WER — this is the point):

```bash
python scripts/03_train.py                                   # per_frame_sum c0-c7  (default)
python scripts/03_train.py --frontend flatten --codebooks 4 --bs 8   # flatten c0-c3
```

Live mic transcription with a trained model:

```bash
python scripts/06_mic_stream.py --model-dir artifacts/runs/<run>/model
```

---

## 2. Recommended commands & flags

| Goal | Command |
|---|---|
| See data progress without fetching | `python scripts/01_fetch_data.py --status` |
| Hard gate before spending GPU time | `python scripts/05_check_data.py --strict` |
| Encode progress | `python scripts/02_encode_data.py --status` |
| Smaller smoke run | edit `data.min_hours` down, `--max-steps 2000` |
| Big GPU (H100/PRO6000) | `--bs 32 --grad-accum 4` |
| Multi-GPU | `accelerate launch scripts/03_train.py` |
| Resume a run anywhere | `--resume <run-name>` (pulls last checkpoint from the runs repo) |
| Local `mimi` (skip re-download) | `--local` (only right after encode on the same box) |

Defaults live in [`configs/phase1.yaml`](configs/phase1.yaml) — one file drives
every stage. Any field is overridable by CLI flag or by editing the YAML.

---

## 3. Don't lose a run to a dropped SSH session — use tmux

Fetch, encode, and train are multi-hour. Always run them inside `tmux` so a
disconnect doesn't kill them:

```bash
tmux new -s kupe                 # start a session
python scripts/02_encode_data.py # ...long job...
# detach: Ctrl-b then d          (job keeps running)
tmux attach -t kupe              # reattach later, from anywhere
tmux ls                          # list sessions
```

Every stage is **also** resumable on its own (ledgers + fingerprints), so even a
hard crash only costs the in-flight shard. `tmux` just spares you re-launching.

---

## 4. What the streaming output looks like

```
hello i would like              ← PARTIAL (updates live)
⚡ prefetch-LLM  hello i would like to      ← PRE_HIT_LLM (prefetch your LLM now)
■ hello i would like to book a table        ← END_OF_SPEECH (commit final)
```

Both flags come from Lumma's own `P(<eos>)` — no second model, no extra training.

---

## 5. Data & hours (Phase 1 target ≈ 3,000 h English)

| Source | HF id | cap (h) | gate |
|---|---|---|---|
| LibriSpeech | `openslr/librispeech_asr` | 960 | ungated |
| People's Speech | `MLCommons/peoples_speech` | 2100 | ungated (CC-BY) |
| Svarah (Indian-accent EN) | `ai4bharat/Svarah` | 12 | gated:auto |
| GigaSpeech (optional) | `speechcolab/gigaspeech` | 0 | gated (set cap>0) |

The pipeline **refuses to train** below `data.min_hours` (default 2500 h). Hours,
per-source counts, shard/byte totals and resume state are tracked in the ledgers
(`ledger/data.json`, `ledger/mimi.json` in the data repo).

---

## 6. Layout

```
configs/phase1.yaml         one config for all stages
kupe_asr_en/
  constants.py env.py config.py text.py         foundation
  ledger.py hub.py audio.py                      state, rate-limits, codecs
  data/ sources.py fetch.py encode.py shards.py bunch.py   the data factory
  modeling/ frontend.py asr_model.py             audio frontend + Lumma wrapper
  collate.py dataset.py evaluate.py train.py stream.py
scripts/ 00..06 + 10_gradio_kaggle   stages + Kaggle Gradio demo
```

---

## 7. Notes

- **Never trains on local-only data.** Every stage reads its input from the Hub
  (train pulls `mimi`; encode pulls `raw`).
- **Rate-limit safe.** Local tiny shards are packed into ~20–25 big `bunch_*.parquet`
  files; a whole dataset is ~25 commits. Downloads only touch those bunches.
- **Factory pattern for data.** Add an English source with one `Source(...)` entry
  in `kupe_asr_en/data/sources.py`; exotic schemas override `extract()`.
- **Anti-hallucination is trained in, not just gated.** Fetch injects `data.silence_frac`
  (5%) of silence/ambient clips labeled `""` so the model learns silence → emit nothing;
  eval reports a **silence→empty rate** (target >95%) alongside WER/CER.
- **W&B** logs train/val loss, live WER/CER, silence→empty; disabled cleanly if no key is set.
- If Lumma ever fails to load, the fix is almost always `pip install transformers==5.4.0`.

---

## 8. Kaggle Gradio demo (T4×2 or P100×1)

Public weights, **no HF token**. Accelerator: **GPU T4×2** (pipeline: Mimi on
`cuda:0`, Kupe-LM on `cuda:1`) or **P100×1**. Enable **Internet**.

```bash
!git clone https://github.com/iNavLabsResearch/kupe-asr-en.git
%cd kupe-asr-en
# Do NOT reinstall torch — Kaggle already ships CUDA torch.
!pip install -q "transformers==5.4.0" "tokenizers>=0.22" accelerate huggingface_hub \
    librosa soundfile soxr pyyaml numpy gradio
!python scripts/10_gradio_kaggle.py --share
```

Open the printed `https://*.gradio.live` URL. Upload audio → **Full file** (whole
clip) or **Realtime chunks** (growing transcript + RTF).

---

## FastConformer encoder + Nandi-Mini-150M track

A second architecture that reuses the **same 3000 h `raw` audio** on the Hub but
replaces the Mimi-codes frontend with a pretrained acoustic encoder:

```
raw 24 kHz audio
  ──[stage 1: fc-encode]──▶  NVIDIA FastConformer encoder  ──▶  features (12.5 fps, D=512)
  ──[stage 2: fc-train]───▶  projector + Nandi-Mini-150M   ──▶  English text
```

- Encoder: [`nvidia/stt_en_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_pc) (NeMo; 16 kHz in)
- Decoder: [`FrontiersMind/Nandi-Mini-150M`](https://huggingface.co/FrontiersMind/Nandi-Mini-150M) (transformers==5.4.0)

Data loading, encoding, and training are **three distinct, parallelizable steps**
(fetch → fc-encode → fc-train), each resumable via its own ledger and reading only
from the Hub. Config: [`configs/fastconformer_nandi.yaml`](configs/fastconformer_nandi.yaml).

**Install the encoder** (only on the GPU box that encodes / fine-tunes):

```bash
pip install "nemo_toolkit[asr]>=1.23"
```

**Offline sanity check first** (no downloads, no GPU — validates the whole assembly):

```bash
make fc-smoke        # == python scripts/14_fc_smoke.py
```

### Stage 1 — encode features (multi-GPU, run once)

Converts the `raw` config into a new `fc` config of FastConformer features. Uses
**all visible GPUs** (round-robin, OOM auto-split), streams downloads/decode/encode/
upload in a pipeline, and is resumable per raw bunch. Run in tmux.

```bash
make fc-encode                                   # or: python scripts/12_fc_encode.py
python scripts/12_fc_encode.py --status          # progress (hours encoded, bunches left)
```

### Stage 2 — train the projector + decoder

Loads the `fc` features + Nandi-Mini-150M, warms up the projector alone
(`train.freeze_base_steps`), then unfreezes the decoder, and optimizes for low WER.
Refuses to start below `data.min_hours` (lower it in the YAML for a small run).

```bash
make fc-train                                    # or: python scripts/13_fc_train.py
python scripts/13_fc_train.py --epochs 3 --bs 32 --lr 3e-4
accelerate launch scripts/13_fc_train.py         # multi-GPU DDP
```

**Optional light encoder fine-tuning** — re-encodes `raw` waveforms on the fly so
gradients flow through FastConformer (use a small LR):

```bash
make fc-train-ft                                 # --finetune-encoder --lr 1e-5
```

Weights (Nandi decoder in `kupe-lm/`, projector `frontend.pt`, optional
`encoder_finetuned.pt`, `asr_config.json`) plus eval reports and ledgers are pushed
to `{owner}/kupe-asr-en-fastConformer` (final model) and
`{owner}/kupe-asr-en-fastConformer-runs` (per-run checkpoints + evals). Reload with
`FastConformerASR.load(model_dir, with_encoder=<bool>)`.

---

## Docker (recommended for the FastConformer track)

The NeMo dependency stack is brittle: NeMo 3.x needs torch ≥ 2.6 while the
matched `torchaudio`/`torchvision` on most CUDA boxes are 2.4.1, and an unpinned
`pip install nemo_toolkit[asr]` pulls a torch-2.6/cu13 nightly that breaks both.
The [`Dockerfile`](Dockerfile) pins the one combo that imports cleanly —
**torch 2.4.1 + NeMo 2.0.0 + transformers 5.4.0** — on top of the official
`pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` base (which already ships the
matched torch trio). The build runs `scripts/14_fc_smoke.py` as its final gate,
so a green build means the stack works.

```bash
git pull origin main
docker build -t kupe-asr-fc .

docker run --gpus all -it --rm \
  -e HF_TOKEN=hf_xxx -e WANDB_API_KEY=xxx -e HF_OWNER=anuj-inavlabs \
  -v "$PWD/artifacts:/kupe-asr-en/artifacts" \
  kupe-asr-fc
```

Inside the container the whole pipeline just runs (tmux is preinstalled):

```bash
python scripts/00_create_repos.py --config configs/fastconformer_nandi.yaml --skip-data-card
python scripts/12_fc_encode.py    --config configs/fastconformer_nandi.yaml     # stage 1
python scripts/13_fc_train.py     --config configs/fastconformer_nandi.yaml --bs 64 --grad-accum 2   # stage 2
```

Mounting `-v $PWD/artifacts:...` keeps encoded features, checkpoints, and ledgers
on the host so a container restart resumes instead of recomputing.
