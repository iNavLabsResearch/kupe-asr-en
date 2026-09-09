# =========================================================================
# kupe-asr-en — reproducible GPU image for the FastConformer + Nandi track.
#
# The base image ALREADY ships the matched torch / torchvision / torchaudio
# 2.4.1+cu124 trio — that is the whole point: it ends the torch<->torchaudio
# ABI-mismatch loop for good. On top we pin the ONE NeMo release that imports
# cleanly on torch 2.4:
#
#   * NeMo 2.0.0  — last version compatible with torch 2.4. NeMo 3.x uses
#                   `torch.nn.Buffer`, which only exists in torch >= 2.6, so it
#                   CANNOT run on this trio. Do not "upgrade" it.
#   * transformers 5.4.0 — required by the Nandi-Mini-150M decoder.
#
# Build:  docker build -t kupe-asr-fc .
# Run:    docker run --gpus all -it --rm \
#           -e HF_TOKEN=hf_xxx -e WANDB_API_KEY=xxx \
#           -v $PWD/artifacts:/kupe-asr-en/artifacts \
#           kupe-asr-fc
# =========================================================================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    PYTHONUNBUFFERED=1

# system libs: audio decode (libsndfile / ffmpeg), git, tmux for detached runs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 tmux ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /kupe-asr-en

# --- Python deps, in the order that keeps the torch trio pinned ---
COPY requirements.txt ./
RUN pip install "nemo_toolkit[asr]==2.0.0" && \
    pip install -r requirements.txt && \
    # belt-and-suspenders: force the trio back to 2.4.1 in case a dep nudged it,
    # --no-deps so the correct cu124 CUDA libs from the base image are untouched.
    pip install --no-deps --force-reinstall \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install "transformers==5.4.0"

# Fail the BUILD (not a 3-hour run) if the dep stack is inconsistent.
RUN python -c "import torch,torchaudio,torchvision; print('torch', torch.__version__, '| audio', torchaudio.__version__, '| vision', torchvision.__version__)" && \
    python -c "from nemo.collections.asr.models import ASRModel; print('nemo import OK')"

# repo code last, so editing it doesn't re-trigger the dep install layers
COPY . .

# offline end-to-end sanity check (no downloads, no GPU) as a final build gate
RUN python scripts/14_fc_smoke.py

CMD ["bash"]
