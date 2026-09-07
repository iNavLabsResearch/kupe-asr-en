"""LummaASR — Lumma-0.6B decoding English text from Mimi audio embeddings.

Sequence per sample (English-only, no language tokens in Phase 1):

    [bos] [audio_bos]  a_0 .. a_{L-1}  [audio_eos]  w_0 .. w_{m-1}
    |-------------- prefix (label = -100) ----------|--- supervised text+eos ---|

Audio enters as `inputs_embeds` (validated to work on Lumma), so the LM's output
vocabulary is never resized — English text already lives in Lumma's 131k vocab.

Each sample is assembled CONTIGUOUSLY and the batch is right-padded, so RoPE
position ids stay correct (no padding in the middle of a sequence).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from ..constants import LUMMA_REQUIRED_TRANSFORMERS
from ..env import hf_token, log
from .frontend import AudioFrontend

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _check_transformers():
    import transformers
    v = transformers.__version__
    if not v.startswith("5."):
        log.warning("transformers %s detected; Lumma needs %s. If loading fails, run: "
                    "pip install transformers==%s", v, LUMMA_REQUIRED_TRANSFORMERS,
                    LUMMA_REQUIRED_TRANSFORMERS)


def load_tokenizer(base_id: str, trust_remote_code: bool = True):
    """Native LummaTokenizer (works on transformers 5.x). Falls back to loading the
    shipped tokenizer.json directly if the custom class import ever breaks."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=trust_remote_code)
    except Exception as e:
        log.warning("native tokenizer load failed (%s) -> tokenizer.json fallback", e)
        import json as _json

        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tj = hf_hub_download(base_id, "tokenizer.json", token=hf_token())
        cfg = _json.load(open(hf_hub_download(base_id, "tokenizer_config.json", token=hf_token())))
        kw = {}
        for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
            v = cfg.get(k)
            v = v.get("content") if isinstance(v, dict) else v
            if v:
                kw[k] = v
        tok = PreTrainedTokenizerFast(tokenizer_file=tj, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class LummaASR(nn.Module):
    def __init__(self, lumma, frontend: AudioFrontend, bos_id: int, eos_id: int,
                 max_audio_frames: int):
        super().__init__()
        self.lumma = lumma
        self.frontend = frontend
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.max_audio_frames = int(max_audio_frames)
        self.config = lumma.config          # Trainer/consumers expect model.config

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_base(cls, cfg, tokenizer):
        from transformers import AutoModelForCausalLM
        _check_transformers()
        dtype = _DTYPES[cfg.base.dtype]
        try:
            lumma = AutoModelForCausalLM.from_pretrained(
                cfg.base.lumma_id, trust_remote_code=cfg.base.trust_remote_code,
                dtype=dtype, attn_implementation=cfg.base.attn_impl, token=hf_token())
        except TypeError:                    # older kwarg name
            lumma = AutoModelForCausalLM.from_pretrained(
                cfg.base.lumma_id, trust_remote_code=cfg.base.trust_remote_code,
                torch_dtype=dtype, attn_implementation=cfg.base.attn_impl, token=hf_token())
        embed_dim = lumma.get_input_embeddings().weight.shape[1]   # 512 for Lumma (factorized)
        log.info("Lumma loaded | input embed dim = %d (factorized) | hidden = %d",
                 embed_dim, getattr(lumma.config, "hidden_size", -1))
        fe = AudioFrontend(embed_dim, int(cfg.audio.codebooks), cfg.audio.frontend,
                           cfg.audio.projector, float(cfg.audio.proj_dropout)).to(dtype)
        return cls(lumma, fe, tokenizer.bos_token_id, tokenizer.eos_token_id,
                   int(cfg.model.max_audio_frames))

    # ------------------------------------------------------------------ internals
    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lumma.get_input_embeddings()(ids)

    @property
    def _dtype(self):
        return self.lumma.get_input_embeddings().weight.dtype

    @property
    def device(self):
        return self.lumma.device

    def build_prefix(self, codes: torch.Tensor, num_frames: torch.Tensor):
        """Return (list of [Lp_i, E] prefix embeds) for generation. codes [B,K,T]."""
        codes = codes.to(self.device)
        audio = self.frontend(codes)                          # [B, La, E]
        alen = self.frontend.audio_len(num_frames.to(self.device))
        bos_e = self._embed_ids(torch.tensor([self.bos_id], device=self.device))[0]
        abos, aeos = self.frontend.audio_bos, self.frontend.audio_eos
        prefixes = []
        for b in range(codes.shape[0]):
            la = int(alen[b].item())
            seq = torch.cat([bos_e[None], abos[None], audio[b, :la], aeos[None]], dim=0)
            prefixes.append(seq.to(self._dtype))
        return prefixes

    # ------------------------------------------------------------------ forward
    def forward(self, codes, num_frames, text_ids, text_lengths, labels=None, **_):
        device = self.device
        codes = codes.to(device)
        num_frames = num_frames.to(device)
        text_ids = text_ids.to(device)
        audio = self.frontend(codes)                          # [B, La, E]
        alen = self.frontend.audio_len(num_frames)
        text_emb = self._embed_ids(text_ids)                  # [B, M, E]
        bos_e = self._embed_ids(torch.tensor([self.bos_id], device=device))[0]
        abos, aeos = self.frontend.audio_bos, self.frontend.audio_eos

        seqs, labs, lens = [], [], []
        B = codes.shape[0]
        for b in range(B):
            la = int(alen[b].item())
            m = int(text_lengths[b].item())
            seq = torch.cat([bos_e[None], abos[None], audio[b, :la], aeos[None],
                             text_emb[b, :m]], dim=0)          # [P+m, E]
            seqs.append(seq)
            lens.append(seq.shape[0])
            if labels is not None:
                p = la + 3                                    # bos + audio_bos + audio_eos
                lab = torch.cat([torch.full((p,), -100, dtype=torch.long, device=device),
                                 labels[b, :m].to(device)], dim=0)
                labs.append(lab)

        Lmax = max(lens)
        E = audio.shape[-1]
        inp = torch.zeros(B, Lmax, E, dtype=self._dtype, device=device)
        attn = torch.zeros(B, Lmax, dtype=torch.long, device=device)
        lab_pad = torch.full((B, Lmax), -100, dtype=torch.long, device=device) if labels is not None else None
        for b in range(B):
            L = lens[b]
            inp[b, :L] = seqs[b]
            attn[b, :L] = 1
            if labels is not None:
                lab_pad[b, :labs[b].shape[0]] = labs[b]

        return self.lumma(inputs_embeds=inp, attention_mask=attn, labels=lab_pad)

    # ------------------------------------------------------------------ misc
    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.lumma, "gradient_checkpointing_enable"):
            self.lumma.gradient_checkpointing_enable(**kw)

    def freeze_lumma(self, freeze: bool = True):
        for p in self.lumma.parameters():
            p.requires_grad = not freeze
        log.info("Lumma base %s | frontend trainable", "FROZEN" if freeze else "trainable")

    # ------------------------------------------------------------------ save / load
    def save(self, out_dir: str, cfg, tokenizer):
        os.makedirs(out_dir, exist_ok=True)
        self.lumma.save_pretrained(os.path.join(out_dir, "kupe-lm"))   # branded backbone dir
        tokenizer.save_pretrained(os.path.join(out_dir, "kupe-lm"))
        torch.save(self.frontend.state_dict(), os.path.join(out_dir, "frontend.pt"))
        meta = {
            "frontend": cfg.audio.to_dict() if hasattr(cfg.audio, "to_dict") else dict(cfg.audio),
            "bos_id": self.bos_id, "eos_id": self.eos_id,
            "max_audio_frames": self.max_audio_frames,
            "embed_dim": self.frontend.embed_dim,
            "base_id": cfg.base.lumma_id, "dtype": cfg.base.dtype,
        }
        with open(os.path.join(out_dir, "asr_config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log.info("saved LummaASR -> %s", out_dir)

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu", dtype: str | None = None):
        from transformers import AutoModelForCausalLM
        _check_transformers()
        meta = json.load(open(os.path.join(model_dir, "asr_config.json")))
        td = _DTYPES[dtype or meta.get("dtype", "float32")]
        # branded dir is "kupe-lm"; fall back to legacy "lumma" for older runs
        base_sub = "kupe-lm" if os.path.isdir(os.path.join(model_dir, "kupe-lm")) else "lumma"
        lumma = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_dir, base_sub), trust_remote_code=True, dtype=td).to(device).eval()
        fe = AudioFrontend(meta["embed_dim"], int(meta["frontend"]["codebooks"]),
                           meta["frontend"]["frontend"], meta["frontend"]["projector"],
                           float(meta["frontend"].get("proj_dropout", 0.0)))
        fe.load_state_dict(torch.load(os.path.join(model_dir, "frontend.pt"), map_location="cpu"))
        fe = fe.to(device).to(td).eval()
        tok = load_tokenizer(os.path.join(model_dir, base_sub))
        m = cls(lumma, fe, meta["bos_id"], meta["eos_id"], meta["max_audio_frames"]).to(device)
        return m, tok
