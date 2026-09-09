"""FastConformerASR — Nandi-Mini-150M decoding English text from FastConformer features.

Sequence per sample (English-only, no language tokens):

    [bos] [audio_bos]  a_0 .. a_{L-1}  [audio_eos]  w_0 .. w_{m-1}
    |-------------- prefix (label = -100) ----------|--- supervised text+eos ---|

Audio enters as `inputs_embeds` (Nandi's text vocab is never resized). Assembly is
identical to modeling/asr_model.py:LummaASR — only the audio source differs:
projected FastConformer features here vs. embedded Mimi codes there.

Two forward paths:
  * feats given (default)         : frozen encoder ran offline in the encode step.
  * wave given + encoder attached : LIGHT encoder fine-tuning (grad flows through it).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from ..env import hf_token, log
from .asr_model import _DTYPES, _check_transformers, load_tokenizer  # reuse decoder loader bits
from .fc_frontend import FeatureFrontend


class FastConformerASR(nn.Module):
    def __init__(self, decoder, frontend: FeatureFrontend, bos_id: int, eos_id: int,
                 max_audio_frames: int, encoder=None):
        super().__init__()
        self.lm = decoder                       # Nandi-Mini causal LM
        self.frontend = frontend
        self.encoder = encoder                  # None unless fine-tuning the encoder
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.max_audio_frames = int(max_audio_frames)
        self.config = decoder.config            # Trainer/consumers expect model.config

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_base(cls, cfg, tokenizer, enc_dim: int, encoder=None):
        from transformers import AutoModelForCausalLM
        _check_transformers()
        dtype = _DTYPES[cfg.base.dtype]
        try:
            dec = AutoModelForCausalLM.from_pretrained(
                cfg.base.decoder_id, trust_remote_code=cfg.base.trust_remote_code,
                dtype=dtype, attn_implementation=cfg.base.attn_impl, token=hf_token())
        except TypeError:                       # older kwarg name
            dec = AutoModelForCausalLM.from_pretrained(
                cfg.base.decoder_id, trust_remote_code=cfg.base.trust_remote_code,
                torch_dtype=dtype, attn_implementation=cfg.base.attn_impl, token=hf_token())
        embed_dim = dec.get_input_embeddings().weight.shape[1]
        log.info("Nandi decoder loaded | input embed dim = %d | hidden = %d",
                 embed_dim, getattr(dec.config, "hidden_size", -1))
        fe = FeatureFrontend(int(enc_dim), embed_dim, cfg.audio.projector,
                             float(cfg.audio.proj_dropout)).to(dtype)
        return cls(dec, fe, tokenizer.bos_token_id, tokenizer.eos_token_id,
                   int(cfg.model.max_audio_frames), encoder=encoder)

    # ------------------------------------------------------------------ internals
    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids)

    @property
    def _dtype(self):
        return self.lm.get_input_embeddings().weight.dtype

    @property
    def device(self):
        return self.lm.device

    def _audio_embeds(self, feats=None, num_frames=None, wave=None, wave_len=None):
        """Return (audio [B, T, E], alen [B]). Either project stored feats, or run
        the (attached) encoder on raw waveforms for light fine-tuning."""
        if self.encoder is not None and wave is not None:
            feats, flen = self.encoder.features(wave.to(self.device), wave_len.to(self.device))
            num_frames = flen
        feats = feats.to(self.device).to(self._dtype)
        audio = self.frontend(feats)                          # [B, T, E]
        alen = self.frontend.audio_len(num_frames.to(self.device))
        return audio, alen

    # ------------------------------------------------------------------ generation prefix
    def build_prefix(self, feats=None, num_frames=None, wave=None, wave_len=None):
        audio, alen = self._audio_embeds(feats, num_frames, wave, wave_len)
        bos_e = self._embed_ids(torch.tensor([self.bos_id], device=self.device))[0]
        abos, aeos = self.frontend.audio_bos, self.frontend.audio_eos
        prefixes = []
        for b in range(audio.shape[0]):
            la = int(alen[b].item())
            seq = torch.cat([bos_e[None], abos[None], audio[b, :la], aeos[None]], dim=0)
            prefixes.append(seq.to(self._dtype))
        return prefixes

    # ------------------------------------------------------------------ forward
    def forward(self, text_ids, text_lengths, feats=None, num_frames=None,
                wave=None, wave_len=None, labels=None, **_):
        device = self.device
        text_ids = text_ids.to(device)
        audio, alen = self._audio_embeds(feats, num_frames, wave, wave_len)
        text_emb = self._embed_ids(text_ids)                  # [B, M, E]
        bos_e = self._embed_ids(torch.tensor([self.bos_id], device=device))[0]
        abos, aeos = self.frontend.audio_bos, self.frontend.audio_eos

        seqs, labs, lens = [], [], []
        B = audio.shape[0]
        for b in range(B):
            la = int(alen[b].item())
            m = int(text_lengths[b].item())
            seq = torch.cat([bos_e[None], abos[None], audio[b, :la], aeos[None],
                             text_emb[b, :m]], dim=0)
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

        return self.lm(inputs_embeds=inp, attention_mask=attn, labels=lab_pad)

    # ------------------------------------------------------------------ misc
    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.lm, "gradient_checkpointing_enable"):
            self.lm.gradient_checkpointing_enable(**kw)

    def freeze_decoder(self, freeze: bool = True):
        for p in self.lm.parameters():
            p.requires_grad = not freeze
        log.info("Nandi decoder %s | frontend%s trainable",
                 "FROZEN" if freeze else "trainable",
                 "+encoder" if self.encoder is not None else "")

    def freeze_encoder(self, freeze: bool = True):
        if self.encoder is None:
            return
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
        self.encoder.train(not freeze)
        log.info("FastConformer encoder %s", "FROZEN" if freeze else "trainable (light fine-tune)")

    # ------------------------------------------------------------------ save / load
    def save(self, out_dir: str, cfg, tokenizer):
        os.makedirs(out_dir, exist_ok=True)
        self.lm.save_pretrained(os.path.join(out_dir, "kupe-lm"))
        tokenizer.save_pretrained(os.path.join(out_dir, "kupe-lm"))
        torch.save(self.frontend.state_dict(), os.path.join(out_dir, "frontend.pt"))
        if self.encoder is not None:
            torch.save(self.encoder.state_dict(), os.path.join(out_dir, "encoder_finetuned.pt"))
        meta = {
            "arch": "fastconformer+nandi",
            "enc_dim": self.frontend.enc_dim, "embed_dim": self.frontend.embed_dim,
            "projector": cfg.audio.projector, "proj_dropout": float(cfg.audio.proj_dropout),
            "bos_id": self.bos_id, "eos_id": self.eos_id,
            "max_audio_frames": self.max_audio_frames,
            "decoder_id": cfg.base.decoder_id, "encoder_id": cfg.base.encoder_id,
            "dtype": cfg.base.dtype, "encoder_finetuned": self.encoder is not None,
        }
        with open(os.path.join(out_dir, "asr_config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log.info("saved FastConformerASR -> %s", out_dir)

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu", dtype: str | None = None,
             with_encoder: bool = False):
        from transformers import AutoModelForCausalLM
        _check_transformers()
        meta = json.load(open(os.path.join(model_dir, "asr_config.json")))
        td = _DTYPES[dtype or meta.get("dtype", "float32")]
        base_sub = "kupe-lm" if os.path.isdir(os.path.join(model_dir, "kupe-lm")) else "nandi"
        dec = AutoModelForCausalLM.from_pretrained(
            os.path.join(model_dir, base_sub), trust_remote_code=True, dtype=td).to(device).eval()
        fe = FeatureFrontend(int(meta["enc_dim"]), int(meta["embed_dim"]),
                             meta.get("projector", "mlp"), float(meta.get("proj_dropout", 0.0)))
        fe.load_state_dict(torch.load(os.path.join(model_dir, "frontend.pt"), map_location="cpu"))
        fe = fe.to(device).to(td).eval()
        encoder = None
        if with_encoder:
            from .fc_encoder import FastConformerEncoder
            encoder = FastConformerEncoder.load(meta["encoder_id"], device, td, trainable=False)
            ckpt = os.path.join(model_dir, "encoder_finetuned.pt")
            if os.path.exists(ckpt):
                encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
                log.info("loaded fine-tuned encoder weights")
        tok = load_tokenizer(os.path.join(model_dir, base_sub))
        m = cls(dec, fe, meta["bos_id"], meta["eos_id"], meta["max_audio_frames"],
                encoder=encoder).to(device)
        return m, tok
