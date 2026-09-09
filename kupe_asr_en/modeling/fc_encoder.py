"""FastConformer acoustic encoder (NVIDIA NeMo) — waveform -> continuous features.

Wraps `nvidia/stt_en_fastconformer_hybrid_large_pc` and exposes ONLY its
preprocessor + Conformer encoder (the RNNT/CTC decoders are dropped — we decode
text with Nandi, not with NeMo). Output is a per-frame feature sequence at
12.5 fps, d_model = 512, which the fc_frontend projects into Nandi's input space.

Two consumers:
  * data/fc_encode.py — batch-encode 3000 h to the `fc` config (frozen, no grad).
  * fc_asr_model.py   — optional LIGHT encoder fine-tuning (grad flows) at train time.

The model runs at 16 kHz; our `raw` config is 24 kHz, so `encode` resamples.
"""
from __future__ import annotations

import numpy as np
import torch

from ..constants import FC_DEFAULT_ID, FC_SAMPLE_RATE
from ..env import log


def _resample(a: np.ndarray, sr: int, target_sr: int = FC_SAMPLE_RATE) -> np.ndarray:
    if sr == target_sr:
        return np.ascontiguousarray(a, dtype=np.float32)
    import librosa
    return np.ascontiguousarray(
        librosa.resample(np.asarray(a, np.float32), orig_sr=sr, target_sr=target_sr),
        dtype=np.float32)


class FastConformerEncoder(torch.nn.Module):
    """Thin nn.Module over NeMo's preprocessor + encoder. `feat_dim` = encoder d_model."""

    def __init__(self, nemo_model):
        super().__init__()
        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        self.feat_dim = int(getattr(nemo_model.encoder, "d_model", 512))

    @classmethod
    def load(cls, model_id: str = FC_DEFAULT_ID, device: str = "cpu",
             dtype: torch.dtype = torch.float32, trainable: bool = False):
        from nemo.collections.asr.models import ASRModel
        m = ASRModel.from_pretrained(model_name=model_id, map_location="cpu")
        enc = cls(m)
        enc = enc.to(device).to(dtype)
        enc.train(trainable)
        for p in enc.parameters():
            p.requires_grad = trainable
        log.info("FastConformer loaded | id=%s | feat_dim=%d | trainable=%s",
                 model_id, enc.feat_dim, trainable)
        return enc

    # ------------------------------------------------------------------ tensor path
    def features(self, wave: torch.Tensor, wave_len: torch.Tensor):
        """wave [B, S] float @16k, wave_len [B] -> (feats [B, T, D], feat_len [B])."""
        proc, proc_len = self.preprocessor(input_signal=wave, length=wave_len)
        enc, enc_len = self.encoder(audio_signal=proc, length=proc_len)   # enc [B, D, T]
        return enc.transpose(1, 2).contiguous(), enc_len                  # -> [B, T, D]

    # ------------------------------------------------------------------ numpy path (encode step)
    @torch.inference_mode()
    def encode_arrays(self, arrays, srs, device: str, autocast: bool):
        """List of float32 waveforms (+ per-clip sr) -> list of float16 [T, D] arrays."""
        waves = [_resample(a, s) for a, s in zip(arrays, srs)]
        lens = torch.tensor([w.shape[0] for w in waves], dtype=torch.long, device=device)
        S = int(lens.max().item())
        batch = torch.zeros(len(waves), S, dtype=torch.float32, device=device)
        for i, w in enumerate(waves):
            batch[i, : w.shape[0]] = torch.from_numpy(w).to(device)
        ac = (torch.autocast("cuda", dtype=torch.float16) if autocast
              else torch.autocast("cpu", enabled=False))
        with ac:
            feats, flen = self.features(batch, lens)
        feats = feats.float().cpu().numpy()
        flen = flen.cpu().numpy()
        return [feats[i, : int(flen[i])].astype(np.float16) for i in range(len(waves))]
