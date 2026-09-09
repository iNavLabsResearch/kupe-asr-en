"""Feature frontend — projects FastConformer features into Nandi's input space.

FastConformer emits continuous [B, T, D_enc] features (D_enc = 512). This module
LayerNorm-normalises them and MLP-projects to the decoder's *input embedding*
width (discovered from Nandi at runtime — 512 for the factorized tied embedding,
NOT hidden_size). It also owns the audio_bos / audio_eos markers and on-the-fly
SpecAugment time masking. These are the ONLY new parameters besides the decoder;
Nandi's 131k text vocab is never resized (text already lives there).

Contrast with modeling/frontend.py (Mimi): that embeds DISCRETE codes; this
projects CONTINUOUS features. Same output contract ([B, T, E] + markers), so the
decoder wrapper (fc_asr_model) assembles sequences identically.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFrontend(nn.Module):
    def __init__(self, enc_dim: int, embed_dim: int, projector: str = "mlp",
                 dropout: float = 0.0):
        super().__init__()
        self.enc_dim = int(enc_dim)
        self.embed_dim = int(embed_dim)
        self.norm = nn.LayerNorm(self.enc_dim)
        if projector == "linear":
            self.proj = nn.Linear(self.enc_dim, self.embed_dim)
        else:  # mlp (default)
            self.proj = nn.Sequential(
                nn.Linear(self.enc_dim, self.embed_dim * 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            )
        self.audio_bos = nn.Parameter(torch.zeros(self.embed_dim))
        self.audio_eos = nn.Parameter(torch.zeros(self.embed_dim))
        self.spec_time_mask = 0        # set by trainer; applied only in train mode
        self.spec_time_blocks = 1
        self._init()

    def _init(self):
        std = self.embed_dim ** -0.5
        nn.init.normal_(self.audio_bos, std=std)
        nn.init.normal_(self.audio_eos, std=std)

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.spec_time_mask <= 0:
            return x
        b, t = x.shape[0], x.shape[1]
        for i in range(b):
            for _ in range(int(self.spec_time_blocks)):
                w = int(torch.randint(1, self.spec_time_mask + 1, (1,)).item())
                if w >= t:
                    continue
                s = int(torch.randint(0, t - w, (1,)).item())
                x[i, s:s + w] = 0.0
        return x

    def audio_len(self, num_frames: torch.Tensor) -> torch.Tensor:
        return num_frames                       # one embed vector per encoder frame

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B, T, D_enc] (0-padded past each clip's real frames) -> [B, T, E]."""
        x = self.norm(feats)
        x = self._time_mask(x)
        return self.proj(x)
