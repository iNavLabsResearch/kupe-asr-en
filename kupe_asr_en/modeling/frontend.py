"""Audio frontend — turns Mimi codes into embedding vectors in Lumma's input space.

This is the Phase-1 A/B experiment, isolated behind one module:

  per_frame_sum : E(t) = proj( Σ_c  Emb_c(code[c,t]) )      -> 1 vector / frame (12.5/s)
  flatten       : per frame emit [proj(Emb_0(c0_t)), ..., proj(Emb_{K-1}(c_{K-1,t}))]
                                                            -> K vectors / frame

`embed_dim` MUST equal Lumma's *input embedding* width (512 for the factorized
tied embedding — NOT hidden_size 1440). We read it from the base model, never
hardcode it, so a different checkpoint can't silently break the projection.

The codebook embeddings and markers are the ONLY new parameters; the LM output
vocabulary is untouched (English text already lives in Lumma's 131k vocab), which
side-steps resizing the factorized tied embedding entirely.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import MIMI_CODEBOOK_SIZE


class AudioFrontend(nn.Module):
    def __init__(self, embed_dim: int, num_codebooks: int, mode: str = "per_frame_sum",
                 projector: str = "mlp", dropout: float = 0.0):
        super().__init__()
        assert mode in ("per_frame_sum", "flatten"), mode
        self.embed_dim = int(embed_dim)
        self.num_codebooks = int(num_codebooks)
        self.mode = mode

        self.codebooks = nn.ModuleList([
            nn.Embedding(MIMI_CODEBOOK_SIZE, self.embed_dim) for _ in range(self.num_codebooks)
        ])
        # input-only markers around the audio span (never predicted)
        self.audio_bos = nn.Parameter(torch.zeros(self.embed_dim))
        self.audio_eos = nn.Parameter(torch.zeros(self.embed_dim))

        if projector == "none":
            self.proj = nn.Identity()
        elif projector == "linear":
            self.proj = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                      nn.Linear(self.embed_dim, self.embed_dim))
        else:  # mlp
            self.proj = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim * 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            )
        # SpecAugment (time masking) — set by the trainer; applied only in train mode.
        self.spec_time_mask = 0
        self.spec_time_blocks = 1
        self._init()

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero random time spans of a [B, T, ...] tensor (frames = dim 1)."""
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

    def _init(self):
        std = self.embed_dim ** -0.5
        for e in self.codebooks:
            nn.init.normal_(e.weight, mean=0.0, std=std)
        nn.init.normal_(self.audio_bos, std=std)
        nn.init.normal_(self.audio_eos, std=std)

    def audio_len(self, num_frames: torch.Tensor) -> torch.Tensor:
        """Length of the audio embedding span for each sample (frames or frames*K)."""
        return num_frames if self.mode == "per_frame_sum" else num_frames * self.num_codebooks

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: LongTensor [B, K, T] (0-padded past each clip's real frames).

        Returns audio embeds:
          per_frame_sum -> [B, T, E]
          flatten       -> [B, T*K, E]
        Per-sample valid lengths are applied by the model wrapper via `audio_len`.
        """
        b, k, t = codes.shape
        assert k >= self.num_codebooks, f"encoded {k} codebooks < needed {self.num_codebooks}"
        per_cb = [self.codebooks[c](codes[:, c, :]) for c in range(self.num_codebooks)]  # K×[B,T,E]
        if self.mode == "per_frame_sum":
            x = torch.stack(per_cb, dim=0).sum(dim=0)               # [B, T, E]
            x = self._time_mask(x)
            return self.proj(x)
        # flatten: interleave -> [B, T, K, E] -> [B, T*K, E]
        x = torch.stack(per_cb, dim=2)                             # [B, T, K, E]
        x = self._time_mask(x)                                     # mask whole frames
        x = self.proj(x)
        return x.reshape(b, t * self.num_codebooks, self.embed_dim)
