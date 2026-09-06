"""English text normalisation shared by data prep and evaluation.

WER/CER are computed on this normalised form so scoring is consistent across
sources (LibriSpeech is upper-cased, People's Speech is mixed, etc.). This is
deliberately light and reproducible, not a linguistic normaliser.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[.,!?;:\"'`~^*_=+\\/<>@#$%&(){}\[\]|…“”‘’—–\-]")
_WS = re.compile(r"\s+")

# spoken-form number map kept intentionally empty: LibriSpeech/People's Speech
# transcripts are already spelled out. Hook left here for future domains.
_NUM_WORDS: dict[str, str] = {}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def is_probably_valid(text: str, min_chars: int = 1) -> bool:
    """Cheap gate used at fetch time before we pay to resample/store a clip."""
    t = normalize(text)
    if len(t) < min_chars or not any(ch.isalpha() for ch in t):
        return False
    # drop transcripts that are mostly digits/symbols (bad labels)
    alpha = sum(ch.isalpha() or ch.isspace() for ch in t)
    return alpha / max(1, len(t)) >= 0.7
