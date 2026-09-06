"""Streaming English ASR with turn flags, driven by Lumma's own P(eos).

  PARTIAL        best-so-far transcript for the current turn.
  PRE_HIT_LLM    P(eos) crossed `pre_llm_threshold` while decoding -> prefetch your LLM.
  END_OF_SPEECH  eos emitted / P(eos) >= eos_threshold / trailing silence -> commit final.

Each hop re-encodes the bounded audio buffer with Mimi (all `codebooks` codebooks),
builds the audio prefix as `inputs_embeds`, and greedy-decodes with repetition
penalty (kills "the the the"). Correct and simple; swap in KV-cache incremental
decoding for lowest latency at scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import MIMI_SAMPLE_RATE
from .denoise import StreamingRNNoise
from .modeling.asr_model import LummaASR

PARTIAL = "partial"
PRE_HIT_LLM = "pre_hit_llm"
END_OF_SPEECH = "end_of_speech"


@dataclass
class Event:
    type: str
    text: str
    t: float


def _pick_device(device):
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StreamingASR:
    def __init__(self, model_dir: str, mimi_id: str = "kyutai/mimi", device=None, *,
                 codebooks: int = 8, pre_llm_threshold=0.30, eos_threshold=0.85,
                 max_context_s=30.0, silence_ms=800.0, silence_rms=0.01,
                 max_new_tokens=200, repetition_penalty=1.3, no_repeat_ngram_size=3,
                 denoise=True):
        from transformers import MimiModel
        self.device = _pick_device(device)
        dtype = "float32" if self.device in ("cpu", "mps") else "bfloat16"
        self.model, self.tok = LummaASR.load(model_dir, device=self.device, dtype=dtype)
        self.mimi = MimiModel.from_pretrained(mimi_id).to(self.device).eval()
        self.codebooks = int(codebooks)
        self.pre_llm_threshold = pre_llm_threshold
        self.eos_threshold = eos_threshold
        self.max_context_s = max_context_s
        self.silence_ms = silence_ms
        self.silence_rms = silence_rms
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.denoiser = StreamingRNNoise() if denoise else None
        self.reset()

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self._fired_pre = False
        self._done = False
        if self.denoiser is not None:
            self.denoiser.reset()

    def add_audio(self, chunk: np.ndarray, sr: int):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.denoiser is not None:
            chunk, sr = self.denoiser.process(chunk, sr)
            if chunk.size == 0:
                return
        if sr != MIMI_SAMPLE_RATE:
            import librosa
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=MIMI_SAMPLE_RATE)
        self.buffer = np.concatenate([self.buffer, chunk])
        mx = int(self.max_context_s * MIMI_SAMPLE_RATE)
        if len(self.buffer) > mx:
            self.buffer = self.buffer[-mx:]

    @torch.inference_mode()
    def _encode(self):
        buf = np.ascontiguousarray(self.buffer, dtype=np.float32)
        iv = torch.from_numpy(buf).view(1, 1, -1).to(self.device)
        codes = self.mimi.encode(iv, num_quantizers=self.codebooks).audio_codes  # [1,K,T]
        return codes[0].to("cpu")                                                # [K, T]

    def _trailing_silence(self):
        n = int(self.silence_ms / 1000.0 * MIMI_SAMPLE_RATE)
        if len(self.buffer) < n:
            return False
        tail = self.buffer[-n:]
        return float(np.sqrt(np.mean(tail ** 2) + 1e-9)) < self.silence_rms

    @torch.inference_mode()
    def step(self) -> list[Event]:
        if self._done:
            return []
        t = len(self.buffer) / MIMI_SAMPLE_RATE
        if len(self.buffer) and float(np.sqrt(np.mean(self.buffer ** 2) + 1e-9)) < self.silence_rms:
            return []                                   # silence gate -> no hallucination
        codes = self._encode()                          # [K, T]
        nf = torch.tensor([codes.shape[1]], dtype=torch.long)
        prefix = self.model.build_prefix(codes[None], nf)[0]   # [Lp, E]
        inp = prefix[None].to(self.device)
        attn = torch.ones(1, prefix.shape[0], dtype=torch.long, device=self.device)
        gen = self.model.lumma.generate(
            inputs_embeds=inp, attention_mask=attn, max_new_tokens=self.max_new_tokens,
            do_sample=False, num_beams=1, eos_token_id=self.model.eos_id,
            pad_token_id=self.tok.pad_token_id, output_scores=True,
            return_dict_in_generate=True, repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size)
        new_ids = gen.sequences[0].tolist()

        pre_step, eos_hit = None, False
        for i, logits in enumerate(gen.scores):
            p_eos = torch.softmax(logits[0].float(), dim=-1)[self.model.eos_id].item()
            if pre_step is None and p_eos >= self.pre_llm_threshold:
                pre_step = i
            if p_eos >= self.eos_threshold or (i < len(new_ids) and new_ids[i] == self.model.eos_id):
                eos_hit = True
                break

        text_ids = new_ids[: new_ids.index(self.model.eos_id)] if self.model.eos_id in new_ids else new_ids
        text = self.tok.decode(text_ids, skip_special_tokens=True).strip()
        events = [Event(PARTIAL, text, t)]

        if pre_step is not None and not self._fired_pre:
            self._fired_pre = True
            draft = self.tok.decode(new_ids[:pre_step], skip_special_tokens=True).strip()
            events.append(Event(PRE_HIT_LLM, draft or text, t))
        if eos_hit or self._trailing_silence():
            self._done = True
            events.append(Event(END_OF_SPEECH, text, t))
        return events


def transcribe_file(model_dir: str, wav_path: str, chunk_ms=480, **kw):
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    asr = StreamingASR(model_dir, **kw)
    hop = int(sr * chunk_ms / 1000.0)
    for start in range(0, len(audio), hop):
        asr.add_audio(audio[start:start + hop], sr)
        for ev in asr.step():
            yield ev
            if ev.type == END_OF_SPEECH:
                return
