"""Streaming utterance segmenter on Silero VAD with fast endpointing.

* A pause starts: at `check_ms` of silence the audio up to that point is snapshotted. The snapshot is sent
  for speculative ASR and checked by Smart Turn.
* Confident sentence end (Smart Turn >= threshold) -> the line commits immediately (~100 ms after speech).
* Otherwise the line commits after `end_silence_ms`, with the SAME snapshot audio, so the speculative ASR
  result is reused. Speech resuming before that discards the speculation.
* Long monologues split at a pause after soft_max_s, hard-cut at the quietest point at hard_max_s.
* 400 ms pre-roll keeps first syllables.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np

from .audio import FRAME, SR
from .events import new_id

FRAME_MS = FRAME * 1000 / SR  # 32 ms


@dataclass
class Segment:
    id: str
    t_start: float
    t_end: float
    audio: np.ndarray
    final: bool
    peak_prob: float = 0.0
    kind: str = "partial"        # partial | spec | final
    early: bool = False          # committed by Smart Turn


class Vad:
    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        self.torch = torch       # Silero runs on onnxruntime; don't touch torch's global thread count
        self.model = load_silero_vad(onnx=True)

    def reset(self):
        self.model.reset_states()

    def __call__(self, frame: np.ndarray) -> float:
        return float(self.model(self.torch.from_numpy(frame), SR).item())


class Segmenter:
    def __init__(self, cfg, on_segment, turn_model=None, on_speech_start=None):
        self.cfg = cfg
        self.on_segment = on_segment
        self.on_speech_start = on_speech_start or (lambda: None)
        self.turn = turn_model if (turn_model is not None and getattr(cfg, "smart_turn", False)) else None
        self.vad = Vad()
        self.keep = max(1, math.ceil(cfg.check_ms / FRAME_MS))     # trailing silence frames kept
        self.stats = {"early": 0, "pause": 0}
        self.reset()

    def reset(self):
        self.vad.reset()
        pre = max(1, int(self.cfg.preroll_ms / FRAME_MS))
        self.preroll: collections.deque = collections.deque(maxlen=pre)
        self.active = False
        self.frames: list[np.ndarray] = []
        self.probs: list[float] = []
        self.t_start = 0.0
        self.voiced_run = 0
        self.silence_run = 0
        self.uid = ""
        self.last_partial_len = 0
        self.checked = False
        self.snap_cut = 0

    # ------------------------------------------------------------------------------------------
    def push(self, t: float, frame: np.ndarray) -> None:
        p = self.vad(frame)
        cfg = self.cfg
        if not self.active:
            self.preroll.append((t, frame, p))
            self.voiced_run = self.voiced_run + 1 if p >= cfg.start_prob else 0
            if self.voiced_run * FRAME_MS >= min(cfg.min_speech_ms, 160):
                self._start()
            return
        self.frames.append(frame)
        self.probs.append(p)
        if p < cfg.end_prob:
            self.silence_run += 1
        else:
            self.silence_run = 0
            if p >= cfg.start_prob:
                self.checked = False          # speech resumed: any speculation is void
        silence_ms = self.silence_run * FRAME_MS
        dur_s = len(self.frames) * FRAME / SR
        if not self.checked and silence_ms >= cfg.check_ms:
            self.checked = True
            self.snap_cut = len(self.frames) - self.silence_run + self.keep
            snap = self._audio(self.snap_cut)
            if self._voiced_ms(self.snap_cut) >= cfg.min_speech_ms:
                self.on_segment(Segment(self.uid, self.t_start, self.t_start + len(snap) / SR, snap,
                                        final=False, kind="spec"))
                if self.turn is not None and len(snap) >= int(0.6 * SR):
                    if self.turn.predict(snap) >= cfg.smart_threshold:
                        self.stats["early"] += 1
                        self._commit(self.snap_cut, early=True)
                        return
        if silence_ms >= cfg.end_silence_ms or (dur_s >= cfg.soft_max_s and silence_ms >= 160):
            self.stats["pause"] += 1
            self._commit(self.snap_cut if self.checked else len(self.frames) - self.silence_run + self.keep)
            return
        if dur_s >= cfg.hard_max_s:
            self._split_at_quietest(window_s=3.0)
            return
        if (cfg.partial_interval_ms > 0 and dur_s >= 0.7 and self.silence_run == 0
                and (len(self.frames) - self.last_partial_len) * FRAME_MS >= cfg.partial_interval_ms):
            self.last_partial_len = len(self.frames)
            self.on_segment(Segment(self.uid, self.t_start, self.t_start + dur_s, np.concatenate(self.frames),
                                    final=False, kind="partial"))

    def flush(self) -> None:
        if self.active:
            self._commit(len(self.frames) - self.silence_run + self.keep)

    # ------------------------------------------------------------------------------------------
    def _audio(self, cut: int) -> np.ndarray:
        return np.concatenate(self.frames[:max(1, min(cut, len(self.frames)))])

    def _voiced_ms(self, cut: int) -> float:
        return sum(1 for p in self.probs[:cut] if p >= self.cfg.start_prob) * FRAME_MS

    def _start(self) -> None:
        self.active = True
        self.uid = new_id("v")
        items = list(self.preroll)
        self.preroll.clear()
        self.t_start = items[0][0] if items else 0.0
        self.frames = [f for _, f, _ in items]
        self.probs = [p for _, _, p in items]
        self.silence_run = 0
        self.voiced_run = 0
        self.last_partial_len = 0
        self.checked = False
        self.snap_cut = 0
        self.on_speech_start()

    def _commit(self, cut: int, early: bool = False) -> None:
        cut = max(1, min(cut, len(self.frames)))
        if self._voiced_ms(cut) >= self.cfg.min_speech_ms:
            audio = self._audio(cut)
            self.on_segment(Segment(self.uid, self.t_start, self.t_start + len(audio) / SR, audio, True,
                                    peak_prob=max(self.probs[:cut]) if self.probs else 0.0, kind="final",
                                    early=early))
        tail = self.frames[cut:]
        tail_t0 = self.t_start + cut * FRAME / SR
        self.active = False
        self.frames, self.probs = [], []
        self.preroll.clear()
        keep = tail[-self.preroll.maxlen:]
        first = len(tail) - len(keep)
        for i, f in enumerate(keep):
            self.preroll.append((tail_t0 + (first + i) * FRAME / SR, f, 0.0))
        self.checked = False

    def _split_at_quietest(self, window_s: float) -> None:
        n = len(self.frames)
        w = int(window_s * SR / FRAME)
        lo = max(1, n - w)
        sm = np.convolve(np.array(self.probs[lo:]), np.ones(3) / 3, mode="same")
        cut = lo + int(np.argmin(sm))
        head_probs = self.probs[:cut]
        audio = np.concatenate(self.frames[:cut])
        self.on_segment(Segment(self.uid, self.t_start, self.t_start + len(audio) / SR, audio, True,
                                peak_prob=max(head_probs) if head_probs else 0.0, kind="final"))
        # continue the remainder as a new utterance without losing audio
        self.uid = new_id("v")
        self.t_start = self.t_start + len(audio) / SR
        self.frames, self.probs = self.frames[cut:], self.probs[cut:]
        self.silence_run = 0
        self.last_partial_len = 0
        self.checked = False
