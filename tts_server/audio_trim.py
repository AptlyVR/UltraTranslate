from __future__ import annotations

from typing import Optional

import numpy as np


class StreamingRmsOnset:
    """Find the first PCM16 RMS window without rescanning the whole stream.

    Keeps only the unfinished window and a possible split sample. This is a
    measurement helper; it neither trims nor changes the audio sent to playback.
    """

    def __init__(self, sample_rate: int, threshold_dbfs: float, window_ms: float):
        self.sample_rate = int(sample_rate)
        self.threshold_dbfs = float(threshold_dbfs)
        self.window_ms = float(window_ms)
        self.window = max(1, int(round(self.sample_rate * self.window_ms / 1000.0)))
        self._tail = np.zeros(0, dtype=np.float32)
        self._pending_byte = b""
        self._samples_seen = 0
        self.onset = None

    def feed(self, chunk: bytes) -> Optional[int]:
        if self.onset is not None:
            return self.onset
        data = self._pending_byte + chunk
        usable = len(data) - len(data) % 2
        self._pending_byte = data[usable:]
        if not usable:
            return None
        samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0
        window_audio = np.concatenate((self._tail, samples))
        origin = self._samples_seen - self._tail.size
        hit = first_rms_onset_sample(
            window_audio, self.sample_rate, self.threshold_dbfs, self.window_ms
        )
        self._samples_seen += samples.size
        if hit is not None:
            self.onset = int(origin + hit)
            self._tail = np.zeros(0, dtype=np.float32)
        else:
            keep = min(self.window - 1, window_audio.size)
            self._tail = window_audio[-keep:].copy() if keep else np.zeros(0, dtype=np.float32)
        return self.onset


def first_rms_onset_sample(
    audio: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float = -40.0,
    window_ms: float = 10.0,
) -> Optional[int]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    window = max(1, int(round(float(sample_rate) * float(window_ms) / 1000.0)))
    if arr.size < window:
        return None
    threshold_power = float(10.0 ** (float(threshold_dbfs) / 10.0))
    squared = arr.astype(np.float64) ** 2
    cumsum = np.concatenate(([0.0], np.cumsum(squared)))
    window_power = (cumsum[window:] - cumsum[:-window]) / float(window)
    hits = np.flatnonzero(window_power >= threshold_power)
    if hits.size == 0:
        return None
    return int(hits[0])


def trim_audio_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    trim_leading: bool = True,
    trim_trailing: bool = True,
    threshold_dbfs: float = -40.0,
    window_ms: float = 10.0,
    preroll_ms: float = 20.0,
) -> tuple[np.ndarray, int, int, Optional[int], Optional[int]]:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr, 0, 0, None, None

    preroll = max(0, int(round(float(sample_rate) * float(preroll_ms) / 1000.0)))
    start = 0
    end = int(arr.size)
    onset: Optional[int] = None
    tail_onset: Optional[int] = None

    if trim_leading:
        onset = first_rms_onset_sample(arr, sample_rate, threshold_dbfs, window_ms)
        if onset is not None:
            start = max(0, onset - preroll)

    if trim_trailing:
        tail_onset = first_rms_onset_sample(arr[::-1], sample_rate, threshold_dbfs, window_ms)
        if tail_onset is not None:
            end = min(arr.size, arr.size - tail_onset + preroll)

    if end <= start:
        return arr, 0, 0, onset, tail_onset
    return arr[start:end], start, int(arr.size - end), onset, tail_onset
