"""Remove our own read-aloud voice from the loopback capture (digital echo cancellation).

The translator listens to "what you hear", which also contains the translated voice we play. On a WASAPI loopback
that echo is an exact digital copy (scaled, delayed by the output + capture buffers, no room acoustics), so we keep
the samples we played, find the delay by cross-correlation, and subtract the best-fit scaled copy. If the voice
plays on another device, or the match is weak, nothing is subtracted.
All at 16 kHz on the capture thread; ~0.1 ms per 32 ms frame, plus a small FFT search twice a second while talking.
"""
from __future__ import annotations

import threading
import time

import numpy as np

SR = 16000
REAL_SCENE = os.environ.get("REAL_SCENE", "real_conversation")   # a real recording + its .diar.json


class EchoCanceller:
    RING_S = 20
    SEARCH = (-int(0.10 * SR), int(0.45 * SR))      # capture lags the reference by about 20-300 ms

    def __init__(self):
        self.t0 = time.monotonic()
        self.n = SR * self.RING_S
        self.ref = np.zeros(self.n, np.float32)
        self.ref_end = 0                    # absolute sample index (16 kHz clock) after the last ref sample
        self.ref_last = -10 ** 9            # last absolute index with real reference audio
        self.cap_pos = None                 # absolute index of the next capture sample
        self.cap_hist = np.zeros(SR, np.float32)    # last 1 s of capture (before cancelling)
        self.delay = None                   # capture index - reference index of the same sound
        self.conf = 0.0
        self.gain = 0.0
        self._xr = self._rr = 0.0
        self._next_search = 0.0
        self._rs = None
        self._rs_rate = 0
        self.lock = threading.Lock()
        self.removed_db = 0.0               # last attenuation achieved (diagnostics)

    def _now(self) -> int:
        return int((time.monotonic() - self.t0) * SR)

    # -- reference: what we played ------------------------------------------------------------------
    def ref_push(self, pcm, rate: int) -> None:
        x = np.asarray(pcm)
        x = (x.astype(np.float32) / 32768.0) if x.dtype == np.int16 else x.astype(np.float32)
        x = x.reshape(-1)
        if rate != SR:
            import soxr
            if self._rs is None or self._rs_rate != rate:
                self._rs = soxr.ResampleStream(rate, SR, 1, dtype="float32", quality="HQ")
                self._rs_rate = rate
            x = self._rs.resample_chunk(x)
        if not len(x):
            return
        with self.lock:
            now = self._now()
            # contiguous while one utterance streams; a fresh utterance starts "now"
            start = self.ref_end if self.ref_end >= now - SR // 10 else now
            self._write(start, x)
            self.ref_end = start + len(x)
            self.ref_last = self.ref_end

    def _write(self, start: int, x: np.ndarray) -> None:
        i = start % self.n
        k = min(len(x), self.n - i)
        self.ref[i:i + k] = x[:k]
        if k < len(x):
            self.ref[:len(x) - k] = x[k:]

    def _read(self, start: int, length: int) -> np.ndarray:
        """Reference samples at absolute [start, start+length); zeros where nothing was played."""
        out = np.zeros(length, np.float32)
        lo, hi = max(start, self.ref_end - self.n), min(start + length, self.ref_end)
        if hi <= lo:
            return out
        i, k = lo % self.n, hi - lo
        seg = self.ref[i:i + k] if i + k <= self.n else np.concatenate([self.ref[i:], self.ref[:i + k - self.n]])
        out[lo - start:hi - start] = seg
        return out

    # -- capture -----------------------------------------------------------------------------------
    def process(self, x: np.ndarray) -> np.ndarray:
        if not len(x):
            return x
        with self.lock:
            now = self._now()
            if self.cap_pos is None or abs(self.cap_pos + len(x) - now) > SR // 5:
                self.cap_pos = now - len(x)          # (re)anchor capture to the clock
            p = self.cap_pos
            self.cap_pos += len(x)
            self.cap_hist = np.concatenate([self.cap_hist[len(x):], x]) if len(x) < SR else x[-SR:].copy()
            if p - self.ref_last > SR:               # nothing played for a second: pass through
                if p - self.ref_last > 5 * SR:
                    self.delay = None
                return x
            t = time.monotonic()
            if t >= self._next_search:
                self._search(p + len(x))
                self._next_search = t + (0.5 if self.delay is not None else 0.2)
            if self.delay is None or self.conf < 0.3:
                return x
            r = self._read(p - self.delay, len(x))
            rr = float(r @ r)
            if rr < 1e-7:
                return x
            # least-squares scale over the last ~0.5 s (it only changes with volume / ducking); a per-frame fit
            # would be biased by whatever else is playing
            self._xr = 0.94 * self._xr + float(x @ r)
            self._rr = 0.94 * self._rr + rr
            self.gain = float(np.clip(self._xr / self._rr, 0.0, 8.0))
            y = x - self.gain * r
            ex, ey = float(x @ x) + 1e-12, float(y @ y) + 1e-12
            if ey > ex:                              # made it worse (misaligned): don't touch the audio
                return x
            self.removed_db = 10 * np.log10(ex / ey)
            return y.astype(np.float32)

    def _search(self, cap_end: int) -> None:
        """Find the delay: correlate the last second of capture with the reference around it."""
        c = self.cap_hist
        lo, hi = self.SEARCH
        r = self._read(cap_end - len(c) - hi, len(c) + hi - lo)
        if float(r @ r) < 1e-5 or float(c @ c) < 1e-6:
            return
        nfft = 1 << int(np.ceil(np.log2(len(r) + len(c))))
        xc = np.fft.irfft(np.fft.rfft(r, nfft) * np.conj(np.fft.rfft(c, nfft)), nfft)
        span = hi - lo
        xc = xc[:span + 1]                           # r is offset so index k  <->  delay = hi - k
        k = int(np.argmax(np.abs(xc)))
        seg = r[k:k + len(c)]
        den = float(np.sqrt((seg @ seg) * (c @ c))) + 1e-12
        conf = float(abs(xc[k]) / den)
        d = hi - k
        if conf > 0.3 and (self.delay is None or abs(d - self.delay) <= 2 or conf > self.conf + 0.1):
            if self.delay is not None and abs(d - self.delay) > 2:
                self._xr = self._rr = 0.0               # new path: refit the scale
            self.delay, self.conf = d, max(conf, 0.5 * self.conf + 0.5 * conf)
        elif conf > 0.3:
            self.conf = max(0.3, self.conf * 0.95)       # a one-off different peak (pitch repeat): keep ours
        elif self.delay is not None:
            self.conf *= 0.7
