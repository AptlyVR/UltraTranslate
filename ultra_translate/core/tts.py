"""Ordered read-aloud through a VoxCPM2 TTS server, fast path:

* /v1/tts/warm at speech onset (GPU leaves idle clocks before the line is even transcribed)
* clone references sent once, then reused by reference_id (X-Reference-ID), full resend if evicted
* keep-alive HTTP session (no per-request connect cost), streaming PCM read in 512-byte chunks
* one output stream kept open across lines (no device open per line), sentence-level lines
* one line at a time, in order; the TTS server may be shared, so a busy server is retried briefly
  and stale lines (older than max_wait_s) are dropped rather than piling up
"""
from __future__ import annotations

import base64
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .voices import Voice


@dataclass
class Job:
    text: str
    speaker_id: str
    voice: Voice | None
    own_reference: np.ndarray | None
    created: float


def _wav16_b64(audio: np.ndarray, sr: int = 16000) -> str:
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


class Readback(threading.Thread):
    def __init__(self, cfg, log=print):
        super().__init__(daemon=True, name="readback")
        self.cfg = cfg
        self.log = log
        self.q: queue.Queue[Job] = queue.Queue(maxsize=16)
        self.stop_event = threading.Event()
        self.ref_ids: dict[str, str] = {}        # voice/speaker key -> server reference_id
        self.ref_b64: dict[str, str] = {}
        self.busy = False
        self.spoken = 0
        self.dropped = 0
        self.first_audio_ms: list[float] = []
        self._session = None
        self._out = None
        self._out_rate = 0
        self._last_warm = 0.0
        self.sink = None                 # embedded use: callable(pcm16_bytes, rate, info) instead of a device
        self.pending: dict[str, threading.Event] = {}
        self.echo = None                 # EchoCanceller fed with every played chunk
        self.playing_until = 0.0         # monotonic time the current audio is expected to end

    @property
    def playing(self) -> bool:
        return time.monotonic() < self.playing_until

    # -- public ------------------------------------------------------------------------------------
    def say(self, text: str, speaker_id: str, voice: Voice | None, own_reference: np.ndarray | None = None) -> None:
        if not self.cfg.enabled or not text.strip():
            return
        try:
            self.q.put_nowait(Job(text.strip(), speaker_id, voice, own_reference, time.monotonic()))
        except queue.Full:
            self.dropped += 1

    def warm(self) -> None:
        """Speech onset: wake the TTS GPU (fire-and-forget, at most every 4 s)."""
        if not self.cfg.enabled or time.monotonic() - self._last_warm < 4.0:
            return
        self._last_warm = time.monotonic()
        url = self.cfg.url.rstrip("/") + "/warm"

        def go():
            try:
                self._sess().post(url, timeout=(0.5, 1.5))
            except Exception:
                pass
        threading.Thread(target=go, daemon=True).start()

    def prepare_async(self, key: str, audio: np.ndarray) -> None:
        """Upload a speaker's voice clip ahead of time (/v1/tts/prepare) so their first read-aloud line
        starts from a cached reference instead of uploading the clip with the request."""
        if not self.cfg.enabled or key in self.ref_ids or key in self.pending:
            return
        ev = threading.Event()
        self.pending[key] = ev
        url = self.cfg.url.rstrip("/") + "/prepare"

        def go():
            import uuid
            try:
                r = self._sess().post(url, json={"mode": "clone", "use_memory_cache": "on",
                                                 "preparation_id": uuid.uuid4().hex,
                                                 "references": [{"audio": _wav16_b64(audio)}]}, timeout=(1, 10))
                if r.status_code == 200 and r.json().get("reference_id"):
                    self.ref_ids[key] = r.json()["reference_id"]
            except Exception:
                pass
            finally:
                ev.set()
                self.pending.pop(key, None)
        threading.Thread(target=go, daemon=True).start()

    def clear(self) -> None:
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        self.stop_event.set()
        self.clear()

    # -- worker ------------------------------------------------------------------------------------
    def _sess(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.trust_env = False
            self._session = s
        return self._session

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            if time.monotonic() - job.created > self.cfg.max_wait_s:
                self.dropped += 1
                continue
            try:
                self.busy = True
                self._speak(job)
                self.spoken += 1
                who = job.voice.name if job.voice else "own cloned voice"
                ttfa = self.first_audio_ms[-1] if self.first_audio_ms else 0
                self.log(f"Read aloud [{job.speaker_id.split(':')[0]}] in {who}: first audio {ttfa:.0f} ms - "
                         f"{job.text[:60]}")
            except Exception as exc:
                self.dropped += 1
                self.log(f"Read-aloud failed: {exc}")
            finally:
                self.busy = False
        self._close_out()

    def _payload(self, job: Job, full_reference: bool) -> tuple[dict, str]:
        body = {"text": job.text, "streaming": True, "streaming_prefix_len": 2, "format": "wav",
                "cfg_value": 2.0, "inference_timesteps": 8, "use_memory_cache": "on", "max_new_tokens": 768}
        v = job.voice
        if v is None and job.own_reference is not None:              # speaker's own voice
            key = "own:" + job.speaker_id
            ev = self.pending.get(key)
            if ev is not None and not full_reference:
                ev.wait(0.6)                                          # prepare in flight: usually done
            body["mode"] = "clone"
            if key in self.ref_ids and not full_reference:
                body["reference_id"], body["references"] = self.ref_ids[key], []
            else:
                body["references"] = [{"audio": _wav16_b64(job.own_reference)}]
            return body, key
        if v is None or v.kind == "design":
            body["mode"] = "voice_design"
            body["control"] = v.control if v else "a calm adult voice, clear and neutral"
            return body, ""
        key = v.id
        body["mode"] = "clone"
        if key in self.ref_ids and not full_reference:
            body["reference_id"], body["references"] = self.ref_ids[key], []
        else:
            if key not in self.ref_b64:
                with open(v.path, "rb") as f:
                    self.ref_b64[key] = base64.b64encode(f.read()).decode()
            body["references"] = [{"audio": self.ref_b64[key], "text": v.text}]
        return body, key

    def _speak(self, job: Job) -> None:
        deadline = job.created + self.cfg.max_wait_s
        full = False
        t0 = time.perf_counter()
        while True:
            body, key = self._payload(job, full)
            r = self._sess().post(self.cfg.url, json=body, stream=True, timeout=(2, 30))
            if r.status_code != 200:
                msg = r.text[:200] if not r.headers.get("Content-Type", "").startswith("application/octet") else ""
                r.close()
                if "reference" in msg.lower() and "not found" in msg.lower() and not full:
                    self.ref_ids.pop(key, None)
                    full = True
                    continue
                if r.status_code in (409, 429, 503) or "busy" in msg.lower():
                    if time.monotonic() > deadline:
                        raise RuntimeError("VoxCPM2 busy; line dropped")
                    time.sleep(0.2)
                    continue
                raise RuntimeError(f"VoxCPM2 HTTP {r.status_code}: {msg}")
            break
        ref = r.headers.get("X-Reference-ID")
        if ref and key:
            self.ref_ids[key] = ref
        rate = int(r.headers.get("X-Stream-Sample-Rate") or 48000)
        out = None if self.sink is not None else self._output(rate)
        rest = b""
        first = True
        first_chunk = True
        # The server streams ~5x faster than real time in tiny pieces. Writing 5 ms pieces to the device makes
        # any small stall (GIL, ASR burst) an audible crackle, so gather 40 ms blocks and start with 80 ms queued.
        block, start_block = rate // 25, rate // 12
        pend: list[np.ndarray] = []
        pend_n = 0
        started = False

        def flush(final: bool = False):
            nonlocal pend, pend_n, started, first_chunk
            if not pend_n or (not final and pend_n < (block if started else start_block)):
                return
            pcm = np.concatenate(pend)
            pend, pend_n = [], 0
            if self.cfg.volume != 1.0:      # read live: the volume slider changes the voice mid-sentence
                pcm = (pcm.astype(np.float32) * self.cfg.volume).clip(-32768, 32767).astype("<i2")
            if out is None:
                self.sink(pcm.tobytes(), rate, {"speaker": job.speaker_id.split(":")[0], "text": job.text,
                                                "first": first_chunk})
            else:
                self.playing_until = time.monotonic() + 0.4 + len(pcm) / rate
                out.write(pcm.reshape(-1, 1))
                if self.echo is not None:      # tell the echo canceller exactly what we played
                    self.echo.ref_push(pcm, rate)
            first_chunk = False
            started = True

        for chunk in r.iter_content(4096):
            if self.stop_event.is_set():
                break
            if not chunk:
                continue
            if first:
                self.first_audio_ms.append((time.perf_counter() - t0) * 1000)
                first = False
            data = rest + chunk
            n = len(data) // 2 * 2
            rest = data[n:]
            x = np.frombuffer(data[:n], dtype="<i2")
            pend.append(x)
            pend_n += len(x)
            flush()
        flush(final=True)
        r.close()

    def _output(self, rate: int):
        if self._out is not None and self._out_rate == rate:
            return self._out
        self._close_out()
        import sounddevice as sd
        dev = None
        if self.cfg.output_device:
            for i, d in enumerate(sd.query_devices()):
                if d["max_output_channels"] > 0 and self.cfg.output_device.lower() in d["name"].lower():
                    dev = i
                    break
        self._out = sd.OutputStream(samplerate=rate, channels=1, dtype="int16", device=dev, latency=0.06)
        self._out.start()
        self._out_rate = rate
        return self._out

    def _close_out(self) -> None:
        if self._out is not None:
            try:
                self._out.stop()
                self._out.close()
            except Exception:
                pass
            self._out = None
