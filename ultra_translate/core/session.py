"""Embeddable live-translation session (for a phone / web front end, or any other app).

    from ultra_translate.core.session import start_session
    h = start_session(tgt_lang="en", on_text=print, on_audio=send_to_phone)   # models load once, shared
    h.push_pcm16(chunk_bytes, rate=16000)     # mono int16 PCM from the phone/earbud mic, any rate
    ...
    h.stop()

on_text(event: dict)  -> {"id","final","speaker","speaker_name","language","text","translation","ms"}
                         (partials update the same id; a final replaces it; "dropped" removes it)
on_audio(pcm16: bytes, rate: int, info: dict) -> translated speech, streamed as it is generated
                         (info = {"speaker", "text", "first": bool}); nothing plays on this PC.
All sessions share ONE copy of the speech models and the shared TranslateHub translator.
"""
from __future__ import annotations

import copy
import threading

import numpy as np
import soxr

from ..settings import Settings, SettingsStore
from .audio import SR, FrameSink
from .pipeline import Pipeline

_shared: Pipeline | None = None
_shared_lock = threading.Lock()


def _models() -> Pipeline:
    """One loaded set of models (Whisper, speaker models, Smart Turn) shared by every session."""
    global _shared
    with _shared_lock:
        if _shared is None:
            base = SettingsStore().load()
            _shared = Pipeline(base, lambda *_: None, lambda: None, lambda m: None)
            _shared.load_models()
        return _shared


class PushSource:
    """AudioSource stand-in fed by push_pcm16(); resamples statefully to 16 kHz."""

    def __init__(self, out_queue):
        self.sink = FrameSink(out_queue)
        self._rs: dict[int, soxr.ResampleStream] = {}
        self._alive = True
        self.hold = lambda: False
        self.holds: list = []

    def push_pcm16(self, data: bytes, rate: int = SR) -> None:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if rate != SR:
            rs = self._rs.setdefault(rate, soxr.ResampleStream(rate, SR, 1, dtype="float32", quality="HQ"))
            x = rs.resample_chunk(x)
        self.sink.push(x)

    @property
    def level(self) -> float:
        return self.sink.level

    def start(self):
        pass

    def stop(self):
        self._alive = False

    def join(self, timeout=None):
        pass

    def is_alive(self) -> bool:
        return self._alive


class SessionHandle:
    def __init__(self, pipe: Pipeline, source: PushSource):
        self.pipe = pipe
        self.source = source

    def push_pcm16(self, data: bytes, rate: int = SR) -> None:
        self.source.push_pcm16(data, rate)

    def rename_speaker(self, speaker_id: str, name: str) -> None:
        self.pipe.spk.rename(speaker_id, name)

    def stop(self) -> None:
        self.pipe.stop()


def start_session(tgt_lang: str = "en", on_text=None, on_audio=None, voice_mode: str = "own",
                  src_lang: str = "", read_aloud: bool = True, interpreter: bool = False,
                  settings: Settings | None = None) -> SessionHandle:
    shared = _models()
    s = copy.deepcopy(settings or shared.s)
    s.translation.target = tgt_lang
    s.asr.fixed_language = src_lang or ""
    s.tts.enabled = bool(read_aloud and on_audio)
    s.tts.voice_mode = voice_mode
    s.tts.simultaneous = interpreter
    s.audio.source = "push"
    on_text = on_text or (lambda e: None)
    holder: dict = {}

    def line_cb(ln):
        p = holder.get("p")
        sp = p.spk.speakers.get(ln.speaker_id) if p else None
        on_text({"id": ln.id, "final": ln.final, "dropped": ln.dropped, "speaker": ln.speaker_id,
                 "speaker_name": sp.label if sp else ln.speaker_id, "language": ln.language,
                 "text": ln.text, "translation": ln.translation, "ms": ln.timings})

    pipe = Pipeline(s, line_cb, lambda: None, lambda m: None, share_from=shared)
    holder["p"] = pipe
    src = PushSource(pipe.frames)
    pipe.start(source=src)
    if pipe.readback is not None and on_audio:
        pipe.readback.sink = on_audio
    return SessionHandle(pipe, src)
