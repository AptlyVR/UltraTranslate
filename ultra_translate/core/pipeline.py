"""Orchestrates capture -> segment -> (diarize + ASR) -> translate -> readback.

Threads (all bounded, none share a pool):
  audio source  -> frame queue (bounded, drops oldest audio only if everything is far behind)
  segment loop  -> finals queue + "latest partial" slot (partials are replaced, never queued)
  speech worker -> finals first; partials only when idle
  translator    -> per-line, with recent conversation context
  readback      -> ordered, drops stale lines
Every failure is reported through on_status; nothing is silently swallowed.
"""
from __future__ import annotations

import collections
import os
import queue
import threading
import time
import traceback

import numpy as np

from .asr import Asr
from .audio import SR, AudioSource
from .events import Line, Speaker, new_id
from .segmenter import Segment, Segmenter
from .speakers import SpeakerEngine
from .translatehub_client import TranslateHubClient, TranslateHubError
from .tts import Readback


# Languages people realistically speak in VRChat; detection outside this set is almost always noise.
COMMON_LANGS = {"en", "ja", "es", "zh", "ko", "fr", "de", "pt", "ru", "it", "nl", "pl", "tr", "ar", "th", "vi",
                "id", "uk", "sv", "fi", "no", "da", "cs", "hu", "ro", "el", "he", "hi", "tl", "ms", "yue", "bg"}


_END = tuple(".!?。！？…」』\"')")


def _complete(text: str) -> bool:
    """Whisper punctuates finished sentences; a line without final punctuation is probably unfinished."""
    return text.rstrip().endswith(_END)


class Pipeline:
    def __init__(self, settings, on_line, on_speakers, on_status, on_stats=None, share_from=None):
        self.s = settings
        self.on_line = on_line
        self.on_speakers = on_speakers
        self.on_status = on_status
        self.on_stats = on_stats or (lambda d: None)
        self.asr = Asr(settings.asr, log=self.log)
        from .endpoint import SmartTurn
        self.turn_model = SmartTurn()
        self.spk = SpeakerEngine(settings.speakers, log=self.log)
        self.hub = TranslateHubClient(settings.translation.hub_url)
        self.readback: Readback | None = None
        from .voices import VoiceAssigner
        self.voices = VoiceAssigner(settings.tts)
        self.frames: queue.Queue = queue.Queue(maxsize=int(30 * SR / 512))
        self.finals: queue.Queue = queue.Queue(maxsize=64)
        self.to_translate: queue.Queue = queue.Queue(maxsize=128)
        self.partial_slot: Segment | None = None
        self.spec_slot: Segment | None = None
        self.prepared: dict[str, dict] = {}
        self.levels: collections.deque = collections.deque(maxlen=40)
        self.tail: dict[str, dict] = {}
        from concurrent.futures import ThreadPoolExecutor
        self.mt_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spec-mt")
        self.mt_cache: dict[tuple, object] = {}
        from .interpreter import Interpreter
        self.interp = Interpreter(self)
        self.live_mt: dict[str, str] = {}
        self.partial_lang: dict[str, str] = {}
        self.live_inflight: set[str] = set()       # (text, target) -> Future[str] started speculatively
        self.pending_tts: dict[str, threading.Timer] = {}
        self.spoken_ids: set[str] = set()
        self.line_audio: dict[str, np.ndarray] = {}   # recent line audio (instant own-voice fallback)     # speaker -> last line (for re-joining early-committed sentences)
        self.partial_lock = threading.Lock()
        self.finalized: collections.deque = collections.deque(maxlen=256)
        self.context: collections.deque = collections.deque(maxlen=6)
        self.source: AudioSource | None = None
        self.running = False
        self.models_ready = False
        if share_from is not None:
            if not share_from.models_ready:
                share_from.load_models()
            self.asr = share_from.asr
            self.turn_model = share_from.turn_model
            for attr in ("seg_model", "powerset", "encoder", "device", "lock"):
                setattr(self.spk, attr, getattr(share_from.spk, attr))
            self.models_ready = True
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.session_lang = ""
        self.lang_seconds: dict[str, float] = {}
        self.stats = {"asr_ms": 0.0, "mt_ms": 0.0, "lag_ms": 0.0, "lines": 0, "dropped": 0, "partials": 0}
        self.audio_clock_origin = 0.0
        self._seg_lock = threading.Lock()
        self.inflight = 0              # finals being processed now (drain/idle checks)
        self.mt_inflight = 0

    def log(self, msg: str) -> None:
        self.on_status(msg)

    # -- lifecycle ---------------------------------------------------------------------------------
    SPEECH_NEED_MIB = int(os.environ.get("ULTRATRANSLATE_SPEECH_NEED_MIB", "2600"))  # Whisper turbo + ECAPA + seg
    HEADROOM_MIB = 3000             # the PC can stall if VRAM overflows

    def load_models(self) -> None:
        t0 = time.perf_counter()
        if self.s.asr.device == "cuda" and self.asr.model is None:
            from .vram import free_vram_mib
            free = free_vram_mib()
            if free is not None and free < self.SPEECH_NEED_MIB + self.HEADROOM_MIB:
                raise RuntimeError(f"Only {free} MiB GPU memory free; speech models need "
                                   f"{self.SPEECH_NEED_MIB} MiB plus {self.HEADROOM_MIB} MiB headroom. "
                                   "Close another GPU app first.")
        self.on_status("Loading speech models…")
        side = []

        def _background_loads():
            # ONE thread, sequential: concurrent imports of transformers/lightning in two threads break them
            if self.s.speakers.enabled:
                try:
                    import pyannote.audio  # noqa: F401   (the slow part of speaker loading is imports)
                    from speechbrain.inference.speaker import EncoderClassifier  # noqa: F401
                except Exception as exc:
                    self.on_status(f"Speaker model import hiccup ({exc}); retrying on load")
            if self.s.vad.smart_turn and self.turn_model.available:
                self.turn_model.load()
        side.append(threading.Thread(target=_background_loads, daemon=True))
        if self.s.translation.enabled and self.hub.ensure_running(wait_s=3):
            threading.Thread(target=self._preload_hub, daemon=True).start()   # translator warms in parallel
        for t in side:
            t.start()
        spk_thread = None
        if self.s.speakers.enabled and self.s.speakers.device == "cpu":
            # CPU speaker models don't depend on VRAM headroom: load them while Whisper loads
            self.spk.device = "cpu"
            spk_thread = threading.Thread(target=self._load_speakers, daemon=True)
            spk_thread.start()
        self.asr.load()
        for t in side:
            t.join()
        if spk_thread is not None:
            spk_thread.join()
        elif self.s.speakers.enabled:
            dev = self.s.speakers.device
            if dev == "auto":
                from .vram import free_vram_mib
                free = free_vram_mib()
                dev = "cuda" if free is None or free >= 700 + self.HEADROOM_MIB else "cpu"   # ~0.6 GB on GPU
            self.spk.device = dev
            self._load_speakers()
            self.on_status(f"Speaker models on {dev}.")
        if self.s.translation.enabled and not self.hub.health():
            self.on_status("TranslateHub not reachable; translations will be skipped until it is.")
        self.models_ready = True
        self.on_status(f"Models ready in {(time.perf_counter() - t0):.1f} s")

    def _load_speakers(self) -> None:
        """Load pyannote/ECAPA. A half-finished lightning import (seen once at app start) leaves broken modules
        in sys.modules ("lightning.pytorch has no attribute utilities"): purge them and import again."""
        import sys
        import traceback
        for attempt in range(3):
            try:
                self.spk.load()
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                self.on_status(f"Speaker models: retrying after import error ({exc})")
                for line in traceback.format_exc().splitlines()[-6:]:
                    self.on_status("  " + line)
                for name in [m for m in sys.modules if m.split(".")[0] in
                             ("lightning", "pytorch_lightning", "lightning_fabric", "pyannote", "torchmetrics")]:
                    sys.modules.pop(name, None)
                time.sleep(0.5)

    def _preload_hub(self) -> None:
        try:
            info = self.hub.load(self.s.translation.model)
            self.on_status(f"Translator ready ({info.get('model')}, {info.get('load_ms', 0)} ms)")
        except Exception as exc:
            self.on_status(f"Translator load failed: {exc}")

    def unload_models(self) -> None:
        self.stop()
        self.asr.unload()
        self.spk.encoder = None
        self.spk.seg_model = None
        self.models_ready = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.on_status("Speech models unloaded.")

    def start(self, source=None) -> None:
        if self.running:
            return
        if not self.models_ready:
            self.load_models()
        self.stop_event.clear()
        for q in (self.finals, self.to_translate):     # nothing stale from a previous run
            while not q.empty():
                q.get_nowait()
        self.partial_slot = None
        self.spec_slot = None
        self.prepared.clear()
        self.tail.clear()
        self.segmenter = Segmenter(self.s.vad, self._on_segment,
                                   None if getattr(self, "demo_turns", False) else self.turn_model,
                                   on_speech_start=self._on_speech_start)
        self.frames = queue.Queue(maxsize=int(30 * SR / 512))
        a = self.s.audio
        if source is not None:                     # embedded session: frames are pushed by the host app
            source.sink.out = self.frames
            self.source = source
        else:
            self.source = AudioSource(self.frames, a.source, a.device, a.file_path, a.monitor_file,
                                      self.s.tts.output_device if a.source == "file" else "", a.gain,
                                      on_error=self.on_status, on_finished=self._on_source_finished)
        self.readback = Readback(self.s.tts, log=self.on_status)
        self.ducker = None
        if a.source == "loopback" and source is None:
            if self.s.tts.echo_cancel:
                from .echo import EchoCanceller
                self.readback.echo = self.source.echo = EchoCanceller()
            if self.s.tts.duck_game:
                from .ducking import GameDucker
                self.ducker = GameDucker(level=self.s.tts.duck_level, log=self.on_status)
                rb = self.readback
                self.ducker.active = lambda: rb.playing
                self.source.comp = lambda: self.ducker.comp if self.ducker else 1.0
                self.ducker.start()
        self.threads = [threading.Thread(target=self._segment_loop, daemon=True, name="segment"),
                        threading.Thread(target=self._speech_loop, daemon=True, name="speech"),
                        threading.Thread(target=self._translate_loop, daemon=True, name="translate")]
        for t in self.threads:
            t.start()
        self.readback.start()
        if self.s.audio.source == "file" and getattr(self, "demo_turns", False):
            self.source.hold = self._demo_hold
        self.source.start()
        self.running = True
        self.on_status("Listening.")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.source:
            self.source.stop()
            self.source.join(timeout=3)
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=3)
        if self.readback:
            self.readback.stop()
        if getattr(self, "ducker", None):
            self.ducker.stop()          # fades VRChat back to the user's own volume
            self.ducker = None
        self.threads = []
        self.on_status("Stopped.")

    def _demo_hold(self) -> bool:
        """Slowed demo: after a line ends, wait until it has been translated and read aloud."""
        if self.pending_tts:
            return False      # last line looks unfinished: keep playing so its continuation can merge
        rb = self.readback
        seg = self.segmenter
        busy = ((seg.active and seg.checked) or self.interp.busy() or not self.finals.empty() or self.inflight or not self.to_translate.empty() or self.mt_inflight
                or (rb is not None and (rb.busy or not rb.q.empty())))
        if busy:
            self._hold_since = getattr(self, "_hold_since", 0.0) or time.monotonic()
            return True
        if getattr(self, "_hold_since", 0.0):        # short tail so the voice is clearly finished
            if time.monotonic() - self._hold_since < 0.6:
                return True
            self._hold_since = 0.0
        return False

    def _on_source_finished(self) -> None:
        if self.s.audio.source == "file":
            self.on_status("File finished.")

    @property
    def level(self) -> float:
        return self.source.level if self.source else 0.0

    # -- segmentation ------------------------------------------------------------------------------
    def _segment_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                t, frame = self.frames.get(timeout=0.2)
            except queue.Empty:
                if self.source is not None and not self.source.is_alive():
                    with self._seg_lock:
                        self.segmenter.flush()
                continue
            try:
                with self._seg_lock:
                    self.segmenter.push(t, frame)
            except Exception as exc:
                self.on_status(f"Segmenter error: {exc}")

    def _on_segment(self, seg: Segment) -> None:
        if seg.final:
            seg.arrived = time.perf_counter()  # type: ignore[attr-defined]
            try:
                self.finals.put(seg, timeout=1.0)
            except queue.Full:
                self.stats["dropped"] += 1
                self.on_status("Speech backlog full; dropped one utterance.")
        elif seg.kind == "spec":
            with self.partial_lock:
                self.spec_slot = seg
        elif self.s.asr.partials:
            with self.partial_lock:
                self.partial_slot = seg

    # -- speech (diarize + ASR) ----------------------------------------------------------------------
    def _speech_loop(self) -> None:
        """Priority: finals > speculative (pause just started) > live partials."""
        while not self.stop_event.is_set():
            try:
                seg = self.finals.get(timeout=0.02)
            except queue.Empty:
                with self.partial_lock:
                    spec, self.spec_slot = self.spec_slot, None
                    part = None
                    if spec is None:
                        part, self.partial_slot = self.partial_slot, None
                try:
                    if spec is not None and spec.id not in self.finalized:
                        self.prepared[spec.id] = self._prepare(spec)
                        self._speculative_translate(self.prepared[spec.id])
                        while len(self.prepared) > 8:
                            self.prepared.pop(next(iter(self.prepared)))
                    elif part is not None and part.id not in self.finalized:
                        self._do_partial(part)
                except Exception as exc:
                    self.on_status(f"Speculative/partial ASR error: {exc}")
                continue
            self.inflight += 1
            try:
                self._do_final(seg)
            except Exception as exc:
                self.on_status(f"ASR error: {exc}\n{traceback.format_exc(limit=3)}")
            finally:
                self.inflight -= 1

    def idle(self) -> bool:
        return self.finals.empty() and self.to_translate.empty() and self.inflight == 0 and self.mt_inflight == 0

    def _guess_language(self, speaker: Speaker | None) -> str | None:
        if self.s.asr.fixed_language:
            return self.s.asr.fixed_language
        if speaker and speaker.language:
            return speaker.language
        return self.session_lang or None

    def _partial_language(self, seg: Segment) -> str | None:
        """Language for a live caption: the speaker this voice matches (cached window embeddings), else
        detect once per utterance from its first ~1.2 s. Never assume the previous speaker's language."""
        if self.s.asr.fixed_language:
            return self.s.asr.fixed_language
        if seg.id in self.partial_lang:
            return self.partial_lang[seg.id]
        cache = self.spk.win_cache.get(seg.id) if self.s.speakers.enabled else None
        if cache:
            sp = self.spk.peek(cache[max(cache)])
            if sp is not None and sp.language:
                self.partial_lang[seg.id] = sp.language
                return sp.language
        if len(seg.audio) >= int(1.2 * SR):
            probs = self._detect(seg.audio[-int(8 * SR):])
            top, ptop = probs[0]
            if ptop >= 0.5:
                self.partial_lang[seg.id] = top
                while len(self.partial_lang) > 32:
                    self.partial_lang.pop(next(iter(self.partial_lang)))
            return top
        return None                                   # too short to tell: let Whisper decide

    def _do_partial(self, seg: Segment) -> None:
        if self.s.speakers.enabled and self.spk.encoder is not None:
            self.spk.precompute(seg.id, seg.audio)     # voice match for this caption (cheap, cached)
        lang = self._partial_language(seg)
        res = self.asr.transcribe(seg.audio[-int(SR * 12):], language=lang, partial=True)
        if res.rejected or seg.id in self.finalized:
            return
        self.stats["partials"] += 1
        line = Line(id=seg.id, source="voice", speaker_id="", text=res.text, language=res.language,
                    final=False, t_start=seg.t_start, t_end=seg.t_end)
        prev = self.live_mt.get(seg.id)
        if prev:
            line.translation = prev           # keep showing the last live translation while it updates
        self.on_line(line)
        self._live_translate(line)
        self.interp.on_partial(seg.id, res.text, res.language or (lang or ""), seg.audio)

    def _detect(self, audio: np.ndarray) -> list:
        return [(l, p) for l, p in self.asr.detect(audio) if l in COMMON_LANGS] or [("en", 0.0)]

    def _choose_language(self, sp: Speaker | None, audio: np.ndarray, force_detect: bool = False,
                         probs: list | None = None) -> tuple[str | None, list]:
        """Per-speaker sticky language with voting; `probs` lets a speculative detection be reused."""
        if self.s.asr.fixed_language:
            return self.s.asr.fixed_language, []
        if sp is not None and sp.language and not force_detect:
            return sp.language, []          # known speaker: no extra encoder pass (~100 ms saved)
        probs = probs or self._detect(audio)
        top, ptop = probs[0]
        dur = len(audio) / SR
        # Short or uncertain clips: trust the conversation's dominant language unless clearly otherwise.
        prior = max(self.lang_seconds.items(), key=lambda kv: kv[1])[0] if self.lang_seconds else ""
        if prior and top != prior and (dur < 2.0 and ptop < 0.85 or ptop < 0.5):
            p_prior = next((p for l, p in probs if l == prior), 0.0)
            if p_prior >= 0.02:
                top, ptop = prior, p_prior
        if sp is None:
            return top, probs
        sticky = sp.language
        if sticky:
            p_sticky = next((p for l, p in probs if l == sticky), 0.0)
            if top != sticky and ptop >= 0.80 and dur >= 1.5:
                sp.lang_votes["_switch_" + top] = sp.lang_votes.get("_switch_" + top, 0) + 1
                if sp.lang_votes["_switch_" + top] >= 2:
                    sp.language = top
                    sp.lang_votes = {top: 2}
                    return top, probs
                return (top if p_sticky < 0.02 else sticky), probs
            for k in [k for k in sp.lang_votes if k.startswith("_switch_")]:
                sp.lang_votes.pop(k)
            return sticky, probs
        if ptop >= 0.5:
            sp.lang_votes[top] = sp.lang_votes.get(top, 0) + ptop * min(dur, 6.0)
            best = max(sp.lang_votes.items(), key=lambda kv: kv[1])
            if best[1] >= 4.0 or (ptop >= 0.92 and dur >= 2.0):
                sp.language = best[0]
        return top, probs

    # -- turn preparation (shared by the speculative pass and the final) -----------------------------
    def _turns(self, audio: np.ndarray, uid: str = ""):
        from .speakers import Turn
        if self.s.speakers.enabled and self.s.speakers.overlap_split and len(audio) / SR >= 1.6:
            try:
                return self.spk.split_turns(audio, uid=uid)
            except Exception as exc:          # never lose the utterance because splitting failed
                self.on_status(f"Speaker split failed ({exc}); keeping one line.")
        return [Turn(0, len(audio))]

    def _prepare(self, seg: Segment) -> dict:
        """Everything that does not change speaker state: turns, embeddings, language detection, ASR.
        Runs speculatively when a pause starts; the final reuses it if the audio is identical."""
        t0 = time.perf_counter()
        audio = seg.audio
        items = []
        prof = {"split": 0.0, "embed": 0.0, "detect": 0.0, "asr": 0.0}
        turns = self._turns(audio, seg.id)
        prof["split"] = (time.perf_counter() - t0) * 1000
        for turn in turns:
            chunk = audio[turn.start:turn.end]
            if len(chunk) < int(0.3 * SR):
                continue
            emb = None
            guess = None
            if self.s.speakers.enabled and len(chunk) >= int(0.5 * SR):
                t1 = time.perf_counter()
                emb = self.spk.turn_embedding(seg.id, turn.start, turn.end)
                if emb is None:
                    emb = self.spk.embed(chunk)
                guess = self.spk.peek(emb)
                prof["embed"] += (time.perf_counter() - t1) * 1000
            probs: list = []
            if self.s.asr.fixed_language:
                lang = self.s.asr.fixed_language
            elif guess is not None and guess.language:
                lang = guess.language
            else:
                t1 = time.perf_counter()
                probs = self._detect(chunk)
                lang = probs[0][0]
                prof["detect"] += (time.perf_counter() - t1) * 1000
            t1 = time.perf_counter()
            res = self.asr.transcribe(chunk, language=lang)
            prof["asr"] += (time.perf_counter() - t1) * 1000
            items.append({"turn": turn, "chunk": chunk, "emb": emb, "lang": lang, "probs": probs, "res": res})
        return {"audio": audio, "items": items, "ms": (time.perf_counter() - t0) * 1000,
                "prof": {k: round(v) for k, v in prof.items()}}

    def _do_final(self, seg: Segment) -> None:
        t_start = time.perf_counter()
        self.finalized.append(seg.id)
        prep = self.prepared.pop(seg.id, None)
        reused = (prep is not None and len(prep["audio"]) == len(seg.audio)
                  and np.array_equal(prep["audio"], seg.audio))
        if reused:
            self.stats["spec_hits"] = self.stats.get("spec_hits", 0) + 1
        else:
            prep = self._prepare(seg)
        any_line = False
        for k, it in enumerate(prep["items"]):
            turn, chunk, res = it["turn"], it["chunk"], it["res"]
            sp, score = (self.spk.assign(chunk, turn.overlap, emb=it["emb"]) if self.s.speakers.enabled
                         else (None, 0.0))
            if self.spk.trace is not None and self.spk.trace and "t" not in self.spk.trace[-1]:
                self.spk.trace[-1]["t"] = [round(seg.t_start + turn.start / SR, 2), round(seg.t_start + turn.end / SR, 2)]
            lang, probs = self._choose_language(sp, chunk, probs=it["probs"] or None)
            if lang and lang != it["lang"]:
                res = self.asr.transcribe(chunk, language=lang)
            if not probs and not res.rejected and (res.avg_logprob < -0.85 or res.compression_ratio > 2.2):
                # sticky language may be wrong for this line (speaker switched language): check and retry
                lang2, probs = self._choose_language(sp, chunk, force_detect=True)
                if lang2 and lang2 != lang:
                    res2 = self.asr.transcribe(chunk, language=lang2)
                    if not res2.rejected and res2.avg_logprob > res.avg_logprob:
                        res2.ms += res.ms
                        res = res2
            line = Line(id=seg.id if k == 0 else f"{seg.id}.{k}", source="voice",
                        speaker_id=sp.id if sp else "", text=res.text, language=res.language or (lang or ""),
                        final=True, overlap=turn.overlap,
                        t_start=seg.t_start + turn.start / SR, t_end=seg.t_start + turn.end / SR)
            line.early = bool(seg.early)  # type: ignore[attr-defined]
            line.timings = {"asr": round(res.ms), "spec": bool(reused), "prep": round(prep["ms"]), **prep["prof"],
                            "commit": round((time.perf_counter() - getattr(seg, "arrived", t_start)) * 1000)}
            if not res.rejected and self._background(chunk, res):
                res.rejected = "background voice"
            if not res.rejected and sp is not None:
                merged = self._maybe_merge(sp, line, chunk)
                if merged is not None:
                    if line.id != merged[0].id:   # this fragment's live-partial bubble goes away
                        self.on_line(Line(id=line.id, source="voice", speaker_id="", final=True, dropped="merged"))
                    line, res = merged
            if not res.rejected and sp is not None:
                self.tail[sp.id] = {"line": line, "audio": chunk if line.id not in self._merged_audio
                                    else self._merged_audio.pop(line.id), "early": bool(seg.early)}
            if res.rejected:
                line.dropped = res.rejected
                if k == 0:
                    self.on_line(line)      # lets the UI remove the partial bubble
                continue
            any_line = True
            self._note_level(chunk)
            if sp is not None:
                ref_audio = self._merged_audio.get(line.id, chunk) if line.timings.get("merged") else chunk
                self.spk.offer_reference(sp.id, ref_audio, res.avg_logprob, turn.overlap)
                self.line_audio[line.id] = ref_audio
                self._prepare_voice(sp.id, line.id)
                while len(self.line_audio) > 32:
                    self.line_audio.pop(next(iter(self.line_audio)))
            self.session_lang = line.language or self.session_lang
            if line.language:
                self.lang_seconds[line.language] = self.lang_seconds.get(line.language, 0.0) + len(chunk) / SR
            self.stats["lines"] += 1
            self.stats["asr_ms"] = res.ms
            self.stats["lag_ms"] = line.timings["commit"]
            self.on_line(line)
            self._queue_translation(line)
        if self.s.speakers.enabled:
            self.on_speakers()
        if not any_line:
            self.on_line(Line(id=seg.id, source="voice", speaker_id="", final=True, dropped="no speech"))
        self.on_stats(dict(self.stats, backlog=self.finals.qsize()))

    # -- re-join sentences that Smart Turn committed early ---------------------------------------------
    _merged_audio: dict = {}

    def _maybe_merge(self, sp: Speaker, line: Line, chunk: np.ndarray):
        """Same speaker continues within 0.7 s of an EARLY commit: it was one sentence. Show it as one line
        (same id, so the UI updates in place) and re-run recognition + translation on the whole thing."""
        prev = self.tail.get(sp.id)
        if not prev:
            return None
        pl = prev["line"]
        if not prev["early"] and _complete(pl.text):
            return None              # a finished sentence that ended on a normal pause stays its own line
        gap = line.t_start - pl.t_end
        if self.source is not None and self.source.holds:   # slowed demo: inserted pauses don't count
            gap -= sum(d for t, d in self.source.holds if pl.t_end - 0.5 <= t <= line.t_start)
        if gap > 0.7 or gap < -0.5 or (pl.t_end - pl.t_start) + (line.t_end - line.t_start) > 10:
            return None
        audio = np.concatenate([prev["audio"], np.zeros(int(0.08 * SR), np.float32), chunk])
        lang, _ = self._choose_language(sp, audio, force_detect=len(prev["audio"]) < int(2.0 * SR))
        res = self.asr.transcribe(audio, language=lang)
        if res.rejected:
            return None
        parts = len(pl.text.strip()) + len(line.text.strip())
        if len(res.text.strip()) < 0.75 * parts:     # joined decode lost words: keep the two lines instead
            return None
        merged = Line(id=pl.id, source="voice", speaker_id=sp.id, text=res.text, language=res.language or lang or "",
                      final=True, overlap=pl.overlap or line.overlap, t_start=pl.t_start, t_end=line.t_end)
        merged.timings = dict(line.timings, asr=round(res.ms), merged=True, fragment_id=line.id,
                              fragment_text=line.text)
        self._merged_audio[merged.id] = audio
        self.stats["merges"] = self.stats.get("merges", 0) + 1
        return merged, res

    # -- background-voice gate ------------------------------------------------------------------------
    def _note_level(self, chunk: np.ndarray) -> None:
        self.levels.append(20 * np.log10(float(np.sqrt(np.mean(chunk ** 2))) + 1e-9))

    def _background(self, chunk: np.ndarray, res) -> bool:
        """Distant crowd chatter: much quieter than the people you are talking with AND the decoder is unsure.
        Clear speech is never dropped, however quiet."""
        if not self.s.asr.background_gate or len(self.levels) < 4:
            return False
        db = 20 * np.log10(float(np.sqrt(np.mean(chunk ** 2))) + 1e-9)
        return db < float(np.median(self.levels)) - 12.0 and res.avg_logprob < -0.55

    # -- text chat ---------------------------------------------------------------------------------
    def add_chat(self, speaker_name: str, text: str, source: str = "chat") -> None:
        line = Line(id=new_id("c"), source=source, speaker_id=speaker_name, text=text.strip(), final=True)
        try:
            from langdetect import detect_langs
            best = detect_langs(text)[0]
            line.language = best.lang.split("-")[0] if best.prob > 0.8 else ""
        except Exception:
            line.language = ""
        self.on_line(line)
        if self.s.translation.translate_chat:
            self._queue_translation(line)

    # -- translation -------------------------------------------------------------------------------
    def _queue_translation(self, line: Line) -> None:
        tr = self.s.translation
        target = tr.target.split("_")[0].lower()
        if not tr.enabled or (line.language and line.language.split("-")[0] == target):
            self.context.append(line.text)
            self._queue_readback(line, line.text, same_language=True)
            return
        try:
            self.to_translate.put_nowait(line)
        except queue.Full:
            self.on_status("Translation backlog full; showing original only.")

    def _translate_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                line = self.to_translate.get(timeout=0.2)
            except queue.Empty:
                continue
            self.mt_inflight += 1
            try:
                self._translate_one(line)
            finally:
                self.mt_inflight -= 1

    def _live_translate(self, line: Line) -> None:
        """Translate the in-progress caption (display only, never read aloud). One request per utterance
        in flight; a newer caption simply waits for the next partial."""
        tr = self.s.translation
        if not (tr.enabled and tr.live) or not line.text or len(line.text) < 4:
            return
        if (line.language or "").split("-")[0] == tr.target.split("_")[0] or line.id in self.live_inflight:
            return
        self.live_inflight.add(line.id)
        uid, text = line.id, line.text

        def done(fut):
            self.live_inflight.discard(uid)
            try:
                out = fut.result()
            except Exception:
                return
            if uid in self.finalized or not out:
                return
            self.live_mt[uid] = out
            while len(self.live_mt) > 16:
                self.live_mt.pop(next(iter(self.live_mt)))
            self.on_line(Line(id=uid, source="voice", speaker_id="", text=text, language=line.language,
                              translation=out, final=False, t_start=line.t_start, t_end=line.t_end))
        fut = self.mt_pool.submit(self.hub.translate, text, target=tr.target, source=line.language or None,
                                  context=list(self.context)[-tr.context_lines:], model=tr.model)
        fut.add_done_callback(done)

    def _speculative_translate(self, prep: dict) -> None:
        """Pause just started and the speculative ASR is in: start translating now, so the translation is
        usually ready the moment the line commits (saves the whole MT time when the text doesn't change)."""
        tr = self.s.translation
        if not tr.enabled:
            return
        target = tr.target
        ctx = list(self.context)[-tr.context_lines:] if tr.context_lines else []
        for it in prep["items"]:
            res = it["res"]
            if res.rejected or not res.text or (res.language or "").split("-")[0] == target.split("_")[0]:
                continue
            key = (res.text, target)
            if key not in self.mt_cache:
                self.mt_cache[key] = self.mt_pool.submit(self.hub.translate, res.text, target=target,
                                                         source=res.language or None, context=ctx,
                                                         model=tr.model)
        while len(self.mt_cache) > 48:
            self.mt_cache.pop(next(iter(self.mt_cache)))

    def _translate_one(self, line: Line) -> None:
        if True:
            t0 = time.perf_counter()
            try:
                ctx = list(self.context)[-self.s.translation.context_lines:] if self.s.translation.context_lines else []
                fut = self.mt_cache.pop((line.text, self.s.translation.target), None)
                text = None
                if fut is not None:
                    try:
                        text = fut.result(timeout=5)
                        line.timings["mt_spec"] = True
                    except Exception:
                        text = None
                if text is None:
                    text = self.hub.translate(line.text, target=self.s.translation.target,
                                              source=line.language or None, context=ctx,
                                              model=self.s.translation.model)
            except (TranslateHubError, OSError) as exc:
                self.on_status(f"Translation unavailable: {exc}")
                return
            except Exception as exc:
                self.on_status(f"Translation error: {exc}")
                return
            ms = (time.perf_counter() - t0) * 1000
            self.stats["mt_ms"] = ms
            line.translation = text
            line.timings["mt"] = round(ms)
            self.context.append(line.text)
            self.on_line(line)
            self._queue_readback(line, text)

    def _on_speech_start(self) -> None:
        if self.readback is not None and self.s.tts.enabled:
            self.readback.warm()

    def _own_voice(self, sid: str, line_id: str) -> tuple[str, np.ndarray | None]:
        ref = self.spk.references.get(sid)
        if ref is not None:
            return f"{sid}:v{self.spk.ref_version.get(sid, 0)}", ref[1]
        return f"{sid}:{line_id}", self.line_audio.get(line_id)     # first line: clone from this sentence

    def _prepare_voice(self, sid: str, line_id: str) -> None:
        """Start uploading the speaker's clip now; it runs while translation does (~80 ms head start+)."""
        tts = self.s.tts
        if not tts.enabled or self.readback is None or tts.voice_mode != "own":
            return
        sp = self.spk.speakers.get(sid)
        if self.voices.voice_for(sid, sp.name if sp else "") is not None:
            return                                         # a custom voice is picked for this speaker
        key, audio = self._own_voice(sid, line_id)
        if audio is not None:
            self.readback.prepare_async("own:" + key, audio)

    MERGE_WAIT_S = 0.6

    def _queue_readback(self, line: Line, text: str, same_language: bool = False) -> None:
        """Early-committed lines may still be re-joined with the speaker's next words; speaking them at once
        would read a fragment and then the whole sentence. So they wait MERGE_WAIT_S; a merge replaces them."""
        tts = self.s.tts
        if not tts.enabled or self.readback is None:
            return
        pend = self.pending_tts.pop(line.id, None)
        if pend is not None:
            pend.cancel()                      # the re-joined sentence supersedes the fragment
        if self.interp.enabled and line.source == "voice":
            # interpreter mode: clauses were spoken live; say only the rest, never re-read merged text
            frag = line.timings.get("fragment_id") if line.timings.get("merged") else None
            uid = frag or line.id
            if frag and line.id in self.interp.utts:
                uid = line.id          # this continuation was being interpreted live under its own id
            if same_language and tts.skip_own_language:
                self.interp.drop(uid)
                return
            key, voice, own = self._voice_for_line(line)
            if self.interp.on_final(uid, line.timings.get("fragment_text") if frag else line.text,
                                    line.language, key, voice, own):
                return
            if frag:
                self._speak_text(key, voice, own, line.timings.get("fragment_text") or "", line, translate=True)
                return
            self._speak_line(line, text, same_language)
            return
        if line.timings.get("merged") and line.id in self.spoken_ids:
            # the first part of this sentence was already read aloud: say only the newly joined words
            key, voice, own = self._voice_for_line(line)
            self._speak_text(key, voice, own, line.timings.get("fragment_text") or "", line, translate=True)
            return
        if (getattr(line, "early", False) or not _complete(text)) and not line.timings.get("merged_final"):
            t = threading.Timer(self.MERGE_WAIT_S, self._speak_line, (line, text, same_language))
            t.daemon = True
            self.pending_tts[line.id] = t
            t.start()
            return
        self._speak_line(line, text, same_language)

    def speak_chat(self, player: str, text: str) -> None:
        """Read a chatbox message: in the player's own cloned voice if they have spoken on mic (and were
        named), otherwise a voice assigned to their name."""
        if self.readback is None or not self.s.tts.enabled or not text:
            return
        sp = next((s for s in self.spk.speakers.values() if s.name and s.name == player), None)
        if sp is not None:
            line = Line(id=f"chat:{player}", source="voice", speaker_id=sp.id, text=text, final=True)
            key, voice, own = self._voice_for_line(line)
        else:
            voice = self.voices.voice_for("chat:" + player, player)
            if voice is None:     # own-voice mode but we never heard them: a consistent designed voice per name
                from .voices import DESIGNED, Voice
                n, c = DESIGNED[abs(hash(player)) % len(DESIGNED)]
                voice = Voice(id="design:" + n, name=n, kind="design", control=c)
            key, own = "chat:" + player, None
        self.readback.say(text, key, voice, own)

    def _voice_for_line(self, line: Line):
        sp = self.spk.speakers.get(line.speaker_id)
        name = sp.name if sp else line.speaker_id
        voice = self.voices.voice_for(line.speaker_id or name, name)
        key, own = line.speaker_id, None
        if voice is None:
            key, own = self._own_voice(line.speaker_id, line.id)
            if own is None:
                from .voices import DESIGNED, Voice
                voice = Voice(id="design:fallback", name="fallback", kind="design", control=DESIGNED[0][1])
        return key, voice, own

    def _voice_for_live(self, uid: str, audio: np.ndarray):
        """Voice for a clause spoken while the person is still talking: match the live audio to a known
        speaker (cached window embeddings), else their own audio so far."""
        sp = None
        cache = self.spk.win_cache.get(uid) if self.s.speakers.enabled else None
        if cache:
            emb = cache[max(cache)]
            sp = self.spk.peek(emb)
        if sp is None and self.spk.last_speaker in self.spk.speakers and not cache:
            sp = self.spk.speakers[self.spk.last_speaker]
        sid = sp.id if sp else f"live-{uid}"
        voice = self.voices.voice_for(sid, sp.name if sp else "") if sp else (
            None if self.s.tts.voice_mode == "own" else self.voices.voice_for(sid, ""))
        own = None
        key = sid
        if voice is None:
            ref = self.spk.references.get(sid) if sp else None
            if ref is not None:
                own, key = ref[1], f"{sid}:v{self.spk.ref_version.get(sid, 0)}"
            elif len(audio) >= int(1.5 * SR):
                own, key = audio[-int(8 * SR):].copy(), f"{sid}:{uid}"
            else:
                from .voices import DESIGNED, Voice
                voice = Voice(id="design:fallback", name="fallback", kind="design", control=DESIGNED[0][1])
        return key, voice, own

    def _speak_text(self, key, voice, own, src: str, line: Line, translate: bool) -> None:
        if not src.strip():
            return
        def go():
            try:
                out = self.hub.translate(src, target=self.s.translation.target, source=line.language or None,
                                         model=self.s.translation.model) if translate else src
            except Exception:
                return
            if out and self.readback is not None:
                self.readback.say(out, key, voice, own)
        self.interp.pool.submit(go)

    def _speak_line(self, line: Line, text: str, same_language: bool = False) -> None:
        self.pending_tts.pop(line.id, None)
        tts = self.s.tts
        if not tts.enabled or self.readback is None:
            return
        self.spoken_ids.add(line.id)
        if line.source != "voice" and not tts.read_chat:
            return
        if same_language and tts.skip_own_language:
            return
        sp = self.spk.speakers.get(line.speaker_id)
        name = sp.name if sp else line.speaker_id
        voice = self.voices.voice_for(line.speaker_id or name, name)
        own, key = None, line.speaker_id
        if voice is None:                                  # own-voice mode, no custom voice picked
            key, own = self._own_voice(line.speaker_id, line.id)
            if own is None:
                from .voices import DESIGNED, Voice
                voice = Voice(id="design:fallback", name="fallback", kind="design", control=DESIGNED[0][1])
        self.readback.say(text, key, voice, own)
