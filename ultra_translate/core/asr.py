"""Multilingual ASR on faster-whisper (local CUDA), with real hallucination defences."""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass

import numpy as np

# Whole-utterance phrases Whisper invents on silence/music/noise (YouTube-caption training artefacts).
_HALLUCINATIONS = [
    "thanks for watching", "thank you for watching", "please subscribe",
    "subtitles by", "subtitles by the amara.org community", "transcription by",
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "チャンネル登録",
    "最後までご視聴いただきありがとうございました", "字幕", "以下の動画",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目", "字幕由amara.org社区提供", "谢谢观看", "謝謝觀看", "感谢观看",
    "시청해주셔서 감사합니다", "구독과 좋아요", "mbc 뉴스", "끝까지 시청해주셔서 감사합니다",
    "gracias por ver", "suscríbete", "subtítulos realizados por la comunidad de amara.org",
    "sous-titres réalisés par la communauté d'amara.org", "merci d'avoir regardé",
    "untertitel im auftrag des zdf", "untertitel der amara.org-community", "vielen dank fürs zuschauen",
    "продолжение следует", "субтитры сделал", "спасибо за просмотр", "редактор субтитров",
    "obrigado por assistir", "le" + "gendas pela comunidade amara.org", "grazie per la visione",
    "sottotitoli creati dalla comunità amara.org", "terima kasih telah menonton",
    "cảm ơn các bạn đã theo dõi", "ขอบคุณที่รับชม",
]
_HALLU_NORM = {re.sub(r"[\s\W_]+", "", h.lower()) for h in _HALLUCINATIONS}
# Real words people do say; only dropped when the decoder itself was unsure.
_WEAK = {"you", "thankyou", "thanks", "bye", "okay", "ok", "hmm", "uh", "um", "ah", "oh", "yeah", "so",
         "ありがとうございました", "おやすみなさい", "お疲れ様でした", "はい", "うん", "谢谢", "嗯", "감사합니다", "네", "gracias", "danke", "merci", "спасибо"}
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def _norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", text).lower())


def collapse_repeats(text: str, max_repeat: int = 2) -> str:
    """Collapse runaway loops: words/phrases (or CJK char-grams) repeated more than max_repeat times."""
    text = text.strip()
    if not text:
        return text
    if _CJK.search(text):
        for n in range(1, 13):
            pat = re.compile(r"((?:.{%d}))\1{%d,}" % (n, max_repeat))
            text = pat.sub(lambda m: m.group(1) * max_repeat, text)
        return text
    words = text.split()
    for n in range(1, 9):
        out, i = [], 0
        while i < len(words):
            gram = words[i:i + n]
            reps = 1
            while words[i + reps * n:i + (reps + 1) * n] == gram and len(gram) == n:
                reps += 1
            if reps == 1:
                out.append(words[i])
                i += 1
                continue
            out.extend(gram * min(reps, max_repeat))
            i += n * reps
        words = out
    return " ".join(words)


def is_hallucination(text: str) -> bool:
    n = _norm(text)
    if not n:
        return True
    if n in _HALLU_NORM:
        return True
    return any(len(h) >= 6 and n.startswith(h) and len(n) <= len(h) + 4 for h in _HALLU_NORM)


@dataclass
class AsrResult:
    text: str
    language: str
    language_prob: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    ms: float
    rejected: str = ""


class Asr:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.model = None
        self.lock = threading.Lock()
        self.load_ms = 0.0

    def load(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        t0 = time.perf_counter()
        name = {"large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo"}.get(self.cfg.model, self.cfg.model)
        try:
            self.model = WhisperModel(name, device=self.cfg.device, compute_type=self.cfg.compute_type,
                                      local_files_only=True)
        except Exception:
            self.model = WhisperModel(name, device=self.cfg.device, compute_type=self.cfg.compute_type)
        # warm the CUDA kernels so the first real utterance is fast
        list(self.model.transcribe(np.zeros(16000, np.float32), language="en", beam_size=1,
                                   without_timestamps=True)[0])
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.log(f"ASR {self.cfg.model} ready in {self.load_ms:.0f} ms")

    def unload(self) -> None:
        with self.lock:
            self.model = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def detect(self, audio: np.ndarray) -> list[tuple[str, float]]:
        """Return [(lang, prob)] sorted desc."""
        with self.lock:
            _, _, probs = self.model.detect_language(audio=audio, vad_filter=False, language_detection_segments=1)
        return sorted(((l, float(p)) for l, p in probs), key=lambda x: -x[1])

    def transcribe(self, audio: np.ndarray, language: str | None, partial: bool = False,
                   prompt: str | None = None) -> AsrResult:
        t0 = time.perf_counter()
        kwargs = dict(
            language=language, task="transcribe", without_timestamps=True, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=prompt or None,
            suppress_blank=True, word_timestamps=False,
        )
        if partial:
            kwargs.update(beam_size=1, best_of=1, temperature=0.0, compression_ratio_threshold=None,
                          log_prob_threshold=None, no_speech_threshold=None)
        else:
            kwargs.update(beam_size=self.cfg.beam_size, best_of=2, patience=1.0,
                          temperature=[0.0, 0.2, 0.4, 0.6, 0.8], compression_ratio_threshold=2.2,
                          log_prob_threshold=-1.0, no_speech_threshold=0.6, repetition_penalty=1.05,
                          no_repeat_ngram_size=0)
        with self.lock:
            segments, info = self.model.transcribe(audio, **kwargs)
            segs = list(segments)
        ms = (time.perf_counter() - t0) * 1000
        texts, lp, nsp, cr = [], [], [], []
        for s in segs:
            if not partial and s.no_speech_prob > 0.6 and s.avg_logprob < -0.7:
                continue
            if not partial and s.compression_ratio > 2.6:
                continue
            texts.append(s.text.strip())
            lp.append(s.avg_logprob)
            nsp.append(s.no_speech_prob)
            cr.append(s.compression_ratio)
        joiner = "" if (info.language in ("ja", "zh", "yue", "th")) else " "
        text = collapse_repeats(joiner.join(t for t in texts if t).strip())
        res = AsrResult(text=text, language=info.language, language_prob=float(info.language_probability or 0),
                        avg_logprob=float(np.mean(lp)) if lp else -9.0,
                        no_speech_prob=float(np.max(nsp)) if nsp else 1.0,
                        compression_ratio=float(np.max(cr)) if cr else 0.0, ms=ms)
        if not text:
            res.rejected = "empty"
        elif is_hallucination(text):
            res.rejected = "hallucination phrase"
        elif not partial and _norm(text) in _WEAK and (res.no_speech_prob > 0.25 or res.avg_logprob < -0.55):
            res.rejected = "weak filler"
        elif not partial and res.avg_logprob < -1.2 and res.no_speech_prob > 0.4:
            res.rejected = "low confidence"
        return res
