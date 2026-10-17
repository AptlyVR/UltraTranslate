"""Simultaneous interpretation: speak the translation while the person is still talking.

Policy "agree" (default): LocalAgreement-2 on the translation (as in CUNI's IWSLT 2025 system and stable-prefix
SimulMT). Each time the stable part of the live caption grows, the WHOLE stable source so far is re-translated,
with the words already spoken forced as the start of the answer (assistant prefill). New words are spoken only
once two consecutive translations agree on them, so a Japanese verb arriving at the end can still change the
English, but nothing already said is ever contradicted or repeated. When the utterance ends, the full sentence is
translated with the spoken part forced, and only the rest is said.
Replay over 8 conversations in 9 languages (tools/replay_interpreter.py, macro average), versus the older chunk
policy: quality vs the full-sentence translation (chrF) 73 -> 81, starts 4.8 -> 3.6 s after the person starts,
ends 6.1 -> 3.7 s behind them, repeated/padded words +25 % -> 0 %. Normal (whole-sentence) mode: 7.0 s / 6.3 s.

Policy "chunks" (older): clause / 8-word source chunks, each translated on its own.
Everything runs on one worker thread, so speech stays in order; stale caption updates are skipped.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯฀-๿]")
_CLAUSE = set(",.;:!?、。，；：！？…")
CJK_LANGS = {"ja", "zh", "yue", "th"}
# fillers that carry nothing for the listener (removed before translating the live prefix)
_FILLERS = re.compile(r"^(えー+|ええと|えっと|あのー*|うーん|まあ|um+|uh+|eh+|euh|ehm|嗯+|那个|呃+|음+|어+)[、,。\s]*", re.I)


def tokens(text: str, lang: str) -> list[str]:
    if lang in CJK_LANGS or (_CJK.search(text) and lang != "ko"):
        return [c for c in text if not c.isspace()]
    return text.split()


def join(toks: list[str], lang: str, sample: str = "") -> str:
    if lang in CJK_LANGS or (_CJK.search(sample) and lang != "ko"):
        return "".join(toks)
    return " ".join(toks)


def _norm(tok: str) -> str:
    return re.sub(r"[^\w]", "", tok.lower())


def lcp(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if _norm(x) != _norm(y):
            break
        n += 1
    return n


@dataclass
class Utt:
    prev: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)       # source tokens already handed to translation (chunks)
    chunks: list[str] = field(default_factory=list)     # source chunk texts (context for the next one)
    lang: str = ""
    speaker: str = ""
    # agree policy
    src: str = ""                 # newest stable source prefix waiting to be translated
    src_n: int = 0
    done_n: int = 0               # source tokens covered by the last translation
    spoken: list[str] = field(default_factory=list)     # target words already spoken
    hyp: list[str] | None = None  # unspoken tail of the previous translation
    queued: bool = False
    voice: tuple = ()


class Interpreter:
    MIN_WORDS = 3                 # speak agreed words in groups of >= 3 (no choppy one-word bursts)
    STEP = 2                      # re-translate when the stable source grew by >= 2 tokens (words / CJK chars)
    TAIL_HOLD = 2                 # never speak a hypothesis' last 2 words: closing padding ("...or something
                                  # like that") lives there and would be said twice

    def __init__(self, pipeline):
        self.p = pipeline
        self.utts: dict[str, Utt] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="interpret")
        self.lock = threading.Lock()
        self.spoken_chunks = 0

    @property
    def enabled(self) -> bool:
        s = self.p.s
        return s.tts.enabled and s.tts.simultaneous and s.translation.enabled

    @property
    def policy(self) -> str:
        return getattr(self.p.s.tts, "interpreter_policy", "agree")

    def _target(self) -> str:
        return self.p.s.translation.target.split("_")[0]

    def _min_chunk(self, cjk: bool) -> int:
        return 8 if cjk else 4

    # -- live partials ------------------------------------------------------------------------------
    def on_partial(self, uid: str, text: str, lang: str, audio) -> None:
        if not self.enabled or not text:
            return
        if (lang or "").split("-")[0] == self._target():
            return
        if self.policy == "chunks":
            return self._partial_chunks(uid, text, lang, audio)
        with self.lock:
            u = self.utts.setdefault(uid, Utt())
            cur = tokens(text, lang)
            stable = min(lcp(u.prev, cur), len(cur) - 1)   # never commit the still-growing last token
            u.prev = cur
            u.lang = lang or u.lang
            if stable <= u.src_n or stable - u.done_n < self.STEP:
                return
            u.src_n = stable
            u.src = _FILLERS.sub("", join(cur[:stable], lang, text))
            if not u.voice:
                sid, voice, own = self.p._voice_for_live(uid, audio)
                u.voice = (sid, voice, own)
                u.speaker = sid
            if u.queued:
                return                     # the worker will pick up this newer prefix when it gets to it
            u.queued = True
        self.pool.submit(self._step, uid)

    def _step(self, uid: str) -> None:
        with self.lock:
            u = self.utts.get(uid)
            if u is None:
                return
            u.queued = False
            src, n = u.src, u.src_n
            if not src or n <= u.done_n:
                return
            prefix = self._join_t(u.spoken)
        try:
            out = self.p.hub.translate_continue(src, self._target(), prefix, model=self.p.s.translation.model)
        except Exception as exc:
            self.p.on_status(f"Interpreter translation failed: {exc}")
            return
        tl = self._target()
        with self.lock:
            u = self.utts.get(uid)
            if u is None:
                return
            u.done_n = n
            hyp = self._tok_t(out)[len(u.spoken):]
            if hyp and hyp[-1][-1:] in ".!?。！？":      # a prefix translation closes the sentence early
                hyp[-1] = hyp[-1].rstrip(".!?。！？")
                if not hyp[-1]:
                    hyp = hyp[:-1]
            say: list[str] = []
            if u.hyp is not None:
                k = min(lcp(u.hyp, hyp), max(0, len(hyp) - self.TAIL_HOLD))
                if k >= self.MIN_WORDS:
                    say = hyp[:k]
            if say:
                u.spoken += say
                u.hyp = hyp[len(say):]
            else:
                u.hyp = hyp
            sid, voice, own = u.voice
        if say:
            text = self._join_t(say)
            self.spoken_chunks += 1
            self.p.on_status(f"Interpreter: [{src[-40:]}] -> [{text[:70]}]")
            if self.p.readback is not None:
                self.p.readback.say(text, sid, voice, own)

    def _tok_t(self, text: str) -> list[str]:
        return tokens(text, self._target())

    def _join_t(self, toks: list[str]) -> str:
        return join(toks, self._target())

    # -- utterance end ------------------------------------------------------------------------------
    def on_final(self, uid: str, text: str, lang: str, sid: str, voice, own) -> bool:
        """Speak what is left of the utterance. Returns True if interpretation handled this line."""
        if self.policy == "chunks":
            return self._final_chunks(uid, text, lang, sid, voice, own)
        with self.lock:
            u = self.utts.get(uid)
            if not self.enabled or u is None:
                self.utts.pop(uid, None)
                return False
        self.pool.submit(self._finish, uid, text, sid, voice, own)
        return True

    def _finish(self, uid: str, text: str, sid, voice, own) -> None:
        with self.lock:
            u = self.utts.pop(uid, None)
        if u is None:
            return
        prefix = self._join_t(u.spoken)
        try:
            out = self.p.hub.translate_continue(text, self._target(), prefix, model=self.p.s.translation.model)
        except Exception as exc:
            self.p.on_status(f"Interpreter translation failed: {exc}")
            return
        rest = self._tok_t(out)[len(u.spoken):]
        if rest and self.p.readback is not None:
            if u.voice:
                sid, voice, own = u.voice
            self.spoken_chunks += 1
            self.p.readback.say(self._join_t(rest), sid, voice, own)

    def drop(self, uid: str) -> None:
        with self.lock:
            self.utts.pop(uid, None)

    def busy(self) -> bool:
        return bool(self.pool._work_queue.qsize())

    # -- older policy: clause chunks translated on their own -------------------------------------------
    def _partial_chunks(self, uid, text, lang, audio):
        with self.lock:
            u = self.utts.setdefault(uid, Utt())
            cur = tokens(text, lang)
            stable = lcp(u.prev, cur)
            u.prev = cur
            u.lang = lang or u.lang
            cjk = lang in CJK_LANGS or bool(_CJK.search(text) and lang != "ko")
            start = len(u.said)
            stable = min(stable, len(cur) - 1)
            if stable - start < self._min_chunk(cjk):
                return
            cut = 0
            for i in range(stable - 1, start - 1, -1):
                tok = cur[i]
                if tok and tok[-1] in _CLAUSE:
                    cut = i + 1
                    break
            long_run = 8 if not cjk else 14
            if not cut and stable - start >= long_run:
                cut = stable
            if not cut or cut - start < self._min_chunk(cjk):
                return
            chunk = cur[start:cut]
            u.said = cur[:cut]
            chunk_text = join(chunk, lang, text)
            ctx = list(u.chunks[-2:])
            u.chunks.append(chunk_text)
            sid, voice, own = self.p._voice_for_live(uid, audio)
            u.speaker = sid
        self.pool.submit(self._speak, chunk_text, lang, ctx, sid, voice, own)

    def _final_chunks(self, uid, text, lang, sid, voice, own) -> bool:
        with self.lock:
            u = self.utts.pop(uid, None)
        if not self.enabled or u is None or not u.said:
            return False
        final = tokens(text, lang)
        n = len(u.said)
        match = lcp(u.said, final)
        rest = final[match:] if match >= 0.6 * n else final[n:]
        if rest:
            rest_text = join(rest, lang or u.lang, text)
            self.pool.submit(self._speak, rest_text, lang or u.lang, u.chunks[-2:], sid, voice, own)
        return True

    def _speak(self, text: str, lang: str, ctx: list[str], sid: str, voice, own) -> None:
        p = self.p
        try:
            out = p.hub.translate(text, target=p.s.translation.target, source=lang or None,
                                  context=ctx, model=p.s.translation.model)
        except Exception as exc:
            p.on_status(f"Interpreter translation failed: {exc}")
            return
        if out and p.readback is not None:
            self.spoken_chunks += 1
            p.on_status(f"Interpreter: [{text[:60]}] -> [{out[:70]}]")
            p.readback.say(out, sid, voice, own)
