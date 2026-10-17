"""Offline replay of interpreter (simultaneous) policies on recorded live-caption traces.

Input: test_audio/traces/<scene>.lines.json from `eval_scenes.py --trace-lines` (every partial/final caption with
its wall time). Each policy sees the partials exactly as they arrived, calls the real translator (TranslateHub,
Hy-MT2), and its spoken output is played into a simulated speaker (one voice queue, ~2.9 English words/s).

Metrics (per utterance, then averaged):
  start_s   first translated audio, seconds after the person started talking (lower = sooner)
  behind_s  when the translated speech for the utterance ends, minus when the person stopped (lower = keeps up)
  chrF      everything spoken for the utterance vs the full-sentence translation (quality kept, 100 = same)
  extra     spoken words / full-sentence words (1.0 = no padding or repeats)
    python tools/replay_interpreter.py [scene ...] [--policies current,la2,la2c]
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultra_translate.core.interpreter import CJK_LANGS, _CJK, _CLAUSE, join, lcp, tokens  # noqa: E402

HUB = "http://127.0.0.1:8097"
WPS = 2.9          # VoxCPM2 English speaking rate (measured: 15 words in ~4.6 s incl. pauses)
TTFA = 0.10
P = "Translate the following segment into English, without additional explanation.\n\n"
PC = ("{ctx}\nReferring to the information above, translate the following text into English. "
      "Do not translate the context above and do not add any explanation:\n{text}")
_cache: dict = {}


def mt(text: str, prefix: str = "", ctx: list[str] | None = None) -> tuple[str, float]:
    key = (text, prefix, tuple(ctx or []))
    if key in _cache:
        return _cache[key]
    content = PC.format(ctx="\n".join(ctx), text=text) if ctx else P + text
    msgs = [{"role": "user", "content": content}]
    if prefix:
        msgs.append({"role": "assistant", "content": prefix})
    body = {"messages": msgs, "temperature": 0.0, "top_k": 1, "repeat_penalty": 1.05, "cache_prompt": True,
            "max_tokens": min(512, 64 + len(text) * 4), "hub_model": "best"}
    req = urllib.request.Request(HUB + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"].strip()
    dt = time.perf_counter() - t
    if prefix and not out.startswith(prefix.strip()[:10]):
        out = (prefix.rstrip() + " " + out).strip()      # some servers return only the continuation
    _cache[key] = (out, dt)
    return out, dt


def words(s: str) -> list[str]:
    return s.split()


def utterances(events):
    """id -> {"partials": [(t, text)], "final": (t, text), "lang"} for voice lines in a foreign language."""
    U = defaultdict(lambda: {"partials": [], "final": None, "lang": ""})
    for e in events:
        if e["dropped"]:
            continue
        u = U[e["id"]]
        if e["lang"]:
            u["lang"] = e["lang"]
        if e["final"]:
            if u["final"] is None and e["text"]:
                u["final"] = (e["t"], e["frag_text"] if e.get("frag") else e["text"], e.get("frag"))
        elif e["text"]:
            u["partials"].append((e["t"], e["text"]))
    out = {}
    for k, u in U.items():
        if u["final"] and u["lang"] and u["lang"] != "en":
            out[k] = u
    return out


# -- policies: each returns [(time_ready, english_text)] for one utterance -------------------------------------
def pol_normal(u):
    t, text, _ = u["final"]
    out, dt = mt(text)
    return [(t + dt, out)], out


def pol_current(u):
    """The shipped interpreter: clause/8-word source chunks translated alone (2 chunks of context)."""
    lang = u["lang"]
    prev, said, chunks, res = [], [], [], []
    clock = 0.0
    for t, text in u["partials"]:
        cur = tokens(text, lang)
        stable = min(lcp(prev, cur), len(cur) - 1)
        prev = cur
        cjk = lang in CJK_LANGS or bool(_CJK.search(text) and lang != "ko")
        start = len(said)
        mn = 8 if cjk else 4
        if stable - start < mn:
            continue
        cut = 0
        for i in range(stable - 1, start - 1, -1):
            if cur[i] and cur[i][-1] in _CLAUSE:
                cut = i + 1
                break
        if not cut and stable - start >= (14 if cjk else 8):
            cut = stable
        if not cut or cut - start < mn:
            continue
        chunk = join(cur[start:cut], lang, text)
        said = cur[:cut]
        out, dt = mt(chunk, ctx=chunks[-2:])
        chunks.append(chunk)
        clock = max(clock, t) + dt
        res.append((clock, out))
    t, text, _ = u["final"]
    final = tokens(text, lang)
    n = len(said)
    m = lcp(said, final)
    rest = final[m:] if m >= 0.6 * n else final[n:]
    if rest:
        out, dt = mt(join(rest, lang, text), ctx=chunks[-2:])
        clock = max(clock, t) + dt
        res.append((clock, out))
    full, _ = mt(text)
    return res, full


def _strip_end(ws: list[str]) -> list[str]:
    """A prefix translation ends the sentence early ('...spring time.'): never commit a final full stop."""
    if ws and ws[-1][-1:] in ".!?。":
        ws = ws[:-1] + [ws[-1].rstrip(".!?。")] if len(ws[-1]) > 1 else ws[:-1]
    return ws


FILLERS = re.compile(r"^(えー+|ええと|えっと|あのー*|うーん|まあ|um+|uh+|eh+|euh|ehm|嗯+|那个|呃+|음+|어+)[、,。\s]*")


def pol_la2(u, min_words=2, clause_bias=False, step_tokens=2, defill=False, hold=None, tail_hold=0):
    """LocalAgreement-2 on the translation (CUNI IWSLT 2025 / stable-prefix SimulMT):
    each time the stable source prefix grows, re-translate the WHOLE prefix with what was already spoken forced
    as the start of the answer; speak the new words that two consecutive hypotheses agree on."""
    lang = u["lang"]
    prev_src, last_src_n = [], 0
    spoken: list[str] = []
    prev_hyp: list[str] | None = None
    res, clock = [], 0.0
    for t, text in u["partials"]:
        cur = tokens(text, lang)
        stable = min(lcp(prev_src, cur), len(cur) - 1)
        prev_src = cur
        if stable <= 0 or stable - last_src_n < step_tokens:
            continue
        if t < clock:                          # translator still busy: this partial is stale, a newer one follows
            continue
        last_src_n = stable
        src = join(cur[:stable], lang, text)
        if defill:
            src = FILLERS.sub("", src)
        prefix = " ".join(spoken)
        out, dt = mt(src, prefix=prefix)
        clock = max(clock, t) + dt
        hyp = _strip_end(words(out)[len(spoken):])
        if hold is not None:                     # hold-n: commit all but the last n words of this hypothesis
            agree = hyp[:max(0, len(hyp) - hold)]
            if len(agree) >= min_words:
                spoken += agree
                res.append((clock, " ".join(agree)))
            prev_hyp = hyp[len(agree):] if len(agree) >= min_words else hyp
            continue
        if prev_hyp is not None:
            agree = []
            for a, b in zip(prev_hyp, hyp):
                if re.sub(r"\W", "", a.lower()) != re.sub(r"\W", "", b.lower()):
                    break
                agree.append(b)
            if tail_hold:                       # never speak a hypothesis' last words (closing padding lives there)
                agree = agree[:max(0, len(hyp) - tail_hold)]
            if clause_bias and len(agree) >= min_words:
                # prefer to stop at a phrase boundary (comma/conjunction) if one is inside the agreed part
                cut = max((i + 1 for i, w in enumerate(agree) if w[-1:] in ",;:" ), default=0)
                if cut >= min_words:
                    agree = agree[:cut]
            if len(agree) >= min_words:
                spoken += agree
                res.append((clock, " ".join(agree)))
                hyp = hyp[len(agree):]
        prev_hyp = hyp
    t, text, _ = u["final"]
    src = FILLERS.sub("", text) if defill else text
    out, dt = mt(src, prefix=" ".join(spoken))
    rest = words(out)[len(spoken):]
    clock = max(clock, t) + dt
    if rest:
        res.append((clock, " ".join(rest)))
    full, _ = mt(text)
    return res, full


POLICIES = {
    "normal": lambda u: pol_normal(u),
    "current": pol_current,
    "la2": lambda u: pol_la2(u, min_words=2),
    "la2_w3": lambda u: pol_la2(u, min_words=3),
    "la2_clause": lambda u: pol_la2(u, min_words=3, clause_bias=True),
    "la2_clause_defill": lambda u: pol_la2(u, min_words=3, clause_bias=True, defill=True),
    "la2_s1": lambda u: pol_la2(u, min_words=2, step_tokens=1),
    "la2_s1_w3": lambda u: pol_la2(u, min_words=3, step_tokens=1, defill=True),
    "la2_s1_w3_t1": lambda u: pol_la2(u, min_words=3, step_tokens=1, defill=True, tail_hold=1),
    "la2_s1_w3_t2": lambda u: pol_la2(u, min_words=3, step_tokens=1, defill=True, tail_hold=2),
    "la2_w3_t2": lambda u: pol_la2(u, min_words=3, step_tokens=2, defill=True, tail_hold=2),
    "la2_w3_t1": lambda u: pol_la2(u, min_words=3, step_tokens=2, defill=True, tail_hold=1),
    "hold2": lambda u: pol_la2(u, min_words=2, step_tokens=1, hold=2),
    "hold3": lambda u: pol_la2(u, min_words=3, step_tokens=1, hold=3),
    "hold4": lambda u: pol_la2(u, min_words=3, step_tokens=1, hold=4),
}


SUMMARY: dict = {}


def simulate(scene, names):
    events = json.loads((ROOT / "test_audio/traces" / f"{scene}.lines.json").read_text(encoding="utf-8"))
    U = utterances(events)
    rows = {n: [] for n in names}
    for n in names:
        speaker_free = 0.0
        hyps, refs = [], []
        for uid, u in sorted(U.items(), key=lambda kv: (kv[1]["partials"] or [(kv[1]["final"][0], "")])[0][0]):
            t_start = (u["partials"][0][0] - 0.6) if u["partials"] else u["final"][0] - 1.0
            t_end = u["final"][0] - 0.3
            pieces, full = (pol_normal(u) if n == "normal" else POLICIES[n](u))
            if n == "normal":
                pieces = pieces
            first = None
            for tr, txt in pieces:
                st = max(tr + TTFA, speaker_free)
                first = st if first is None else first
                speaker_free = st + len(words(txt)) / WPS
            spoken = " ".join(p for _, p in pieces)
            hyps.append(spoken)
            refs.append(full if n != "normal" else spoken)
            rows[n].append({"start_s": (first or t_end) - t_start, "behind_s": speaker_free - t_end,
                            "extra": len(words(spoken)) / max(1, len(words(full))), "pieces": len(pieces),
                            "spoken": spoken, "full": full})
        if n != "normal":
            ch = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        else:
            ch = 100.0
        r = rows[n]
        r_chrf = ch
        rows[n] = r
        SUMMARY.setdefault(n, []).append({"scene": scene, "chrF": ch,
                                          "start": sorted(x["start_s"] for x in r)[len(r) // 2],
                                          "behind": sorted(x["behind_s"] for x in r)[len(r) // 2],
                                          "extra": sum(x["extra"] for x in r) / len(r)})
        med = lambda k: sorted(x[k] for x in r)[len(r) // 2]
        print(json.dumps({"scene": scene, "policy": n, "utterances": len(r), "start_s_median": round(med("start_s"), 2),
                          "behind_s_median": round(med("behind_s"), 2), "chrF_vs_full": round(ch, 1),
                          "extra_words": round(sum(x["extra"] for x in r) / len(r), 2),
                          "pieces_per_utt": round(sum(x["pieces"] for x in r) / len(r), 2)}), flush=True)
    return rows


def main():
    scenes = [a for a in sys.argv[1:] if not a.startswith("--")] or [
        p.name.replace(".lines.json", "") for p in sorted((ROOT / "test_audio/traces").glob("*.lines.json"))]
    names = ["normal", "current", "la2", "la2_w3", "la2_clause_defill", "la2_s1", "la2_s1_w3"]
    for a in sys.argv:
        if a.startswith("--policies="):
            names = a.split("=", 1)[1].split(",")
    allrows = {}
    for sc in scenes:
        allrows[sc] = simulate(sc, names)
    print("# macro average over scenes (each conversation counts equally):")
    for n, rs in SUMMARY.items():
        k = len(rs)
        print(json.dumps({"policy": n, "scenes": k, "chrF": round(sum(x["chrF"] for x in rs) / k, 1),
                          "start_s": round(sum(x["start"] for x in rs) / k, 2),
                          "behind_s": round(sum(x["behind"] for x in rs) / k, 2),
                          "extra": round(sum(x["extra"] for x in rs) / k, 2)}))
    out = ROOT / "test_audio" / "interp_replay.json"
    out.write_text(json.dumps(allrows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
