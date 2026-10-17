"""Run test scenes through the real pipeline (same code the app uses) and score them.

Audio is fed at real-time pace by default (so latency numbers are honest); --fast feeds it as fast as
possible. Scores: character error rate per turn (CJK) / word error rate (others), language accuracy,
speaker purity/coverage (how consistently each true speaker maps to one label), and latency.

    python tools/eval_scenes.py [scene names...] [--fast] [--no-translate]
"""
from __future__ import annotations

import argparse
import numpy as np
import json
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultra_translate.core.pipeline import Pipeline  # noqa: E402
from ultra_translate.settings import Settings  # noqa: E402

SCENES = ROOT / "test_audio" / "scenes"
CJK = {"ja", "zh", "ko", "yue", "th"}


def norm(t: str, lang: str) -> list[str]:
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[^\w\s]", "", t)
    if lang in CJK:
        return list(re.sub(r"\s+", "", t))
    return t.split()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def run_scene(name: str, fast: bool, translate: bool, settings: Settings, pipe_cache: dict):
    truth = json.loads((SCENES / f"{name}.json").read_text(encoding="utf-8"))
    lines: dict[str, object] = {}
    statuses = []
    done = threading.Event()
    s = settings
    s.audio.source = "file"
    s.audio.file_path = str(SCENES / f"{name}.wav")
    s.audio.monitor_file = False
    s.translation.enabled = translate
    if "pipe" not in pipe_cache:
        pipe = Pipeline(s, lambda ln: lines.__setitem__(ln.id, ln), lambda: None, statuses.append)
        pipe.load_models()
        pipe_cache["pipe"] = pipe
    pipe = pipe_cache["pipe"]
    seen_final: dict[str, float] = {}
    seen_mt: dict[str, float] = {}

    events = []

    def on_line(ln):
        now = time.perf_counter()
        if "--trace-lines" in sys.argv:
            events.append({"t": now, "id": ln.id, "final": ln.final, "text": ln.text, "lang": ln.language,
                           "dropped": ln.dropped, "spk": ln.speaker_id,
                           "frag": ln.timings.get("fragment_id") if ln.timings.get("merged") else None,
                           "frag_text": ln.timings.get("fragment_text") if ln.timings.get("merged") else None})
        if ln.final and ln.id not in seen_final:
            seen_final[ln.id] = now
        if ln.translation and ln.id not in seen_mt:
            seen_mt[ln.id] = now
        lines[ln.id] = ln
    pipe.on_line = on_line
    pipe.on_status = statuses.append
    pipe.spk.reset_session()
    pipe.spk.trace = [] if "--trace" in sys.argv else None
    pipe.stats["spec_hits"] = 0
    pipe.levels.clear()
    pipe.session_lang = ""
    pipe.lang_seconds = {}
    if fast:
        import ultra_translate.core.audio as A
        A._FAST = True
    t0 = time.perf_counter()
    pipe.start()
    t_audio0 = time.perf_counter()
    while pipe.source.is_alive():
        time.sleep(0.1)
    # wait for queues to drain
    deadline = time.time() + 60
    time.sleep(0.5)
    while time.time() < deadline and (pipe.frames.qsize() or not pipe.idle()):
        time.sleep(0.05)
    pipe.stop()
    if "--trace-lines" in sys.argv:
        tdir = ROOT / "test_audio" / "traces"
        tdir.mkdir(exist_ok=True)
        for e in events:
            e["t"] = round(e["t"] - t_audio0, 3)
        (tdir / f"{name}.lines.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    wall = time.perf_counter() - t0
    finals = sorted([l for l in lines.values() if l.final and not l.dropped and l.text], key=lambda l: l.t_start)
    for l in finals:                       # labels merged later (same voice) count as the label they became
        l.speaker_id = pipe.spk.resolve(l.speaker_id) if l.speaker_id else l.speaker_id
    # match hypotheses to truth turns by time overlap
    turns = truth["turns"]
    assigned = defaultdict(list)
    for l in finals:
        best, ov = None, 0.0
        for k, t in enumerate(turns):
            o = min(l.t_end, t["end"]) - max(l.t_start, t["start"])
            if o > ov:
                best, ov = k, o
        if best is not None:
            assigned[best].append(l)
    rows = []
    errs = refs = 0
    lang_ok = lang_n = 0
    spk_map = defaultdict(Counter)
    for k, t in enumerate(turns):
        hyp = assigned.get(k, [])
        joiner = "" if t["lang"] in CJK else " "
        htext = joiner.join(h.text for h in hyp)
        r, h = norm(t["text"], t["lang"]), norm(htext, t["lang"])
        e = edit_distance(r, h)
        errs += e
        refs += len(r)
        hl = Counter(x.language for x in hyp).most_common(1)[0][0] if hyp else ""
        lang_ok += int(hl == t["lang"])
        lang_n += 1
        for x in hyp:
            spk_map[t["speaker"]][x.speaker_id] += x.t_end - x.t_start
        rows.append({"turn": k, "speaker": t["speaker"], "lang": t["lang"], "hyp_lang": hl,
                     "hyp_speakers": sorted({x.speaker_id for x in hyp}), "err_rate": round(e / max(1, len(r)), 3),
                     "ref": t["text"], "hyp": htext, "translation": " ".join(x.translation for x in hyp if x.translation),
                     "asr_ms": [x.timings.get("asr") for x in hyp], "mt_ms": [x.timings.get("mt") for x in hyp]})
    # speaker purity: share of each true speaker's time on its dominant label; label reuse across true speakers
    purity = {}
    label_owner = defaultdict(Counter)
    for spk, c in spk_map.items():
        tot = sum(c.values())
        lab, sec = c.most_common(1)[0] if c else ("", 0)
        purity[spk] = round(sec / tot, 2) if tot else 0.0
        for lab2, sec2 in c.items():
            label_owner[lab2][spk] += sec2
    confused = {lab: dict(v) for lab, v in label_owner.items() if len(v) > 1}
    lat = [l.timings.get("asr", 0) + l.timings.get("spk", 0) + l.timings.get("diar", 0) for l in finals]
    # alignment-free: whole scene, tokens in time order (CJK chars, others words)
    ref_tok = [tok for t in sorted(turns, key=lambda t: t["start"]) for tok in norm(t["text"], t["lang"])]
    hyp_tok = [tok for l in finals for tok in norm(l.text, l.language or "en")]
    scene_err = round(edit_distance(ref_tok, hyp_tok) / max(1, len(ref_tok)), 3)
    # real-time latency: speech end (truth) -> text shown / translation shown
    lat_txt, lat_mt = [], []
    if not fast:
        for k, t in enumerate(turns):
            hyp = assigned.get(k, [])
            if not hyp:
                continue
            last = max(hyp, key=lambda x: x.t_end)
            end_wall = t_audio0 + t["end"]
            if last.id in seen_final:
                lat_txt.append((seen_final[last.id] - end_wall) * 1000)
            if last.id in seen_mt:
                lat_mt.append((seen_mt[last.id] - end_wall) * 1000)
    pct = lambda v, q: round(float(np.percentile(v, q))) if v else None
    summary = {"scene": name, "scene_error": scene_err,
               "speech_end_to_text_ms": {"p50": pct(lat_txt, 50), "p90": pct(lat_txt, 90)},
               "speech_end_to_translation_ms": {"p50": pct(lat_mt, 50), "p90": pct(lat_mt, 90)},
               "spec_hits": pipe.stats.get("spec_hits", 0), "early_commits": pipe.segmenter.stats["early"], "audio_s": round(truth["duration_s"], 1), "wall_s": round(wall, 1),
               "error_rate": round(errs / max(1, refs), 3), "language_acc": f"{lang_ok}/{lang_n}",
               "true_speakers": len(spk_map), "labels_used": len({l.speaker_id for l in finals}),
               "speaker_purity": purity, "labels_shared_by_speakers": confused,
               "lines": len(finals), "proc_ms_median": sorted(lat)[len(lat) // 2] if lat else None,
               "proc_ms_max": max(lat) if lat else None,
               "errors": [m for m in statuses if "error" in m.lower() or "failed" in m.lower()][:5]}
    ids = list(pipe.spk.centroids)
    summary["centroid_sims"] = {f"{x}-{y}": round(float(pipe.spk.centroids[x] @ pipe.spk.centroids[y]), 3)
                                for i, x in enumerate(ids) for y in ids[i + 1:]}
    if pipe.spk.trace is not None:
        tdir = ROOT / "test_audio" / "traces"
        tdir.mkdir(exist_ok=True)
        (tdir / f"{name}.json").write_text(json.dumps([x for x in pipe.spk.trace if "t" in x]))
    summary["centroid_weights"] = {x: round(pipe.spk.weights.get(x, 0), 1) for x in ids}
    summary["all_lines"] = [{"t": [round(l.t_start, 2), round(l.t_end, 2)], "spk": l.speaker_id, "lang": l.language,
                             "text": l.text, "dropped": l.dropped, "ms": l.timings}
                            for l in sorted(lines.values(), key=lambda l: l.t_start) if l.final]
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-lines", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "test_audio" / "results.json"))
    args = ap.parse_args()
    names = args.scenes or sorted(p.stem for p in SCENES.glob("*.wav"))
    s = Settings()
    cache = {}
    results = []
    for n in names:
        summ, rows = run_scene(n, args.fast, not args.no_translate, s, cache)
        results.append({"summary": summ, "turns": rows})
        print(json.dumps({k: v for k, v in summ.items() if k != "all_lines"}, ensure_ascii=False))
        if args.verbose:
            for t in json.loads((SCENES / f"{n}.json").read_text(encoding="utf-8"))["turns"]:
                print(f"   TRUTH {t['start']:6.2f}-{t['end']:6.2f} {t['speaker']}/{t['lang']}: {t['text'][:80]}")
            for l in summ["all_lines"]:
                print(f"   HYP   {l['t'][0]:6.2f}-{l['t'][1]:6.2f} {l['spk']}/{l['lang']} {l['dropped'] or ''}: {l['text'][:80]}")
        for r in rows:
            print(f"   [{r['speaker']}/{r['lang']}->{r['hyp_lang']} {','.join(r['hyp_speakers'])} "
                  f"err={r['err_rate']}] {r['hyp'][:90]}" + (f"  => {r['translation'][:90]}" if r['translation'] else ""))
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
