"""Score chatbox OCR results against truth.  python tools/score_ocr.py results1.json results2.json ..."""
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_scenes import edit_distance  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "test_audio" / "ocr_bench" / "truth.json").read_text(encoding="utf-8"))


def norm(t):
    return list(re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", t).lower()))


def cer(ref, hyp):
    r = norm(ref)
    return edit_distance(r, norm(hyp)) / max(1, len(r))


def score(res, only=None):
    n = found = exact = names_ok = fp = 0
    errs, ms = [], []
    by_lang = {}
    for t in TRUTH:
        if only == "live" and not t.get("real"):
            continue
        if only == "synthetic" and t.get("real"):
            continue
        r = res["images"].get(t["image"])
        if r is None:
            continue
        ms.append(r["ms"])
        preds = list(r.get("bubbles", []))
        used = set()
        for b in t["bubbles"]:
            n += 1
            best, bi = 9.0, None
            for i, p in enumerate(preds):
                if i in used:
                    continue
                c = cer(b["text"], p.get("text", ""))
                if c < best:
                    best, bi = c, i
            ok = bi is not None and best < 0.5
            if ok:
                used.add(bi)
                found += 1
                exact += best == 0
                errs.append(best)
                names_ok += cer(b["name"], preds[bi].get("name", "")) <= 0.15
            else:
                errs.append(1.0)
            by_lang.setdefault(b["lang"], []).append(best if ok else 1.0)
        fp += len(preds) - len(used)
    ms.sort()
    return {"engine": res["engine"], "load_s": res["load_s"], "vram_mib": res["vram_mib"],
            "ms_p50": ms[len(ms) // 2], "ms_max": ms[-1], "bubbles": n, "found": found, "exact": exact,
            "cer": round(sum(errs) / max(1, len(errs)), 3), "names_ok": names_ok, "false_bubbles": fp,
            "cer_by_lang": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_lang.items())}}


if __name__ == "__main__":
    only = next((a[7:] for a in sys.argv if a.startswith("--only=")), None)
    rows = [score(json.loads(Path(p).read_text(encoding="utf-8")), only) for p in sys.argv[1:] if not p.startswith("--")]
    print(f"{'engine':40s} {'load':>6s} {'VRAM':>7s} {'ms p50':>7s} {'ms max':>7s} {'found':>7s} {'exact':>6s} {'CER':>6s} {'names':>7s} {'false':>6s}")
    for r in rows:
        print(f"{r['engine']:40s} {r['load_s']:5.1f}s {r['vram_mib']:5d}MB {r['ms_p50']:6d} {r['ms_max']:7d} "
              f"{r['found']:3d}/{r['bubbles']:<3d} {r['exact']:5d} {r['cer']:6.3f} {r['names_ok']:3d}/{r['bubbles']:<3d} {r['false_bubbles']:5d}")
    for r in rows:
        print("  ", r["engine"], "per-language CER:", r["cer_by_lang"])
