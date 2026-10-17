"""Score online speaker labels against a reference diarization (time overlap).
    python tools/score_speakers.py results.json <scene> [--ref diar|turns]
Speaker error = share of line-time whose label doesn't map to its true speaker under the best one-to-one mapping
(the 'confusion' part of DER, on the lines the app produced)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]

def ref_segments(scene):
    d = ROOT / "test_audio/scenes" / f"{scene}.diar.json"
    if d.exists():
        return [(x["start"], x["end"], x["spk"]) for x in json.loads(d.read_text())]
    T = json.loads((ROOT / "test_audio/scenes" / f"{scene}.json").read_text(encoding="utf-8"))["turns"]
    return [(t["start"], t["end"], t["speaker"]) for t in T]


def score(lines, ref):
    ov = defaultdict(lambda: defaultdict(float))
    for a, b, lab in lines:
        for s, e, spk in ref:
            o = min(b, e) - max(a, s)
            if o > 0:
                ov[lab][spk] += o
    labs = sorted(ov); spks = sorted({s for _, _, s in ref})
    M = np.array([[ov[l][s] for s in spks] for l in labs]) if labs else np.zeros((0, len(spks)))
    total = M.sum()
    r, c = linear_sum_assignment(-M) if M.size else ([], [])
    correct = M[r, c].sum() if M.size else 0
    purity = sum(M[i].max() for i in range(len(labs))) / total if total else 0
    cover = {spks[j]: round(float(M[:, j].max() / max(M[:, j].sum(), 1e-9)), 3) for j in range(len(spks))}
    return {"labels": len(labs), "true_speakers": len(spks), "speaker_error": round(1 - correct / total, 3) if total else None,
            "purity": round(float(purity), 3), "coverage": cover,
            "label_seconds": {l: round(float(M[i].sum()), 1) for i, l in enumerate(labs)}}

def main():
    res = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    tot_err = tot_w = 0.0
    for r in res:
        scene = r["summary"]["scene"]
        L = [(l["t"][0], l["t"][1], l["spk"]) for l in r["summary"]["all_lines"] if not l["dropped"] and l["spk"]]
        sc = score(L, ref_segments(scene))
        w = sum(sc["label_seconds"].values())
        tot_err += (sc["speaker_error"] or 0) * w
        tot_w += w
        print(f"{scene:28s} true {sc['true_speakers']:2d} labels {sc['labels']:2d}  speaker_error {sc['speaker_error']:.3f}  purity {sc['purity']:.3f}")
    print(f"{'ALL (time-weighted)':28s} speaker_error {tot_err / max(tot_w, 1e-9):.3f}")


if __name__ == "__main__":
    main()
