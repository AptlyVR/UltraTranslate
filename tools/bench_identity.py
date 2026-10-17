"""Voiceprint identification bench on traced turns: enrol each true speaker from their first ~30 % of speech,
then identify every later turn (>= 1.5 s) against ALL enrolled speakers from ALL scenes (cross-scene impostors).
Reports correct-ID rate, false-ID rate (named the wrong person) and unknown rate at each threshold.
    python tools/bench_identity.py [ecapa|titanet|redimnet_b5|ecapa+redimnet_b5 ...]"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from score_speakers import ref_segments  # noqa: E402
TR = ROOT / "test_audio/traces"

def embs(scene, rows, model):
    parts = []
    for m in model.split("+"):
        E = np.array([r["emb"] for r in rows], np.float32) if m == "ecapa" else np.load(TR / f"{scene}.{m}.npy")
        parts.append(E / np.linalg.norm(E, axis=1, keepdims=True))
    E = np.concatenate(parts, 1)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def truth_of(t, ref):
    best, lab = 0, None
    for s, e, spk in ref:
        o = min(t[1], e) - max(t[0], s)
        if o > best:
            best, lab = o, spk
    return lab if best > 0.6 * (t[1] - t[0]) else None

for model in sys.argv[1:] or ["ecapa", "titanet", "redimnet_b5", "ecapa+redimnet_b5"]:
    enrol, tests = {}, []
    for f in sorted(TR.glob("*.json")):
        sc = f.stem
        if sc.startswith("05_"):
            continue
        rows = json.loads(f.read_text()); E = embs(sc, rows, model); ref = ref_segments(sc)
        lab = [truth_of(r["t"], ref) for r in rows]
        by = {}
        for i, (r, l) in enumerate(zip(rows, lab)):
            if l and not r["overlap"]:
                by.setdefault(l, []).append(i)
        for l, idx in by.items():
            tot = sum(rows[i]["dur"] for i in idx); acc = 0; k = 0
            while k < len(idx) and acc < max(6.0, 0.3 * tot):
                acc += rows[idx[k]]["dur"]; k += 1
            if k >= len(idx):
                continue
            key = f"{sc}|{l}"
            w = np.array([rows[i]["dur"] for i in idx[:k]])[:, None]
            v = (E[idx[:k]] * w).sum(0); enrol[key] = v / np.linalg.norm(v)
            tests += [(key, E[i]) for i in idx[k:] if rows[i]["dur"] >= 1.5]
    keys = list(enrol); M = np.stack([enrol[k] for k in keys])
    out = {"model": model, "enrolled": len(keys), "tests": len(tests)}
    for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        ok = wrong = unk = 0
        for key, e in tests:
            s = M @ e; j = int(np.argmax(s)); top = s[j]; second = np.partition(s, -2)[-2]
            if top < thr or top - second < 0.05:
                unk += 1
            elif keys[j] == key:
                ok += 1
            else:
                wrong += 1
        n = len(tests)
        out[f"thr{thr}"] = f"correct {ok / n:.1%} wrong {wrong / n:.1%} unknown {unk / n:.1%}"
    print(json.dumps(out))
