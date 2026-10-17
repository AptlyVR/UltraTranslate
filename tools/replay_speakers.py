"""Deterministic replay of the real-time speaker assignment on recorded traces (tools/eval_scenes.py --trace).

Each trace row is one assign() call from a real pipeline run: turn time range, duration, overlap flag, embedding.
Replaying lets us try clustering settings, or a different voice model (re-embedding the same turns), in seconds.
    python tools/replay_speakers.py                      # current settings, ECAPA from the trace
    python tools/replay_speakers.py --grid               # parameter sweep
    python tools/replay_speakers.py --model titanet      # re-embed turns with another model (one model per process)
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from score_speakers import ref_segments, score  # noqa: E402
from ultra_translate.core.speakers import SpeakerEngine  # noqa: E402
from ultra_translate.settings import Settings  # noqa: E402

TR = ROOT / "test_audio" / "traces"
SR = 16000
REAL_SCENE = os.environ.get("REAL_SCENE", "real_conversation")   # a real recording + its .diar.json


def load_traces(model: str | None):
    out = {}
    for f in sorted(TR.glob("*.json")):
        rows = json.loads(f.read_text())
        if model and "+" in model:          # fused signature: concatenation of unit embeddings from cached models
            parts = []
            for m in model.split("+"):
                if m == "ecapa":
                    E = np.array([r["emb"] for r in rows], np.float32)
                else:
                    E = np.load(TR / f"{f.stem}.{m}.npy")
                parts.append(E / np.linalg.norm(E, axis=1, keepdims=True))
            E = np.concatenate(parts, axis=1)
            for r, e in zip(rows, E):
                r["emb"] = (e / np.linalg.norm(e)).tolist()
            out[f.stem] = rows
            continue
        if model:
            cache = TR / f"{f.stem}.{model}.npy"
            if cache.exists():
                E = np.load(cache)
            else:
                import soundfile as sf
                sys.path.insert(0, str(ROOT / "tools"))
                import bake_embeddings as B
                import torch
                fn = B.MODELS[model]() if not hasattr(load_traces, "fn") else load_traces.fn
                load_traces.fn = fn
                wav, _ = sf.read(str(ROOT / "test_audio/scenes" / f"{f.stem}.wav"), dtype="float32")
                wav = wav.mean(1) if wav.ndim > 1 else wav
                E = []
                with torch.inference_mode():
                    for r in rows:
                        a, b = int(r["t"][0] * SR), int(r["t"][1] * SR)
                        x = wav[a:max(b, a + SR // 2)]
                        E.append(fn(torch.from_numpy(np.ascontiguousarray(x))[None]).float().cpu().numpy().reshape(-1))
                E = np.stack(E)
                np.save(cache, E)
            for r, e in zip(rows, E):
                r["emb"] = (e / np.linalg.norm(e)).tolist()
        out[f.stem] = rows
    return out


def replay(rows, cfg):
    eng = SpeakerEngine(cfg, log=lambda m: None)
    eng.store.items = {}
    eng._identify = lambda sp: None
    labels = []
    for r in rows:
        sp, _ = eng.assign(np.zeros(int(r["dur"] * SR), np.float32), r["overlap"],
                           emb=np.asarray(r["emb"], np.float32))
        labels.append(sp.id)
    return [(r["t"][0], r["t"][1], eng.resolve(l)) for r, l in zip(rows, labels)]


def run(traces, cfg, verbose=False):
    tot_e = tot_w = extra = 0.0
    per = {}
    for scene, rows in traces.items():
        sc = score(replay(rows, cfg), ref_segments(scene))
        w = sum(sc["label_seconds"].values())
        tot_e += sc["speaker_error"] * w
        tot_w += w
        extra += max(0, sc["labels"] - sc["true_speakers"])
        per[scene] = (sc["true_speakers"], sc["labels"], sc["speaker_error"])
        if verbose:
            print(f"  {scene:28s} true {sc['true_speakers']:2d} labels {sc['labels']:2d} err {sc['speaker_error']:.3f}")
    return tot_e / tot_w, extra, per


def main():
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
    traces = load_traces(model)
    base = Settings().speakers
    err, extra, _ = run(traces, base, verbose=True)
    print(f"{model or 'ecapa(trace)'} current: speaker_error {err:.3f}, extra labels {extra:.0f}")
    if "--grid" in sys.argv:
        res = []
        for st, nm, mt in itertools.product([0.24, 0.28, 0.32, 0.36, 0.40], [1.5, 2.0, 2.5, 3.0], [0.35, 0.40, 0.45, 0.50]):
            cfg = copy.deepcopy(base)
            cfg.session_threshold, cfg.new_min_s, cfg.merge_threshold = st, nm, mt
            e, x, _ = run(traces, cfg)
            res.append((e + 0.004 * x, e, x, st, nm, mt))
        res.sort()
        for r in res[:8]:
            print(f"err {r[1]:.3f} extra {r[2]:.0f}  session_threshold {r[3]} new_min_s {r[4]} merge_threshold {r[5]}")
        best = res[0]
        cfg = copy.deepcopy(base)
        cfg.session_threshold, cfg.new_min_s, cfg.merge_threshold = best[3], best[4], best[5]
        print("best per scene:")
        run(traces, cfg, verbose=True)


if __name__ == "__main__":
    main()
