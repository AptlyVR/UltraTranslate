"""Speaker-embedding bake-off on real similar voices (and the labelled scenes).

Truth: community-1 diarization of a real 2-person recording (REAL_SCENE) + the labelled test scenes.
For each model: same-vs-different separation (EER, d'), 2-cluster accuracy on the video, speed per 1.5 s window.
    python tools/bake_embeddings.py [model ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "test_audio/scenes"
SR = 16000
REAL_SCENE = os.environ.get("REAL_SCENE", "real_conversation")   # a real recording + its .diar.json
os.environ.setdefault("HF_HUB_OFFLINE", "0")


def m_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    e = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=os.environ["APPDATA"] + "/UltraTranslate/models/ecapa",
                                       run_opts={"device": "cuda:0"})
    return lambda b: e.encode_batch(b.cuda()).squeeze(1)


def m_wespeaker34():
    from pyannote.audio import Model
    w = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM").eval().cuda()
    return lambda b: w(b[:, None].cuda())


def m_titanet():
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    m = m.eval().cuda()

    def f(b):
        b = b.cuda()
        lens = torch.full((b.shape[0],), b.shape[1], device="cuda")
        _, emb = m.forward(input_signal=b, input_signal_length=lens)
        return emb
    return f


def m_redimnet():
    m = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b5", train_type="ptn", dataset="vox2", trust_repo=True)
    m = m.eval().cuda()
    return lambda b: m(b.cuda())


def _wespeaker_hub(name):
    import types
    import torchaudio
    for sub in ("sox_effects", "backend"):
        sys.modules.setdefault(f"torchaudio.{sub}", types.ModuleType(f"torchaudio.{sub}"))
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *a, **k: None
    import wespeaker
    m = wespeaker.load_model(name)
    m.set_device("cuda:0")
    model = m.model.eval()

    def f(b):
        feats = torch.stack([m.compute_fbank(x[None], sample_rate=SR, cmn=True) for x in b]).cuda()
        out = model(feats)
        return out[-1] if isinstance(out, tuple) else out
    return f


def m_campplus():
    return _wespeaker_hub("campplus")


def m_resnet293():
    return _wespeaker_hub("english")


MODELS = {"ecapa": m_ecapa, "wespeaker34": m_wespeaker34, "titanet": m_titanet, "redimnet_b5": m_redimnet,
          "campplus": m_campplus, "wespeaker_r221": m_resnet293}


def segments():
    """(group, speaker, audio) for 1.5 s windows. Video truth from community-1 (exclusive), scenes from turns."""
    out = []
    wav, _ = sf.read(str(SC / f"{REAL_SCENE}.wav"), dtype="float32")
    for s in json.loads((SC / f"{REAL_SCENE}.diar.json").read_text()):
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        for st in range(a, b - int(1.5 * SR), int(1.5 * SR)):
            out.append(("video", s["spk"], wav[st:st + int(1.5 * SR)]))
    for sc in ("02_ja_en_two_people", "12_two_people_natural", "13_group_multilang_natural", "15_nine_lang_conversation",
               "10_en_meeting_real"):
        w, sr = sf.read(str(SC / f"{sc}.wav"), dtype="float32")
        w = w.mean(1) if w.ndim > 1 else w
        for t in json.loads((SC / f"{sc}.json").read_text(encoding="utf-8"))["turns"]:
            a, b = int(t["start"] * SR), int(t["end"] * SR)
            for st in range(a, b - int(1.5 * SR), int(1.5 * SR)):
                out.append((sc, t["speaker"], w[st:st + int(1.5 * SR)]))
    return out


def eer(same, diff):
    ths = np.linspace(-0.3, 1.0, 261)
    t = min(ths, key=lambda x: abs((same < x).mean() - (diff >= x).mean()))
    return float(((same < t).mean() + (diff >= t).mean()) / 2), float(t)


def evaluate(name, fn, segs):
    X, t_ms = [], []
    with torch.inference_mode():
        for i in range(0, len(segs), 32):
            b = torch.from_numpy(np.stack([s[2] for s in segs[i:i + 32]]))
            torch.cuda.synchronize()
            t = time.perf_counter()
            e = fn(b).float().cpu().numpy()
            torch.cuda.synchronize()
            t_ms.append((time.perf_counter() - t) * 1000 / len(b))
            X.append(e)
    X = np.concatenate(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    groups = np.array([s[0] for s in segs])
    labs = np.array([s[0] + "|" + s[1] for s in segs])
    res = {"model": name, "dim": int(X.shape[1]), "ms_per_window": round(float(np.median(t_ms)), 2)}
    for g in ("video", "scenes"):
        idx = np.where(groups == "video")[0] if g == "video" else np.where(groups != "video")[0]
        Xi, li, gi = X[idx], labs[idx], groups[idx]
        S = Xi @ Xi.T
        iu = np.triu_indices(len(idx), 1)
        within = (gi[:, None] == gi[None, :])[iu]       # only compare windows from the same recording
        same = (li[:, None] == li[None, :])[iu][within]
        sc = S[iu][within]
        e, thr = eer(sc[same], sc[~same])
        dprime = (sc[same].mean() - sc[~same].mean()) / np.sqrt(0.5 * (sc[same].var() + sc[~same].var()))
        res[g] = {"EER": round(e, 3), "thr": round(thr, 2), "dprime": round(float(dprime), 2),
                  "same_p50": round(float(np.median(sc[same])), 3), "diff_p99": round(float(np.percentile(sc[~same], 99)), 3)}
        if g == "video":
            from sklearn.cluster import AgglomerativeClustering
            pred = AgglomerativeClustering(n_clusters=2, metric="cosine", linkage="average").fit_predict(Xi)
            truth = (li == li[0]).astype(int)
            acc = max((pred == truth).mean(), (pred != truth).mean())
            res[g]["2cluster_acc"] = round(float(acc), 3)
    return res


def main():
    want = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    segs = segments()
    print(f"# {len(segs)} windows ({sum(s[0] == 'video' for s in segs)} from the real recording)", flush=True)
    for name in want:
        try:
            t = time.perf_counter()
            fn = MODELS[name]()
            load = time.perf_counter() - t
            r = evaluate(name, fn, segs)
            r["load_s"] = round(load, 1)
            r["vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
            print(json.dumps(r), flush=True)
        except Exception as exc:
            print(json.dumps({"model": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}), flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
