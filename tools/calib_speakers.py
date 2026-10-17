"""Compare speaker-embedding models on labelled scenes: same- vs different-speaker cosine, EER, thresholds.
    python tools/calib_speakers.py"""
import json, os, sys, numpy as np, soundfile as sf, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "test_audio/scenes"
SCENES = ["02_ja_en_two_people", "03_multilang_party", "05_ami_meeting", "07_two_reverb_hiss", "10_en_meeting_real",
          "12_two_people_natural", "13_group_multilang_natural", "14_group_natural_noisy", "15_nine_lang_conversation"]

def load_models():
    from pyannote.audio import Model
    w = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM").eval().cuda()
    from speechbrain.inference.speaker import EncoderClassifier
    e = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=os.environ["APPDATA"] + "/UltraTranslate/models/ecapa", run_opts={"device": "cuda:0"})
    def wes(x):
        with torch.inference_mode():
            return w(torch.from_numpy(x)[None, None].cuda()).squeeze().float().cpu().numpy()
    def eca(x):
        with torch.inference_mode():
            return e.encode_batch(torch.from_numpy(x)[None].cuda()).squeeze().float().cpu().numpy()
    return {"wespeaker": wes, "ecapa": eca}

def eer(same, diff):
    ths = np.linspace(-0.2, 0.9, 221)
    best = min(ths, key=lambda t: abs((same < t).mean() - (diff >= t).mean()))
    return float((same < best).mean()), float(best)

def main():
    M = load_models()
    for dur_mode in ("turn", "1.5s"):
        res = {k: {"same": [], "diff": []} for k in M}
        for sc in SCENES:
            wav, sr = sf.read(str(SC / f"{sc}.wav"), dtype="float32")
            wav = wav.mean(1) if wav.ndim > 1 else wav
            T = json.loads((SC / f"{sc}.json").read_text(encoding="utf-8"))["turns"]
            segs = []
            for t in T:
                a, b = int(t["start"] * sr), int(t["end"] * sr)
                if dur_mode == "1.5s":
                    if b - a < int(1.5 * sr):
                        continue
                    m = (a + b) // 2
                    a, b = m - int(0.75 * sr), m + int(0.75 * sr)
                elif b - a < sr:
                    continue
                segs.append((t["speaker"], wav[a:b]))
            for k, f in M.items():
                X = np.stack([f(x) for _, x in segs])
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                lab = np.array([s for s, _ in segs])
                S = X @ X.T
                iu = np.triu_indices(len(segs), 1)
                same = (lab[:, None] == lab[None, :])[iu]
                res[k]["same"] += list(S[iu][same]); res[k]["diff"] += list(S[iu][~same])
        for k, r in res.items():
            s, d = np.array(r["same"]), np.array(r["diff"])
            e, t = eer(s, d)
            print(json.dumps({"segments": dur_mode, "model": k, "pairs_same": len(s), "pairs_diff": len(d),
                              "same_p10": round(float(np.percentile(s, 10)), 3), "same_p50": round(float(np.median(s)), 3),
                              "diff_p50": round(float(np.median(d)), 3), "diff_p90": round(float(np.percentile(d, 90)), 3),
                              "diff_p99": round(float(np.percentile(d, 99)), 3), "EER": round(e, 3), "eer_threshold": round(t, 3)}))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
