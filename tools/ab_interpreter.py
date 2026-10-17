"""A/B listening test: the same clip through Normal read-aloud and Interpreter mode, recorded as you'd hear it.

Feeds a scene at real-time pace through the embeddable session (same pipeline as the app), plays the translated
voice into a virtual "speaker" paced like a real device, and mixes it over the original (lowered to 30 %, like
the game-ducking feature). Writes test_audio/ab/<scene>_normal.wav and _interpreter.wav + timing summary.

    python tools/ab_interpreter.py [scene] [--target en]
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultra_translate.core.session import start_session  # noqa: E402

SR = 16000
REAL_SCENE = os.environ.get("REAL_SCENE", "real_conversation")   # a real recording + its .diar.json
OUT_SR = 48000


def run(scene: str, interpreter: bool, target: str) -> dict:
    audio, sr = sf.read(str(ROOT / "test_audio/scenes" / f"{scene}.wav"), dtype="float32")
    audio = audio.mean(1) if audio.ndim > 1 else audio
    if sr != SR:
        audio = soxr.resample(audio, sr, SR)
    truth = json.loads((ROOT / "test_audio/scenes" / f"{scene}.json").read_text(encoding="utf-8"))
    pieces: list[tuple[float, np.ndarray]] = []       # (time placed in the track, 48 kHz float audio)
    play_end = [0.0]
    t0 = [0.0]
    lock = threading.Lock()
    texts = []

    def on_audio(pcm16: bytes, rate: int, info: dict):
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if rate != OUT_SR:
            x = soxr.resample(x, rate, OUT_SR)
        now = time.monotonic() - t0[0]
        with lock:
            at = max(now, play_end[0])                  # a speaker plays one thing after another
            pieces.append((at, x, info.get("text", ""), bool(info.get("first"))))
            play_end[0] = at + len(x) / OUT_SR
        ahead = play_end[0] - (time.monotonic() - t0[0])
        if ahead > 0.25:                                # pace like a real device's buffer (no per-chunk sleeps:
            time.sleep(ahead - 0.2)                     # Windows sleeps >= 15 ms and that made gaps = crackle)

    def on_text(e):
        if e.get("final") and e.get("translation"):
            texts.append((round(time.monotonic() - t0[0], 2), e["language"], e["text"], e["translation"]))

    h = start_session(tgt_lang=target, on_text=on_text, on_audio=on_audio, voice_mode="own",
                      interpreter=interpreter)
    time.sleep(1.0)
    t0[0] = time.monotonic()
    step = 512
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    for i in range(0, len(pcm), step):
        target_t = t0[0] + i / SR
        d = target_t - time.monotonic()
        if d > 0:
            time.sleep(d)
        h.push_pcm16(pcm[i:i + step].tobytes(), SR)
    # trailing silence so the last line finishes and is spoken
    for _ in range(int(12 * SR / step)):
        h.push_pcm16(np.zeros(step, "<i2").tobytes(), SR)
        time.sleep(step / SR)
    h.stop()
    dur = max(len(audio) / SR, play_end[0]) + 0.5
    track = np.zeros(int(dur * OUT_SR), np.float32)
    orig48 = soxr.resample(audio, SR, OUT_SR) * 0.30
    track[:len(orig48)] += orig48
    firsts = []
    gaps = 0                                            # mid-sentence gaps (would be audible clicks)
    prev_end = None
    for at, x, text, first in pieces:
        if prev_end is not None and not first and at > prev_end + 0.002:
            gaps += 1
        prev_end = at + len(x) / OUT_SR
        k = int(at * OUT_SR)
        track[k:k + len(x)] += x[:len(track) - k]
        if first:
            firsts.append((round(at, 2), text))
    peak = float(np.abs(track).max()) or 1.0
    track = track / max(1.0, peak / 0.95)
    name = "interpreter" if interpreter else "normal"
    out = ROOT / "test_audio/ab" / f"{scene}_{name}.wav"
    out.parent.mkdir(exist_ok=True)
    sf.write(str(out), track, OUT_SR)
    # for each true sentence: when did its translated voice start, measured from when they started / stopped talking
    lags = []
    for tt in truth["turns"]:
        after = [a for a, _ in firsts if a >= tt["start"] - 0.2]
        if after:
            lags.append({"lang": tt["lang"], "start": tt["start"], "end": tt["end"],
                         "voice_from_start_s": round(after[0] - tt["start"], 2),
                         "voice_from_end_s": round(after[0] - tt["end"], 2)})
    return {"mode": name, "file": str(out), "spoken_pieces": len(firsts), "mid_sentence_gaps": gaps, "voice_starts": firsts,
            "per_sentence": lags, "finals": texts}


def main():
    scene = next((a for a in sys.argv[1:] if not a.startswith("--")), "16_interpreter_examples")
    target = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else "en"
    res = [run(scene, False, target), run(scene, True, target)]
    (ROOT / "test_audio/ab" / f"{scene}_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                                  encoding="utf-8")
    for r in res:
        ps = r["per_sentence"]
        print(json.dumps({"mode": r["mode"], "pieces": r["spoken_pieces"], "gaps": r["mid_sentence_gaps"],
                          "voice_from_start_s": [p["voice_from_start_s"] for p in ps],
                          "voice_from_end_s": [p["voice_from_end_s"] for p in ps]}, ensure_ascii=False))
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
