"""Score the chatbox parser offline on cached RapidOCR lines (no OCR/GPU).  python tools/tune_chatbox.py [-v]"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultra_translate" / "screen"))
sys.path.insert(0, str(ROOT / "tools"))
import importlib  # noqa: E402
import chatbox  # noqa: E402
importlib.reload(chatbox)
from score_ocr import cer  # noqa: E402

L = json.loads((ROOT / "test_audio/ocr_bench/lines.json").read_text(encoding="utf-8"))
T = {t["image"]: t for t in json.loads((ROOT / "test_audio/ocr_bench/truth.json").read_text(encoding="utf-8"))}
n = found = exact = names = fp = 0
errs, ms = [], []
for im, lines in L.items():
    if im not in T:
        continue
    img = cv2.imread(str(ROOT / "test_audio/ocr_bench/images" / im))
    t0 = time.perf_counter()
    if "--live" in sys.argv and not T[im].get("real") or "--synthetic" in sys.argv and T[im].get("real"):
        continue
    players = (T[im].get("players") or [b["name"] for b in T[im]["bubbles"]] + ["3-banana-3", "Płυм․", "ninety-"])         if "--no-players" not in sys.argv else None
    preds = chatbox.find_chatboxes([(l["text"], l["score"], np.array(l["box"])) for l in lines], img, players)
    ms.append((time.perf_counter() - t0) * 1000)
    used = set()
    for b in T[im]["bubbles"]:
        n += 1
        best, bi = 9, None
        for i, p in enumerate(preds):
            if i not in used and cer(b["text"], p["text"]) < best:
                best, bi = cer(b["text"], p["text"]), i
        if bi is not None and best < 0.5:
            used.add(bi)
            found += 1
            exact += best == 0
            errs.append(best)
            ok = cer(b["name"], preds[bi]["name"]) <= 0.15 or ("--no-players" in sys.argv and not preds[bi]["name"])
            names += ok
            if "-v" in sys.argv and not ok:
                print(f"  name miss {im}: want {b['name']!r} got {preds[bi]['name']!r}")
        else:
            errs.append(1.0)
            if "-v" in sys.argv:
                print(f"  bubble miss {im} scale={b['scale']}: {b['text'][:30]!r}")
    fp += len(preds) - len(used)
    if "-v" in sys.argv and len(preds) > len(used):
        print(f"  false in {im}:", [p["text"][:25] for i, p in enumerate(preds) if i not in used])
print(f"found {found}/{n}  exact {exact}  CER {sum(errs)/len(errs):.3f}  names {names}/{n}  false {fp}  parser {np.median(ms):.1f} ms")
