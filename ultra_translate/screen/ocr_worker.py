"""Isolated OCR worker (runs in the OCR venv, not the speech process). JSON lines on stdin/stdout.

  {"cmd":"load","device":"cuda"}                       -> {"ok":true,"ms":..}
  {"cmd":"snap","window":"VRChat","crop":1.0,"source":"auto","only_if_changed":false}
                                                      -> {"ok":true,"blocks":[..],"ms":..,"src":..,"changed":true}
  {"cmd":"quit"}
Killing the process frees all OCR memory (that is what "Unload" does).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

_last_thumb = None


def thumb(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)


def main():
    global _last_thumb
    engine = None
    out = sys.stdout
    for raw in sys.stdin:
        try:
            req = json.loads(raw)
        except ValueError:
            continue
        cmd = req.get("cmd")
        t0 = time.perf_counter()
        try:
            if cmd == "quit":
                break
            if cmd == "load":
                from ocr import OcrEngine
                engine = OcrEngine(log=lambda m: None, device=req.get("device", "cuda"))
                engine.load()
                resp = {"ok": True}
            elif cmd == "snap":
                from capture import grab
                img, src = grab(req.get("window", "VRChat"), float(req.get("crop", 1.0)))
                th = thumb(img)
                changed = _last_thumb is None or _last_thumb.shape != th.shape or \
                    float(np.mean(np.abs(th - _last_thumb))) > float(req.get("change_threshold", 2.5))
                if req.get("only_if_changed") and not changed:
                    resp = {"ok": True, "changed": False, "blocks": [], "src": src}
                else:
                    _last_thumb = th
                    if engine is None:
                        from ocr import OcrEngine
                        engine = OcrEngine(log=lambda m: None, device=req.get("device", "cuda"))
                    from ocr import group_paragraphs
                    lines = engine.read(img, source=req.get("source", "auto"))
                    resp = {"ok": True, "changed": True, "src": src, "size": [int(img.shape[1]), int(img.shape[0])]}
                    if req.get("mode") == "chatbox":
                        from chatbox import find_chatboxes
                        resp["chatboxes"] = find_chatboxes([(l.text, l.score, l.box) for l in lines], img,
                                                           req.get("players") or None)
                        resp["blocks"] = []
                    else:
                        resp["blocks"] = group_paragraphs(lines)
            else:
                resp = {"ok": False, "error": f"unknown cmd {cmd}"}
        except Exception as exc:
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        resp["ms"] = round((time.perf_counter() - t0) * 1000)
        out.write(json.dumps(resp, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
