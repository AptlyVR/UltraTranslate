"""OCR: RapidOCR (PaddleOCR ONNX). Text detection on CUDA, recognition on CPU.

Primary pass: PP-OCRv6 multilingual (Chinese, Japanese, English, Latin scripts).
Fallback pass (Auto mode): lines the primary model reads poorly are re-read with
PP-OCRv5 Korean and East-Slavic (Cyrillic) recognizers; best score wins.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

try:  # load pip-installed CUDA/cuDNN DLLs (nvidia-* wheels) into this process
    ort.preload_dlls()
except Exception:
    pass

from rapidocr import RapidOCR  # noqa: E402
from rapidocr.utils.typings import LangRec, ModelType, OCRVersion  # noqa: E402

# Detection runs on CUDA at ONE fixed canvas size (onnxruntime+cuDNN on Blackwell
# stalls ~4 s for every new input shape, so we never give it a new one).
# Recognition runs on CPU (2 threads): tiny models, any crop width, no stalls.
DET_CANVAS = (1600, 896)  # w, h — multiples of 32
REC_THREADS = 2
_BASE = {"Global.log_level": "error"}
_DET_GPU = {
    **_BASE,
    "Global.use_det": True, "Global.use_cls": False, "Global.use_rec": False,
    "Global.use_preprocess_img": False,
    "Det.limit_side_len": DET_CANVAS[1], "Det.limit_type": "min",
    "EngineConfig.onnxruntime.use_cuda": True,
    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
    "EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": "kSameAsRequested",
}
_REC_CPU = {
    **_BASE,
    "Global.use_det": False, "Global.use_cls": False, "Global.use_rec": True,
    "EngineConfig.onnxruntime.intra_op_num_threads": REC_THREADS,
    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
}

# source profiles: (lang_type, ocr_version, model_type)
FALLBACK_RECS = {
    "korean": (LangRec.KOREAN, OCRVersion.PPOCRV5, ModelType.MOBILE),
    "cyrillic": (LangRec.ESLAV, OCRVersion.PPOCRV5, ModelType.MOBILE),
}
SOURCES = ["auto", "cjk_latin", "korean", "cyrillic"]
FALLBACK_SCORE = 0.80


@dataclass
class OcrLine:
    text: str
    score: float
    box: np.ndarray  # 4x2 points
    engine: str = "multi"


def _rec_engine(lang=None, ver=None, mtype=None):
    p = dict(_REC_CPU)
    if lang is not None:
        p.update({"Rec.lang_type": lang, "Rec.ocr_version": ver, "Rec.model_type": mtype})
    return RapidOCR(params=p)


def _letterbox(img: np.ndarray) -> tuple[np.ndarray, float]:
    cw, ch = DET_CANVAS
    h, w = img.shape[:2]
    scale = min(cw / w, ch / h)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas, scale


def _crop(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    pts = box.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    w, h = max(w, 4), max(h, 4)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h * 1.0 / w >= 1.5:
        out = np.rot90(out)
    return out


def _det_params(device: str) -> dict:
    if device == "cuda":
        return _DET_GPU
    p = {k: v for k, v in _DET_GPU.items() if not k.startswith("EngineConfig.onnxruntime.cuda")}
    p["EngineConfig.onnxruntime.use_cuda"] = False
    p["EngineConfig.onnxruntime.intra_op_num_threads"] = 4
    p["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
    return p


class OcrEngine:
    """device: "cuda" (det on GPU, ~0.5 GB VRAM) or "cpu" (no VRAM at all)."""

    def __init__(self, log=print, device: str = "cuda"):
        self._log = log
        self.device = device
        self._lock = threading.Lock()
        self._det = None
        self._rec = None
        self._fallback: dict[str, RapidOCR] = {}

    @property
    def loaded(self) -> bool:
        return self._det is not None

    def load(self, warm: bool = True):
        with self._lock:
            if self._det is not None:
                return
            t = time.perf_counter()
            self._det = RapidOCR(params=_det_params(self.device))
            self._rec = _rec_engine()
            for name, spec in FALLBACK_RECS.items():
                self._fallback[name] = _rec_engine(*spec)
            if warm:  # the single det shape pays its cuDNN setup here, not on capture
                dummy = np.full((DET_CANVAS[1], DET_CANVAS[0], 3), 255, np.uint8)
                cv2.putText(dummy, "Warm up 123", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
                self._det(dummy)
                line = dummy[120:230, 20:640]
                self._rec(line)
                for eng in self._fallback.values():
                    eng(line)
            self._log(f"OCR loaded in {time.perf_counter() - t:.1f}s (det {self.device}, rec cpu)")

    def unload(self):
        with self._lock:
            self._det = self._rec = None
            self._fallback.clear()
        import gc
        gc.collect()
        self._log("OCR unloaded")

    @staticmethod
    def _best(r) -> tuple[str, float]:
        if r is None or not r.txts:
            return "", 0.0
        return str(r.txts[0]), float(r.scores[0])

    def read(self, img_bgr: np.ndarray, source: str = "auto", min_score: float = 0.5) -> list[OcrLine]:
        self.load()
        with self._lock:
            canvas, scale = _letterbox(img_bgr)
            det = self._det(canvas)
            boxes = getattr(det, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []
            lines = []
            for box in boxes:
                box = np.asarray(box, np.float32) / scale  # back to full-res coords
                crop = _crop(img_bgr, box)
                if source in self._fallback:
                    text, score = self._best(self._fallback[source](crop))
                    ln = OcrLine(text, score, box, source)
                else:
                    text, score = self._best(self._rec(crop))
                    ln = OcrLine(text, score, box)
                    if source == "auto" and _suspect(ln):
                        for name, eng in self._fallback.items():
                            t2, s2 = self._best(eng(crop))
                            if s2 > ln.score and not _suspect(OcrLine(t2, s2, box)):
                                ln = OcrLine(t2, s2, box, name)
                lines.append(ln)
        return [ln for ln in lines if ln.score >= min_score and ln.text.strip()]


def _suspect(ln: OcrLine) -> bool:
    """Primary model silently drops scripts it doesn't know (Hangul, Cyrillic)."""
    if ln.score < FALLBACK_SCORE:
        return True
    pts = ln.box
    w = float(np.linalg.norm(pts[0] - pts[1]))
    h = max(float(np.linalg.norm(pts[0] - pts[3])), 1.0)
    letters = sum(1 for c in ln.text if c.isalnum())
    weight = sum(2 if _is_cjk(c) else 1 for c in ln.text if c.isalnum())
    return letters == 0 or weight < 0.45 * (w / h)


def group_paragraphs(lines: list[OcrLine]) -> list[str]:
    """Merge OCR lines into blocks (reading order, top→bottom, left→right)."""
    if not lines:
        return []
    items = []
    for ln in lines:
        xs, ys = ln.box[:, 0], ln.box[:, 1]
        items.append((ys.min(), ys.max(), xs.min(), xs.max(), ln.text))
    items.sort(key=lambda r: (r[0], r[2]))
    blocks: list[list] = []
    for it in items:
        h = it[1] - it[0]
        for blk in blocks:
            last = blk[-1]
            lh = last[1] - last[0]
            gap = it[0] - last[1]
            overlap_x = min(it[3], last[3]) - max(it[2], last[2])
            if -0.5 * lh < gap < 0.9 * max(h, lh) and overlap_x > -0.3 * max(h, lh) \
                    and 0.55 < h / max(lh, 1) < 1.8:
                blk.append(it)
                break
        else:
            blocks.append([it])
    out = []
    for blk in blocks:
        parts = [b[4].strip() for b in blk]
        cjk = any(_is_cjk(p) for p in parts)
        out.append(("" if cjk else " ").join(parts))
    return out


def _is_cjk(s: str) -> bool:
    return any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in s)
