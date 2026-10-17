"""Smart Turn v3.2 (8 MB int8 ONNX, CPU, ~20 ms): is this a finished sentence?  Ported from an earlier voice app.

Used only to end an utterance EARLY. Low confidence keeps the normal pause endpoint, so a wrong
"not finished" costs nothing; only a confident wrong "finished" can split a sentence.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "models" / "smart-turn-v3.2-cpu.onnx"
WINDOW = 8 * 16000


class SmartTurn:
    def __init__(self, path: Path = MODEL, threads: int = 4):
        self.path = path
        self.threads = threads
        self.session = None
        self.fe = None
        self.error = ""
        self.lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.path.is_file() and not self.error

    def load(self) -> None:
        with self.lock:
            if self.session is not None:
                return
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor
            so = ort.SessionOptions()
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = self.threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
            self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.predict(np.zeros(16000, np.float32))

    def predict(self, audio: np.ndarray) -> float:
        try:
            if self.session is None:
                self.load()
            a = np.asarray(audio, np.float32).reshape(-1)
            a = a[-WINDOW:] if a.size > WINDOW else np.pad(a, (WINDOW - a.size, 0))
            x = self.fe(a, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=WINDOW,
                        truncation=True, do_normalize=True).input_features
            return float(self.session.run(None, {"input_features": x.astype(np.float32)})[0].reshape(-1)[0])
        except Exception as exc:     # never break capture; the pause endpoint still applies
            self.error = str(exc)
            return 0.0
