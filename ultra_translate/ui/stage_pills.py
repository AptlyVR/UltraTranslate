"""A row of colour-coded stage pills (Translator · Speech · Speakers · Turn-taking · Listening) with animation.

pending = grey, working = amber with a running shimmer, done = green with a tick and time, failed = red,
live (Listening) = green pulse that brightens with the audio level. Driven by the pipeline's status lines.
"""
from __future__ import annotations

import math
import re
import time

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget

STAGES = [("mt", "Translator"), ("asr", "Speech"), ("spk", "Speakers"), ("turn", "Turn-taking"), ("live", "Listening")]
COL = {"pending": ("#232740", "#3a4066", "#7d84a8"), "working": ("#6b4c10", "#e0af68", "#fff3d6"),
       "done": ("#1d4d33", "#3fbf7f", "#e6fff0"), "failed": ("#5c1f2a", "#f7768e", "#ffe3e8"),
       "off": ("#1b1e2b", "#2a2f4a", "#565d80"), "live": ("#1f6b3f", "#4fd17a", "#ffffff")}


class StagePills(QWidget):
    def __init__(self, parent=None, level=lambda: 0.0):
        super().__init__(parent)
        self.level = level
        self.state = {k: "pending" for k, _ in STAGES}
        self.note = {k: "" for k, _ in STAGES}
        self.started = {}
        self.setMinimumHeight(34)
        self._t = QTimer(self, interval=50, timeout=self._tick)
        self._t.start()

    # -- state ---------------------------------------------------------------------------------------
    def set(self, key: str, state: str, note: str = "") -> None:
        if state == "working" and self.state.get(key) != "working":
            self.started[key] = time.monotonic()
        self.state[key], self.note[key] = state, note
        self.update()

    def on_status(self, msg: str) -> None:
        m = msg.strip()
        if m.startswith("Loading speech models"):
            for k in ("asr", "spk", "turn"):
                if self.state[k] != "done":
                    self.set(k, "working")
            if self.state["mt"] != "done":
                self.set("mt", "working")
            self.set("live", "pending")
        elif m.startswith("Translator ready"):
            ms = re.search(r"(\d+) ms", m)
            self.set("mt", "done", f"{int(ms.group(1)) / 1000:.1f}s" if ms else "")
        elif m.startswith("TranslateHub not reachable") or m.startswith("Translator load failed"):
            self.set("mt", "failed")
        elif m.startswith("ASR") and "ready" in m:
            ms = re.search(r"(\d+) ms", m)
            self.set("asr", "done", f"{int(ms.group(1)) / 1000:.1f}s" if ms else "")
        elif m.startswith("Speaker models ready"):
            ms = re.search(r"(\d+) ms", m)
            self.set("spk", "done", f"{int(ms.group(1)) / 1000:.1f}s" if ms else "")
        elif m.startswith("Models ready"):
            for k in ("asr", "spk", "turn"):
                if self.state[k] == "working":
                    self.set(k, "done")
        elif m.startswith("Model load failed") or m.startswith("Start failed"):
            for k, _ in STAGES:
                if k != "mt" and self.state[k] == "working":      # the translator loads on its own
                    self.set(k, "failed")
        elif m == "Listening.":
            self.set("live", "live")
        elif m == "Stopped.":
            self.set("live", "pending")
        elif m.startswith("Models unloaded") or m.startswith("Speech models unloaded"):
            for k in ("asr", "spk", "turn", "live"):
                self.set(k, "pending")

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        f = QFont("Segoe UI", 10)
        f.setWeight(QFont.Weight.DemiBold)
        from PyQt6.QtGui import QFontMetrics
        fm = QFontMetrics(f)
        return QSize(sum(fm.horizontalAdvance(f"◐ {l} 00.0s") + 34 for _, l in STAGES), 34)

    def _tick(self):
        if any(s in ("working", "live") for s in self.state.values()):
            self.update()

    # -- paint ---------------------------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        f = QFont("Segoe UI", 10)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        fm = p.fontMetrics()
        now = time.monotonic()
        x, h = 2.0, self.height() - 6.0
        for i, (k, label) in enumerate(STAGES):
            st = self.state[k]
            icon = {"done": "✓ ", "failed": "✕ ", "live": "● ", "working": "◌ "}.get(st, "")
            if st == "working":
                icon = "◐◓◑◒"[int(now * 6) % 4] + " "
                secs = now - self.started.get(k, now)
                text = f"{icon}{label} {secs:.0f}s"
            else:
                text = f"{icon}{label}" + (f" {self.note[k]}" if self.note[k] else "")
            w = fm.horizontalAdvance(text) + 26
            r = QRectF(x, 3, w, h)
            bg, border, fg = (QColor(c) for c in COL[st])
            if st == "live":       # breathing pulse, brighter while someone is talking
                a = 0.5 + 0.5 * math.sin(now * 3.2)
                lv = min(1.0, max(0.0, float(self.level() or 0.0)) * 2.5)
                bg = QColor.fromHsvF(0.40, 0.65, min(1.0, 0.36 + 0.16 * a + 0.35 * lv))
                border = QColor.fromHsvF(0.40, 0.6, min(1.0, 0.7 + 0.3 * max(a, lv)))
            p.setPen(QPen(border, 1.6))
            p.setBrush(bg)
            p.drawRoundedRect(r, h / 2, h / 2)
            if st == "working":    # shimmer sweeping across the pill
                pos = (now * 0.9 + i * 0.17) % 1.0
                g = QLinearGradient(r.left(), 0, r.right(), 0)
                g.setColorAt(max(0.0, pos - 0.25), QColor(255, 220, 140, 0))
                g.setColorAt(pos, QColor(255, 220, 140, 90))
                g.setColorAt(min(1.0, pos + 0.25), QColor(255, 220, 140, 0))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(g)
                p.drawRoundedRect(r, h / 2, h / 2)
            p.setPen(fg)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
            x += w + 8
            if i < len(STAGES) - 1:   # connector
                done = st in ("done", "live")
                p.setPen(QPen(QColor("#3fbf7f" if done else "#3a4066"), 2))
                p.drawLine(int(x - 7), int(3 + h / 2), int(x - 1), int(3 + h / 2))
        p.end()
