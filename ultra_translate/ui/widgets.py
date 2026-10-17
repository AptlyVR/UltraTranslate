"""Small shared widgets: a real toggle switch (track + sliding knob) that is a drop-in QCheckBox."""
from __future__ import annotations

from PyQt6.QtCore import QPropertyAnimation, QRectF, QSize, Qt, pyqtProperty
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QCheckBox


class Toggle(QCheckBox):
    TRACK_W, TRACK_H = 40, 22

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(140)
        self.toggled.connect(self._animate)
        self.setMinimumHeight(30)

    def _get(self) -> float:
        return self._pos

    def _set(self, v: float) -> None:
        self._pos = v
        self.update()

    knob = pyqtProperty(float, _get, _set)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool) -> None:      # programmatic changes jump without animating
        super().setChecked(on)
        self._anim.stop()
        self._pos = 1.0 if on else 0.0
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(self.TRACK_W + 12 + fm.horizontalAdvance(self.text()) + 4, max(30, fm.height() + 8))

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()
        tw, th = self.TRACK_W, self.TRACK_H
        track = QRectF(1, (h - th) / 2, tw, th)
        off, on = QColor("#2a2f4a"), QColor("#2f9e63")
        t = self._pos
        c = QColor(int(off.red() + (on.red() - off.red()) * t), int(off.green() + (on.green() - off.green()) * t),
                   int(off.blue() + (on.blue() - off.blue()) * t))
        if not self.isEnabled():
            c = QColor("#22263a")
        p.setPen(QColor("#4fd17a") if t > 0.5 and self.isEnabled() else QColor("#3a4066"))
        p.setBrush(c)
        p.drawRoundedRect(track, th / 2, th / 2)
        d = th - 6
        x = track.left() + 3 + (tw - d - 6) * t
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#ffffff") if self.isEnabled() else QColor("#5b6188"))
        p.drawEllipse(QRectF(x, track.top() + 3, d, d))
        p.setPen(QColor("#d8dcf0") if self.isEnabled() else QColor("#5b6188"))
        p.drawText(QRectF(tw + 12, 0, self.width() - tw - 12, h),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())
        p.end()
