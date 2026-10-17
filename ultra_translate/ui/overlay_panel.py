"""The VR wrist panel: one painted widget (no child widgets, no timers), re-rendered only on change.

Chat view: speaker name + original (small) + translation (large), newest at the bottom.
Tap a speaker's name with the controller laser -> the panel becomes a big grid of the players in your
instance; tap one to name that speaker (Clear / Cancel tiles too).
"""
from __future__ import annotations

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QWidget

W = 640


class OverlayPanel(QWidget):
    name_picked = pyqtSignal(str, str)        # speaker id, name ("" = clear)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lines: list[tuple] = []          # (name, color, original, translation, speaker_id)
        self.status = ""
        self.players: list[str] = []
        self.picking = ""                     # speaker id being named, "" = chat view
        self.picking_label = ""
        self._name_rects: list[tuple[QRect, str, str]] = []
        self._tiles: list[tuple[QRect, str]] = []
        self.on_change = lambda: None         # ask the VR service to re-render
        self.resize(W, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)

    # -- data -----------------------------------------------------------------------------------
    def set_lines(self, lines, status: str = "") -> None:
        self.lines = [tuple(x) + ("",) * (5 - len(x)) for x in lines[-self.cfg.lines:]]
        self.status = status
        if not self.picking:
            self.resize(W, self._content_height())

    def set_players(self, players: list[str]) -> None:
        self.players = players[:14]

    def _fonts(self):
        px = self.cfg.font_px
        f_name, f_orig, f_tr = QFont("Segoe UI"), QFont("Segoe UI"), QFont("Segoe UI")
        f_name.setPixelSize(int(px * 0.72))
        f_name.setBold(True)
        f_orig.setPixelSize(int(px * 0.78))
        f_tr.setPixelSize(px)
        f_tr.setWeight(QFont.Weight.DemiBold)
        return f_name, f_orig, f_tr

    def _content_height(self) -> int:
        f_name, f_orig, f_tr = self._fonts()
        flags, inner, h = int(Qt.TextFlag.TextWordWrap), W - 40, 36
        for _, _, orig, tr, _ in self.lines:
            show_orig = self.cfg.show_original and orig and orig != tr
            h += QFontMetrics(f_name).height() + 16
            if show_orig:
                h += QFontMetrics(f_orig).boundingRect(QRect(0, 0, inner, 4000), flags, orig).height()
            h += QFontMetrics(f_tr).boundingRect(QRect(0, 0, inner, 4000), flags, tr or orig).height()
        return max(140, min(900, h))

    # -- input (laser clicks arrive as mouse events from the SteamVR service) ---------------------
    def mousePressEvent(self, ev):
        pos = ev.position().toPoint()
        if self.picking:
            for rect, value in self._tiles:
                if rect.contains(pos):
                    sid = self.picking
                    self.picking = ""
                    if value != "\x00cancel":
                        self.name_picked.emit(sid, value)
                    self.resize(W, self._content_height())
                    self.on_change()
                    return
            return
        for rect, sid, label in self._name_rects:
            if sid and rect.adjusted(-10, -8, 60, 8).contains(pos):
                self.picking, self.picking_label = sid, label
                cols, rows = 2, (len(self.players) + 2 + 1) // 2
                self.resize(W, 90 + rows * 76)
                self.on_change()
                return

    # -- painting -------------------------------------------------------------------------------
    def paintEvent(self, _):
        p = QPainter(self)
        self.paint_into(p)
        p.end()

    def render(self, painter, *args, **kwargs):  # used by the SteamVR service
        self.paint_into(painter)

    def paint_into(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(12, 14, 22, 230))
        p.drawRoundedRect(QRect(0, 0, w, h), 22, 22)
        if self.picking:
            self._paint_picker(p, w, h)
        else:
            self._paint_chat(p, w, h)

    def _paint_picker(self, p: QPainter, w: int, h: int) -> None:
        title = QFont("Segoe UI")
        title.setPixelSize(26)
        title.setBold(True)
        tile_f = QFont("Segoe UI")
        tile_f.setPixelSize(22)
        p.setPen(QColor("#c0caf5"))
        p.setFont(title)
        p.drawText(QRect(20, 14, w - 40, 50), int(Qt.AlignmentFlag.AlignVCenter), f"Who is {self.picking_label}?")
        options = [(n, n) for n in self.players] + [("Clear name", ""), ("Cancel", "\x00cancel")]
        if not self.players:
            options = [("No players found in the VRChat log", "\x00cancel")] + options[-2:]
        self._tiles = []
        tw, th, gap = (w - 60) // 2, 64, 12
        for i, (label, value) in enumerate(options):
            x = 20 + (i % 2) * (tw + gap + 8)
            y = 80 + (i // 2) * (th + gap)
            rect = QRect(x, y, tw, th)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#2c3152") if not value.startswith("\x00") and value else QColor("#3a2232"))
            p.drawRoundedRect(rect, 14, 14)
            p.setPen(QColor("#ffffff"))
            p.setFont(tile_f)
            p.drawText(rect.adjusted(14, 0, -10, 0), int(Qt.AlignmentFlag.AlignVCenter), label[:22])
            self._tiles.append((rect, value))

    def _paint_chat(self, p: QPainter, w: int, h: int) -> None:
        f_name, f_orig, f_tr = self._fonts()
        flags = int(Qt.TextFlag.TextWordWrap)
        inner = w - 40
        blocks = []
        for name, color, orig, tr, sid in self.lines:
            show_orig = self.cfg.show_original and orig and orig != tr
            hh = QFontMetrics(f_name).height() + 2
            if show_orig:
                hh += QFontMetrics(f_orig).boundingRect(QRect(0, 0, inner, 4000), flags, orig).height()
            if tr or not show_orig:
                hh += QFontMetrics(f_tr).boundingRect(QRect(0, 0, inner, 4000), flags, tr or orig).height()
            blocks.append((hh + 14, name, color, orig if show_orig else "", tr or orig, sid))
        y = h - 18
        drawn = []
        for b in reversed(blocks):
            if y - b[0] < 14:
                break
            y -= b[0]
            drawn.append((y, b))
        self._name_rects = []
        fm = QFontMetrics(f_name)
        for y0, (hh, name, color, orig, tr, sid) in drawn:
            p.setPen(QColor(color))
            p.setFont(f_name)
            p.drawText(20, y0 + fm.ascent(), name)
            self._name_rects.append((QRect(20, y0, fm.horizontalAdvance(name), fm.height()), sid, name))
            if sid and not name.startswith("Screen"):
                p.setPen(QColor("#565f89"))
                p.drawText(24 + fm.horizontalAdvance(name), y0 + fm.ascent(), "  ✎")
            yy = y0 + fm.height() + 2
            if orig:
                p.setPen(QColor("#a9b1d6"))
                p.setFont(f_orig)
                br = QFontMetrics(f_orig).boundingRect(QRect(0, 0, inner, 4000), flags, orig)
                p.drawText(QRect(20, yy, inner, br.height()), flags, orig)
                yy += br.height()
            p.setPen(QColor("#ffffff"))
            p.setFont(f_tr)
            br = QFontMetrics(f_tr).boundingRect(QRect(0, 0, inner, 4000), flags, tr)
            p.drawText(QRect(20, yy, inner, br.height()), flags, tr)
        if not drawn:
            p.setPen(QColor("#7d8590"))
            p.setFont(f_orig)
            p.drawText(QRect(0, 0, w, h), int(Qt.AlignmentFlag.AlignCenter), self.status or "Listening…")
