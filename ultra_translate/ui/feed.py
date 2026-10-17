"""Conversation feed: model/view list (O(1) upsert by id), painted by a delegate. Cheap at any size."""
from __future__ import annotations

import dataclasses

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QAbstractItemView, QListView, QStyledItemDelegate

from ..core.events import Line

LANG_LABEL = {"ja": "JA", "en": "EN", "es": "ES", "zh": "ZH", "ko": "KO", "fr": "FR", "de": "DE", "ru": "RU",
              "pt": "PT", "it": "IT", "nl": "NL", "pl": "PL", "tr": "TR", "ar": "AR", "th": "TH", "vi": "VI",
              "id": "ID", "uk": "UK", "hi": "HI", "sv": "SV", "fi": "FI", "cs": "CS"}


class FeedModel(QAbstractListModel):
    changed = pyqtSignal()

    def __init__(self, max_items: int = 400):
        super().__init__()
        self.rows: list[Line] = []
        self.index_of: dict[str, int] = {}
        self.max_items = max_items
        self.names: dict[str, tuple[str, str]] = {}    # speaker_id -> (label, color)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.UserRole:
            return self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            ln = self.rows[index.row()]
            return f"{ln.text}\n{ln.translation}"
        return None

    def _reindex(self, start: int = 0):
        for i in range(start, len(self.rows)):
            self.index_of[self.rows[i].id] = i

    def upsert(self, line: Line) -> None:
        i = self.index_of.get(line.id)
        if line.dropped:
            if i is not None:
                self.beginRemoveRows(QModelIndex(), i, i)
                del self.rows[i]
                self.index_of.pop(line.id, None)
                self.endRemoveRows()
                self._reindex(i)
                self.changed.emit()
            return
        if i is not None:
            old = self.rows[i]
            if old.final and not line.final:
                return  # late partial after the final: ignore
            if not line.translation and old.translation:
                # keep showing the previous translation until the new one arrives (no blank flash);
                # a copy, so the pipeline's own line object is never modified by the UI
                line = dataclasses.replace(line, translation=old.translation)
                line.stale = True  # type: ignore[attr-defined]
            self.rows[i] = line
            idx = self.index(i)
            self.dataChanged.emit(idx, idx)
        else:
            n = len(self.rows)
            self.beginInsertRows(QModelIndex(), n, n)
            self.rows.append(line)
            self.index_of[line.id] = n
            self.endInsertRows()
            if n + 1 > self.max_items:
                drop = n + 1 - self.max_items
                self.beginRemoveRows(QModelIndex(), 0, drop - 1)
                for ln in self.rows[:drop]:
                    self.index_of.pop(ln.id, None)
                del self.rows[:drop]
                self.endRemoveRows()
                self._reindex()
        self.changed.emit()

    def refresh_names(self, names: dict[str, tuple[str, str]]) -> None:
        self.names = names
        if self.rows:
            self.dataChanged.emit(self.index(0), self.index(len(self.rows) - 1))
        self.changed.emit()

    def clear(self) -> None:
        self.beginResetModel()
        self.rows.clear()
        self.index_of.clear()
        self.endResetModel()
        self.changed.emit()

    def recent(self, n: int) -> list[Line]:
        return [r for r in self.rows if r.final or r.text][-n:]


class FeedDelegate(QStyledItemDelegate):
    PAD = 10

    def __init__(self, model: FeedModel, scale: float = 1.0, show_original: bool = True):
        super().__init__()
        self.model = model
        self.show_original = show_original
        self.set_scale(scale)

    def set_scale(self, scale: float):
        base = QFont("Segoe UI")
        self.f_name = QFont(base)
        self.f_name.setPointSizeF(9.5 * scale)
        self.f_name.setBold(True)
        self.f_meta = QFont(base)
        self.f_meta.setPointSizeF(8.0 * scale)
        self.f_orig = QFont(base)
        self.f_orig.setPointSizeF(10.5 * scale)
        self.f_tr = QFont(base)
        self.f_tr.setPointSizeF(13.0 * scale)
        self.f_tr.setWeight(QFont.Weight.DemiBold)

    def _texts(self, ln: Line):
        tr = ln.translation
        orig = ln.text if (self.show_original or not tr) else ""
        if tr and tr.strip() == ln.text.strip():
            orig = ""
        return orig, tr

    def _speaker(self, ln: Line):
        if ln.source == "screen":
            return "Screen", "#e0af68"
        if ln.source == "chatbox":
            return f"{ln.speaker_id} · chatbox", "#73daca"
        if ln.source != "voice":
            return f"{ln.speaker_id} · chat", "#c0caf5"
        if not ln.speaker_id:
            return "…", "#8089a8"
        return self.model.names.get(ln.speaker_id, (f"Speaker {ln.speaker_id[1:]}", "#7aa2ff"))

    def sizeHint(self, option, index):
        ln: Line = index.data(Qt.ItemDataRole.UserRole)
        w = max(200, option.rect.width() or 600) - 2 * self.PAD - 14
        orig, tr = self._texts(ln)
        h = self.PAD + QFontMetrics(self.f_name).height() + 4
        flags = int(Qt.TextFlag.TextWordWrap)
        if orig:
            h += QFontMetrics(self.f_orig).boundingRect(QRect(0, 0, w, 10000), flags, orig).height() + 2
        if tr:
            h += QFontMetrics(self.f_tr).boundingRect(QRect(0, 0, w, 10000), flags, tr).height() + 2
        return QSize(w, h + self.PAD)

    def paint(self, p: QPainter, option, index):
        ln: Line = index.data(Qt.ItemDataRole.UserRole)
        r = option.rect.adjusted(4, 3, -4, -3)
        name, color = self._speaker(ln)
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor("#1b1e2b") if ln.final else QColor("#171925")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, 8, 8)
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRect(r.left(), r.top(), 4, r.height()), 2, 2)
        x = r.left() + self.PAD + 4
        y = r.top() + self.PAD - 4
        w = r.width() - 2 * self.PAD - 6
        p.setFont(self.f_name)
        p.setPen(QColor(color))
        fm = QFontMetrics(self.f_name)
        p.drawText(x, y + fm.ascent(), name)
        meta = []
        if ln.language:
            meta.append(LANG_LABEL.get(ln.language, ln.language.upper()))
        if ln.overlap:
            meta.append("overlap")
        if not ln.final:
            meta.append("listening…")
        elif ln.timings:
            t = ln.timings
            stage = "ocr" if ln.source in ("screen", "chatbox") else "asr"
            meta.append(f"{stage} {t.get('asr', 0)}ms" + (f" · mt {t['mt']}ms" if "mt" in t else ""))
        p.setFont(self.f_meta)
        p.setPen(QColor("#7d8590"))
        p.drawText(x + fm.horizontalAdvance(name) + 10, y + fm.ascent(), "  ·  ".join(meta))
        y += fm.height() + 4
        orig, tr = self._texts(ln)
        flags = int(Qt.TextFlag.TextWordWrap)
        if orig:
            p.setFont(self.f_orig)
            p.setPen(QColor("#a9b1d6") if tr else QColor("#e6e9f5"))
            br = QFontMetrics(self.f_orig).boundingRect(QRect(0, 0, w, 10000), flags, orig)
            p.drawText(QRect(x, y, w, br.height()), flags, orig)
            y += br.height() + 2
        if tr:
            p.setFont(self.f_tr)
            p.setPen(QColor("#c8cde0") if getattr(ln, "stale", False) else QColor("#ffffff"))
            br = QFontMetrics(self.f_tr).boundingRect(QRect(0, 0, w, 10000), flags, tr)
            p.drawText(QRect(x, y, w, br.height()), flags, tr)
        p.restore()


class FeedView(QListView):
    def __init__(self, model: FeedModel, delegate: FeedDelegate):
        super().__init__()
        self.setModel(model)
        self.setItemDelegate(delegate)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setUniformItemSizes(False)
        self.setWordWrap(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setStyleSheet("QListView{background:#11131c;border:none;}")
        self._follow = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        model.rowsInserted.connect(self._autoscroll)
        model.dataChanged.connect(self._relayout)

    def _on_scroll(self, v):
        sb = self.verticalScrollBar()
        self._follow = v >= sb.maximum() - 40

    def _relayout(self, *_):
        # row heights change when partial text grows or a translation arrives
        self.scheduleDelayedItemsLayout()
        self._autoscroll()

    def _autoscroll(self, *_):
        if self._follow:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.scrollToBottom)
