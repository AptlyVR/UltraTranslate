"""Speaker list. Each speaker's name is a dropdown of the players in your VRChat instance (from the
VRChat log), or any name you type. Save/forget voiceprint, merge two speakers."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

UNNAMED = "— pick a player or type a name —"


class NoWheelCombo(QComboBox):
    """Scrolling the speaker list over a dropdown must not silently change it."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, e):
        if self.hasFocus() and self.view().isVisible():
            super().wheelEvent(e)
        else:
            e.ignore()


class NameCombo(NoWheelCombo):
    def __init__(self, sid: str):
        super().__init__()
        self.sid = sid
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.lineEdit().setPlaceholderText(UNNAMED)
        self.setMaxVisibleItems(25)

    def set_players(self, players: list[str], current: str) -> None:
        if self.view().isVisible() or self.lineEdit().hasFocus():
            return   # user is choosing/typing
        self.blockSignals(True)
        self.clear()
        self.addItem("")
        self.addItems(players)
        self.setEditText(current)
        self.blockSignals(False)


class SpeakersPanel(QWidget):
    renamed = pyqtSignal(str, str)
    save_print = pyqtSignal(str)
    forget_print = pyqtSignal(str)
    merge = pyqtSignal(str, str)     # keep, drop
    voice_changed = pyqtSignal(str, str)   # speaker id, voice id ("" = auto)

    COLS = ["", "Who is this?", "Voice", "Lang", "#", "✓"]

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("People")
        self.title.setStyleSheet("font-size:11pt;font-weight:700;color:#7aa2f7;padding:2px 2px 6px 2px;background:transparent;")
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(42)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setShowGrid(False)
        self.table.setStyleSheet("QTableWidget::item:selected{background:#262c47;}")
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 18)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(2, 120)
        h.setMinimumSectionSize(40)
        for c in (3, 4, 5):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnHidden(3, True)        # language and line count are shown in the conversation itself
        self.table.setColumnHidden(4, True)
        self.table.setColumnWidth(2, 110)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row2 = QHBoxLayout()
        self.btn_save = QPushButton("Remember voice")
        self.btn_save.setToolTip("Remember this voice under its name so it is recognised automatically next time.")
        self.btn_forget = QPushButton("Forget")
        self.merge_into = QComboBox()
        self.btn_merge = QPushButton("Same person as")
        self.btn_merge.setToolTip("Selected speaker is the same person as the one chosen on the right: merge them.")
        for b in (self.btn_save, self.btn_forget, self.btn_merge):
            b.setMinimumHeight(36)
        self.merge_into.setMinimumHeight(36)
        row.addWidget(self.btn_save, 1)
        row.addWidget(self.btn_forget)
        row2.addWidget(self.btn_merge)
        row2.addWidget(self.merge_into, 1)
        lay.addLayout(row)
        lay.addLayout(row2)
        self.btn_save.clicked.connect(lambda: self._with_selected(self.save_print.emit))
        self.btn_forget.clicked.connect(lambda: self._with_selected(self.forget_print.emit))
        self.btn_merge.clicked.connect(self._merge)
        self._ids: list[str] = []
        self._combos: dict[str, NameCombo] = {}
        self.players: list[str] = []
        self._speakers: list = []
        self._voice_combos: dict[str, QComboBox] = {}
        self._voices: list = []
        self._manual = False

    def _selected(self) -> str:
        r = self.table.currentRow()
        return self._ids[r] if 0 <= r < len(self._ids) else ""

    def _with_selected(self, fn):
        sid = self._selected()
        if sid:
            fn(sid)

    def _merge(self):
        drop = self._selected()
        keep = self.merge_into.currentData()
        if drop and keep and drop != keep:
            self.merge.emit(keep, drop)

    def set_players(self, players: list[str], world: str = "") -> None:
        self.players = players
        self.title.setText(f"People  ·  {len(players)} players in {world}" if world else "People")
        for sp in self._speakers:
            c = self._combos.get(sp.id)
            if c:
                c.set_players(players, sp.name)

    def _combo_for(self, sp) -> NameCombo:
        c = self._combos.get(sp.id)
        if c is None:
            c = NameCombo(sp.id)
            c.setPlaceholderText(sp.label)
            c.activated.connect(lambda _i, cc=c: self.renamed.emit(cc.sid, cc.currentText().strip()))
            c.lineEdit().editingFinished.connect(lambda cc=c: self.renamed.emit(cc.sid, cc.currentText().strip()))
            self._combos[sp.id] = c
        c.lineEdit().setPlaceholderText(f"{sp.label}  ·  pick a player or type a name")
        c.set_players(self.players, sp.name)
        return c

    def set_voices(self, voices: list, manual: bool) -> None:
        self._voices = voices
        self._manual = manual
        self.table.setColumnHidden(2, not manual)
        for c in self._voice_combos.values():
            self._fill_voice_combo(c, c.currentData() or "")

    def _fill_voice_combo(self, c: QComboBox, current: str) -> None:
        c.blockSignals(True)
        c.clear()
        c.addItem("(auto)", "")
        for v in self._voices:
            c.addItem(v.name, v.id)
        c.setCurrentIndex(max(0, c.findData(current)))
        c.blockSignals(False)

    def _voice_combo_for(self, sid: str, current: str) -> QComboBox:
        c = self._voice_combos.get(sid)
        if c is None:
            c = NoWheelCombo()
            c.setMaxVisibleItems(25)
            self._fill_voice_combo(c, current)
            c.currentIndexChanged.connect(lambda _i, cc=c, s=sid: self.voice_changed.emit(s, cc.currentData() or ""))
            self._voice_combos[sid] = c
        elif (c.currentData() or "") != current and not c.view().isVisible():
            c.blockSignals(True)
            c.setCurrentIndex(max(0, c.findData(current)))
            c.blockSignals(False)
        return c

    def update_speakers(self, speakers: list, voices: dict | None = None) -> None:
        self._speakers = speakers
        sel = self._selected()
        ids = [s.id for s in speakers]
        if ids != self._ids:
            self.table.setRowCount(len(speakers))
            for gone in set(self._combos) - set(ids):
                self._combos.pop(gone, None)
        self._ids = ids
        self.merge_into.blockSignals(True)
        self.merge_into.clear()
        for r, s in enumerate(speakers):
            dot = QTableWidgetItem("●")
            dot.setForeground(QColor(s.color))
            self.table.setItem(r, 0, dot)
            combo = self._combo_for(s)
            if self.table.cellWidget(r, 1) is not combo:
                self.table.setCellWidget(r, 1, combo)
            vc = self._voice_combo_for(s.id, (voices or {}).get(s.id, ""))
            if self.table.cellWidget(r, 2) is not vc:
                self.table.setCellWidget(r, 2, vc)
            for col, text in ((3, (s.language or "?").upper()), (4, str(s.utterances)),
                              (5, f"✓ {s.match_score:.2f}" if s.identity and s.match_score else ("✓" if s.identity else ""))):
                self.table.setItem(r, col, QTableWidgetItem(text))
            self.merge_into.addItem(s.label, s.id)
        self.merge_into.blockSignals(False)
        if sel in self._ids:
            self.table.selectRow(self._ids.index(sel))
