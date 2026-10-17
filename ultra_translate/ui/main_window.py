from __future__ import annotations

import threading
import time

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                             QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..core import audio as audio_mod
from ..core.pipeline import Pipeline
from ..core.events import Line
from ..intake.players import PlayerTracker
from ..intake.vrcx import VrcxIntake
from ..screen.service import ScreenTranslator
from ..settings import Settings, SettingsStore
from .feed import FeedDelegate, FeedModel, FeedView
from .overlay_panel import OverlayPanel
from .speakers_panel import SpeakersPanel
from .stage_pills import StagePills
from .widgets import Toggle

TARGETS = [("English", "en"), ("Japanese", "ja"), ("Spanish", "es"), ("Chinese", "zh"), ("Korean", "ko"),
           ("French", "fr"), ("German", "de"), ("Portuguese", "pt"), ("Russian", "ru"), ("Italian", "it"),
           ("Thai", "th"), ("Vietnamese", "vi"), ("Indonesian", "id"), ("Turkish", "tr"), ("Polish", "pl"),
           ("Dutch", "nl"), ("Arabic", "ar"), ("Ukrainian", "uk"), ("Hindi", "hi")]

STYLE = """
QMainWindow, QWidget { background:#0f1118; color:#d8dcf0; font-family:'Segoe UI'; font-size:10.5pt; }
QLabel#title { font-size:15pt; font-weight:700; color:#c0caf5; padding-right:6px; }
QFrame#card { background:#161927; border:1px solid #232741; border-radius:14px; }
QFrame#feedcard { background:#12141f; border:1px solid #1f2337; border-radius:14px; }
QFrame#vsep { background:#2a2f4a; margin:6px 2px; }
QLabel#cardlabel { color:#9aa5ce; font-size:11pt; font-weight:600; background:transparent; }
QLabel#volval { color:#e6e9f5; font-size:12pt; font-weight:700; background:transparent; }
QLabel#volval:disabled, QLabel#cardlabel:disabled { color:#4b5070; }
QLabel#hint { color:#6b7194; font-size:9.5pt; background:transparent; }
QLabel#section { color:#7aa2f7; font-size:11pt; font-weight:700; padding:12px 0 2px 0; background:transparent; }
QLabel#status { color:#9aa5ce; }
QLabel#stats { color:#5d6488; }
QPushButton { background:#232740; border:1px solid #30365a; border-radius:8px; padding:6px 14px; }
QPushButton:hover { background:#2c3152; }
QPushButton:pressed { background:#4a5590; border-color:#7aa2f7; }
QPushButton:disabled { color:#5b6188; background:#1a1d2c; border-color:#252a42; }
QPushButton#flat { background:transparent; border:none; color:#6b7194; padding:2px 8px; }
QPushButton#flat:hover { color:#c0caf5; }
QPushButton#start { font-size:13pt; font-weight:700; min-height:48px; border-radius:12px; padding:0 22px;
                    background:#3d59a1; border:1px solid #5a78c8; color:#ffffff; }
QPushButton#start:hover { background:#4766b4; }
QPushButton#start:pressed { background:#5a78c8; }
QPushButton#start[state="loading"] { background:#8a6416; border-color:#e0af68; color:#fff6e0; }
QPushButton#start[running="true"] { background:#2d7d46; border:2px solid #4fd17a; }
QPushButton#start[running="true"]:pressed { background:#3f9d5c; }
QPushButton#chip { font-size:11pt; min-height:46px; border-radius:23px; padding:0 18px;
                   background:#1c2033; border:1px solid #2e3452; color:#9aa5ce; }
QPushButton#chip:hover { background:#252a42; color:#d8dcf0; }
QPushButton#chip:pressed { background:#323a5e; }
QPushButton#chip[on="true"] { background:#1f5a3a; border:2px solid #3fbf7f; color:#eafff2; }
QPushButton#chip[on="true"]:hover { background:#256b45; }
QPushButton#chip[on="true"]:pressed { background:#2f8556; }
QPushButton#action { font-size:11pt; min-height:46px; border-radius:12px; padding:0 18px;
                     background:#2a2f4a; border:1px solid #3d4466; }
QPushButton#action:pressed { background:#e0af68; color:#1a1408; }
QComboBox { background:#1b1e2b; border:1px solid #30365a; border-radius:8px; padding:5px 10px; min-height:22px; }
QComboBox:hover { border-color:#4d5680; }
QComboBox#bar { font-size:11.5pt; min-height:36px; border-radius:12px; padding:4px 14px; min-width:150px; }
QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:center right; width:28px;
                      border:none; background:transparent; }
QComboBox#bar::drop-down { width:34px; }
QComboBox::down-arrow { image:url(CHEVRON); width:12px; height:12px; }
QComboBox QAbstractItemView { background:#1b1e2b; selection-background-color:#3d59a1; font-size:11pt; }
QCheckBox { spacing:10px; background:transparent; padding:2px 0; }
QCheckBox::indicator { width:36px; height:20px; border-radius:10px; background:#2a2f4a; border:1px solid #3a4066; }
QCheckBox::indicator:checked { background:#3fbf7f; border-color:#4fd17a; }
QCheckBox::indicator:hover { border-color:#7aa2f7; }
QSlider#volume { background:transparent; }
QSlider#volume::groove:horizontal { height:12px; border-radius:6px; background:#262a42; }
QSlider#volume::sub-page:horizontal { border-radius:6px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                      stop:0 #2d7d46, stop:1 #4fd17a); }
QSlider#volume::handle:horizontal { width:30px; height:30px; margin:-10px 0; border-radius:15px;
                                    background:#eafff2; border:2px solid #3fbf7f; }
QSlider#volume::handle:horizontal:hover { background:#ffffff; border-color:#7ee2a8; }
QSlider#volume::sub-page:horizontal:disabled { background:#33384f; }
QSlider#volume::handle:horizontal:disabled { background:#5b6188; border-color:#3a4066; }
QTabWidget#settings::pane { border:none; }
QTabWidget#settings QTabBar::tab { background:transparent; color:#7d84a8; padding:8px 12px; margin:0 2px;
                                   border:none; border-radius:10px; font-weight:600; }
QTabWidget#settings QTabBar::tab:hover { color:#d8dcf0; background:#1f2335; }
QTabWidget#settings QTabBar::tab:selected { color:#ffffff; background:#2b3256; }
QTabWidget#settings QStackedWidget, QTabWidget#settings QStackedWidget > QWidget { background:#161927; }
QWidget#page { background:#161927; }
QWidget#page QWidget { background:transparent; }
QWidget#page QComboBox { background:#1f2335; }
QLabel#group { color:#7aa2f7; font-size:10pt; font-weight:700; padding-top:8px; background:transparent; }
QScrollArea#drawer { background:#161927; border:1px solid #232741; border-radius:14px; }
QWidget#drawerinner { background:#161927; }
QWidget#drawerinner QWidget { background:transparent; }
QWidget#drawerinner QComboBox, QWidget#drawerinner QPushButton { background:#1f2335; }
QTableWidget { background:transparent; gridline-color:#232740; border:none; }
QHeaderView::section { background:transparent; color:#6b7194; border:none; padding:4px; }
QPlainTextEdit { background:#0d0f16; color:#8089a8; border:none; border-radius:8px; font-family:Consolas; font-size:9pt; }
QProgressBar { background:#1b1e2b; border:none; border-radius:4px; }
QProgressBar::chunk { background:#9ece6a; border-radius:4px; }
QSplitter::handle { background:transparent; width:10px; }
QScrollBar:vertical { background:transparent; width:10px; }
QScrollBar::handle:vertical { background:#2a2f4a; border-radius:5px; min-height:30px; }
QScrollBar::add-line, QScrollBar::sub-line { height:0; }
"""


class Bridge(QObject):
    line = pyqtSignal(object)
    speakers = pyqtSignal()
    status = pyqtSignal(str)
    stats = pyqtSignal(dict)
    chat = pyqtSignal(str, str)
    models_done = pyqtSignal(bool, str)
    screen_result = pyqtSignal(list, dict)
    chat_result = pyqtSignal(list, dict)
    hotkey = pyqtSignal()
    players = pyqtSignal(list, str, str)


class MainWindow(QMainWindow):
    def __init__(self, autostart_file: str = "", autostart: bool = False, exit_when_done: bool = False,
                 demo_turns: bool = False):
        super().__init__()
        self.exit_when_done = exit_when_done
        self.store = SettingsStore()
        self.s: Settings = self.store.load()
        self.bridge = Bridge()
        self.pipe = Pipeline(self.s, self.bridge.line.emit, self.bridge.speakers.emit,
                             self.bridge.status.emit, self.bridge.stats.emit)
        self.pipe.demo_turns = demo_turns
        self.intake = VrcxIntake(self.s.intake, self.bridge.chat.emit, self.bridge.status.emit)
        self.players = PlayerTracker(self.bridge.players.emit, self.bridge.status.emit)
        self.overlay_service = None
        self._loading = False
        self.screen = ScreenTranslator(self.s.screen, self.pipe.hub, lambda: self.s.translation.target,
                                       self.bridge.screen_result.emit, self.bridge.status.emit,
                                       on_chat=self.bridge.chat_result.emit,
                                       players=lambda: getattr(self, "players_list", []))
        self.screen.head_turning = self._head_turning
        self._chord_prev = False
        self._last_hmd_fwd = None
        self.setWindowTitle("UltraTranslate")
        from pathlib import Path as _P
        self.setStyleSheet(STYLE.replace("CHEVRON", (_P(__file__).parent / "assets" / "chevron.svg").as_posix()))
        self.resize(1180, 760)
        self._build()
        self._wire()
        self._populate_devices()
        self._apply_settings_to_ui()
        self._save_timer = QTimer(self, singleShot=True, interval=800, timeout=self._save)
        self._overlay_timer = QTimer(self, singleShot=True, interval=120, timeout=self._push_overlay)
        self._meter = QTimer(self, interval=100, timeout=self._update_meter)
        self._wire_modes()
        if self.s.intake.vrcx_ws or self.s.intake.vrcx_sqlite:
            self.intake.start()
        self.players.start()
        if self.s.overlay.enabled:
            self._toggle_overlay(True)
        self._vr_timer = QTimer(self, interval=40, timeout=self._vr_poll)
        self._start_hotkey()
        if autostart_file:
            self.s.audio.source = "file"
            self.s.audio.file_path = autostart_file
            self._apply_settings_to_ui()
        if autostart or autostart_file:
            QTimer.singleShot(300, self._start_clicked)

    # -- layout ------------------------------------------------------------------------------------
    def _build(self):
        from PyQt6.QtWidgets import QFormLayout, QFrame, QSlider
        root = QWidget()
        v = QVBoxLayout(root)
        v.setContentsMargins(14, 10, 14, 8)
        v.setSpacing(10)

        # ---- every control (the settings drawer and the main bar share these) ----
        self.btn_models = QPushButton("Load models")
        self.btn_start = QPushButton("Start")          # kept for the drawer / older wiring; hidden
        self.cmb_source = QComboBox()
        self.cmb_source.addItem("System audio (what you hear)", "loopback")
        self.cmb_source.addItem("Microphone / input", "input")
        self.cmb_source.addItem("Test file", "file")
        self.cmb_device = QComboBox()
        self.cmb_device.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_device.setMinimumContentsLength(14)
        self.btn_file = QPushButton("Choose file…")
        self.cmb_target = QComboBox()
        self.cmb_target.setObjectName("bar")
        for name, code in TARGETS:
            self.cmb_target.addItem(f"Into {name}", code)
        self.cmb_mt = QComboBox()
        self.cmb_mt.addItem("Hy-MT2 7B (best)", "best")
        self.cmb_mt.addItem("Hy-MT2 1.8B (fast, low VRAM)", "fast")
        self.chk_speakers = Toggle("Tell speakers apart")
        self.chk_split = Toggle("Split people talking over each other")
        self.chk_translate = Toggle("Translate")
        self.chk_partials = Toggle("Live captions while they talk")
        self.chk_tts = Toggle("Read aloud")
        self.chk_simul = Toggle("Interpreter mode")
        self.chk_simul.setToolTip("Off: read each whole sentence after they finish - best translation.\n"
                                  "On: speak translated clauses while they are still talking - sooner, but rougher,\n"
                                  "especially for Japanese/Korean where the verb comes last.")
        self.chk_echo = Toggle("Ignore its own voice (echo cancel)")
        self.cmb_voice = QComboBox()
        self.cmb_voice.addItem("Each speaker's own voice (clone)", "own")
        self.cmb_voice.addItem("Different voice per speaker", "auto")
        self.cmb_voice.addItem("One voice for everyone", "single")
        self.cmb_voice.addItem("Pick per speaker", "manual")
        self.btn_import_voice = QPushButton("Import voice…")
        self.btn_import_voice.setToolTip("Add a voice clip (wav/mp3, 3-15 s of clean speech) to pick for any speaker")
        self.cmb_single = QComboBox()
        self.cmb_single.setToolTip("Voice used in 'One voice for everyone' mode")
        self.chk_overlay = Toggle("VR panel")
        self.chk_orig = Toggle("Show original text")
        self.chk_learn = Toggle("Auto-learn voiceprints")
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedSize(90, 8)
        self.meter.setToolTip("Input level")
        self.btn_ocr = QPushButton("Load screen reader")
        self.btn_snap = QPushButton("Read screen now (F8)")
        self.btn_snap.setToolTip("Also: hold LEFT GRIP and press LEFT B in VR")
        self.chk_live = Toggle("Keep reading (live)")
        self.cmb_screen_mode = QComboBox()
        self.cmb_screen_mode.addItem("Players' chat bubbles", "chatbox")
        self.cmb_screen_mode.addItem("All text (signs, posters)", "text")
        self.cmb_ocr_dev = QComboBox()
        self.cmb_ocr_dev.addItem("GPU (~0.5 GB)", "cuda")
        self.cmb_ocr_dev.addItem("CPU (no VRAM)", "cpu")
        self.cmb_ocr_engine = QComboBox()
        self.cmb_ocr_engine.addItem("Qwen3-VL-4B · most accurate · 5.5 GB", "vlm")
        self.cmb_ocr_engine.addItem("RapidOCR · fastest · <1 GB", "rapidocr")
        self.cmb_pin = QComboBox()
        for label, data in (("Left wrist", "left_wrist"), ("Right wrist", "right_wrist"), ("Follows head", "head"),
                            ("Fixed in world", "world")):
            self.cmb_pin.addItem(label, data)
        self.cmb_vsize = QComboBox()
        for label, w in (("Small", 0.20), ("Medium", 0.28), ("Large", 0.40), ("Huge", 0.60)):
            self.cmb_vsize.addItem(label, w)
        self.cmb_vlines = QComboBox()
        for n in (3, 4, 6, 8, 12):
            self.cmb_vlines.addItem(f"{n} lines", n)
        self.btn_place = QPushButton("Place in front of me")
        self.btn_place.setToolTip("Fixed-in-world: put the panel 0.8 m in front of where you are looking.")
        self.chk_fade = Toggle("Fade when I'm not looking at it")

        # ---- top bar: name · stage pills · level · settings ----
        head = QHBoxLayout()
        head.setSpacing(12)
        title = QLabel("UltraTranslate")
        title.setObjectName("title")
        head.addWidget(title)
        self.pills = StagePills(self, level=lambda: self.pipe.level if self.pipe.running else 0.0)
        head.addWidget(self.pills, 1)
        head.addWidget(self.meter)
        self.btn_settings = QPushButton("⚙  Settings")
        self.btn_settings.setObjectName("chip")
        self.btn_settings.setCheckable(True)
        head.addWidget(self.btn_settings)
        v.addLayout(head)

        # ---- control bar: one row, pointer-friendly but not huge ----
        bar = QFrame()
        bar.setObjectName("card")
        hb = QHBoxLayout(bar)
        hb.setContentsMargins(10, 10, 10, 10)
        hb.setSpacing(8)
        self.s_start = QPushButton("▶  Start")
        self.s_start.setObjectName("start")
        self.s_start.setMinimumWidth(200)
        hb.addWidget(self.s_start)
        hb.addWidget(self.cmb_target)
        sep = QFrame()
        sep.setObjectName("vsep")
        sep.setFixedWidth(1)
        hb.addWidget(sep)
        self.s_tts = QPushButton("🔊  Read aloud")
        self.s_simul = QPushButton("🗣  Interpreter")
        self.s_simul.setToolTip(self.chk_simul.toolTip())
        self.s_chat = QPushButton("💬  Chat bubbles")
        self.s_chat.setToolTip("Reads players' chat bubbles off the screen (no hooking), translates them and says who wrote them")
        self.s_overlay = QPushButton("🥽  VR panel")
        for b in (self.s_tts, self.s_simul, self.s_chat, self.s_overlay):
            b.setObjectName("chip")
            hb.addWidget(b)
        hb.addStretch(1)
        self.s_snap = QPushButton("📸  Read screen")
        self.s_snap.setObjectName("action")
        self.s_snap.setToolTip("Translate what's on screen now (F8, or left grip + B in VR)")
        hb.addWidget(self.s_snap)
        v.addWidget(bar)

        # ---- voice volume: a big slider, plus "lower the game while it speaks" ----
        vol = QFrame()
        vol.setObjectName("card")
        hv = QHBoxLayout(vol)
        hv.setContentsMargins(14, 6, 10, 6)
        hv.setSpacing(12)
        lab = QLabel("Voice volume")
        lab.setObjectName("cardlabel")
        hv.addWidget(lab)
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setObjectName("volume")
        self.sld_volume.setRange(0, 200)
        self.sld_volume.setPageStep(10)
        self.sld_volume.setMinimumHeight(40)
        hv.addWidget(self.sld_volume, 1)
        self.lbl_volume = QLabel("90%")
        self.lbl_volume.setObjectName("volval")
        self.lbl_volume.setFixedWidth(52)
        hv.addWidget(self.lbl_volume)
        self.s_duck = QPushButton("🔉  Lower game while it speaks")
        self.s_duck.setObjectName("chip")
        self.s_duck.setToolTip("Fades VRChat down while the translation speaks so you can hear it; the translator\n"
                               "boosts its own input by the same amount, so it still hears everyone clearly.")
        hv.addWidget(self.s_duck)
        v.addWidget(vol)
        self.vol_card = vol

        # ---- main: conversation | people (or settings drawer) ----
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        self.feed_model = FeedModel(self.s.ui.max_feed_items)
        self.delegate = FeedDelegate(self.feed_model, self.s.ui.font_scale, self.s.ui.show_original)
        self.feed = FeedView(self.feed_model, self.delegate)
        feedcard = QFrame()
        feedcard.setObjectName("feedcard")
        fl = QVBoxLayout(feedcard)
        fl.setContentsMargins(4, 4, 4, 4)
        fl.addWidget(self.feed)
        split.addWidget(feedcard)

        from PyQt6.QtWidgets import QStackedWidget
        self.side = QStackedWidget()
        self.side.setMinimumWidth(360)
        people = QFrame()
        people.setObjectName("card")
        pl = QVBoxLayout(people)
        pl.setContentsMargins(10, 8, 10, 8)
        self.speakers = SpeakersPanel()
        pl.addWidget(self.speakers)
        tip = QLabel("Tip: click a line in the conversation to name who said it.")
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        pl.addWidget(tip)
        self.side.addWidget(people)

        # settings: tabs, one aligned column each, nothing wider than the panel
        from PyQt6.QtWidgets import QSizePolicy, QTabWidget
        tabs = QTabWidget()
        tabs.setObjectName("settings")
        tabs.setDocumentMode(True)

        def page(rows):
            w = QWidget()
            w.setObjectName("page")
            f = QFormLayout(w)
            f.setContentsMargins(14, 14, 14, 14)
            f.setHorizontalSpacing(12)
            f.setVerticalSpacing(10)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for r in rows:
                if isinstance(r, str):
                    h = QLabel(r)
                    h.setObjectName("group")
                    f.addRow(h)
                elif isinstance(r, tuple):
                    f.addRow(r[0], r[1])
                else:
                    f.addRow(r)
            return w

        def row(*ws):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            for x in ws:
                h.addWidget(x)
            h.addStretch(1)
            return w

        for c in (self.cmb_source, self.cmb_device, self.cmb_mt, self.cmb_voice, self.cmb_single,
                  self.cmb_screen_mode, self.cmb_ocr_engine, self.cmb_ocr_dev, self.cmb_pin, self.cmb_vsize,
                  self.cmb_vlines):
            c.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            c.setMinimumContentsLength(8)
            c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_duck = QComboBox()
        for label, lv in (("down to 15 %", 0.15), ("down to 30 %", 0.3), ("down to 50 %", 0.5), ("down to 70 %", 0.7)):
            self.cmb_duck.addItem(label, lv)
        self.single_row_label = QLabel("Voice")
        tabs.addTab(page([("Hear", self.cmb_source), ("Device", self.cmb_device), ("", row(self.btn_file)),
                          "People", self.chk_speakers, self.chk_split, self.chk_learn,
                          "Captions", self.chk_partials,
                          "Models", row(self.btn_models)]), "Listen")
        tabs.addTab(page([("Model", self.cmb_mt), self.chk_translate, self.chk_orig]), "Translate")
        tabs.addTab(page([("Voices", self.cmb_voice), (self.single_row_label, self.cmb_single),
                          ("", row(self.btn_import_voice)),
                          "While it speaks", ("Lower game", self.cmb_duck), self.chk_echo]), "Voice")
        tabs.addTab(page([("Read", self.cmb_screen_mode), ("Engine", self.cmb_ocr_engine),
                          ("Runs on", self.cmb_ocr_dev), self.chk_live, ("", row(self.btn_ocr)),
                          ("", row(self.btn_snap))]), "Screen")
        tabs.addTab(page([self.chk_overlay, ("Pinned to", self.cmb_pin), ("", row(self.btn_place)),
                          ("Size", self.cmb_vsize), ("Lines", self.cmb_vlines), self.chk_fade]), "VR")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        logw = QWidget()
        ll = QVBoxLayout(logw)
        ll.setContentsMargins(8, 10, 8, 8)
        ll.addWidget(self.log)
        tabs.addTab(logw, "Log")
        drawer = QFrame()
        drawer.setObjectName("card")
        dv = QVBoxLayout(drawer)
        dv.setContentsMargins(6, 6, 6, 6)
        dv.addWidget(tabs)
        self.settings_tabs = tabs
        self.side.addWidget(drawer)
        split.addWidget(self.side)
        split.setStretchFactor(0, 1)
        split.setSizes([800, 400])
        v.addWidget(split, 1)

        # ---- footer: one quiet line ----
        self.foot = QWidget()
        foot = QHBoxLayout(self.foot)
        foot.setContentsMargins(2, 0, 2, 0)
        self.lbl_status = QLabel("Ready. Press Start (models load on first start).")
        self.lbl_status.setObjectName("status")
        self.lbl_stats = QLabel("")
        self.lbl_stats.setObjectName("stats")
        self.btn_clear = QPushButton("Clear conversation")
        self.btn_clear.setObjectName("flat")
        foot.addWidget(self.lbl_status, 1)
        foot.addWidget(self.lbl_stats)
        foot.addWidget(self.btn_clear)
        v.addWidget(self.foot)
        self.setCentralWidget(root)
        self.overlay_panel = OverlayPanel(self.s.overlay)

    # ---- main bar <-> settings ----------------------------------------------------------------------
    def _wire_modes(self):
        self.btn_settings.toggled.connect(lambda on: self._set_mode("advanced" if on else "simple"))
        self.s_start.clicked.connect(self._start_clicked)
        self.s_tts.clicked.connect(lambda: self.chk_tts.setChecked(not self.chk_tts.isChecked()))
        self.s_overlay.clicked.connect(lambda: self.chk_overlay.setChecked(not self.chk_overlay.isChecked()))
        self.s_snap.clicked.connect(self._snap)
        self.s_simul.clicked.connect(lambda: self.chk_simul.setChecked(not self.chk_simul.isChecked()))
        self.s_chat.clicked.connect(self._toggle_chatbox_reader)
        self.s_duck.clicked.connect(self._toggle_duck)
        self.cmb_duck.currentIndexChanged.connect(self._duck_level_changed)
        self.chk_echo.toggled.connect(self._echo_toggled)
        self.sld_volume.valueChanged.connect(self._volume_changed)
        self.cmb_voice.currentIndexChanged.connect(lambda *_: self._sync_simple())
        self.cmb_screen_mode.currentIndexChanged.connect(self._screen_mode_changed)
        self.cmb_ocr_engine.currentIndexChanged.connect(self._screen_engine_changed)
        for sig in (self.chk_tts.toggled, self.chk_overlay.toggled, self.chk_live.toggled, self.chk_simul.toggled):
            sig.connect(lambda *_: self._sync_simple())
        t = self.s.tts
        self.sld_volume.blockSignals(True)
        self.sld_volume.setValue(int(round(t.volume * 100)))
        self.sld_volume.blockSignals(False)
        self.lbl_volume.setText(f"{int(round(t.volume * 100))}%")
        self.chk_echo.blockSignals(True)
        self.chk_echo.setChecked(t.echo_cancel)
        self.chk_echo.blockSignals(False)
        self.cmb_duck.blockSignals(True)
        self.cmb_duck.setCurrentIndex(min(range(self.cmb_duck.count()),
                                          key=lambda i: abs(self.cmb_duck.itemData(i) - t.duck_level)))
        self.cmb_duck.blockSignals(False)
        self.btn_start.setVisible(False)
        self._sync_simple()
        self._set_mode(self.s.ui.mode, save=False)

    def _volume_changed(self, val: int):
        self.s.tts.volume = val / 100.0              # read live by the player: changes the voice mid-sentence
        self.lbl_volume.setText(f"{val}%")
        self._save_timer.start()

    def _toggle_duck(self):
        self.s.tts.duck_game = not self.s.tts.duck_game
        self._on_status("Game is lowered while the translation speaks (applies on next Start)."
                        if self.s.tts.duck_game else "Game volume is left alone.")
        if not self.s.tts.duck_game and getattr(self.pipe, "ducker", None):
            self.pipe.ducker.stop()
            self.pipe.ducker = None
        self._sync_simple()
        self._save_timer.start()

    def _duck_level_changed(self, *_):
        self.s.tts.duck_level = float(self.cmb_duck.currentData())
        if getattr(self.pipe, "ducker", None):
            self.pipe.ducker.level = self.s.tts.duck_level
        self._save_timer.start()

    def _echo_toggled(self, on: bool):
        self.s.tts.echo_cancel = on
        self._save_timer.start()

    def _simple_place(self):
        if self.cmb_pin.currentData() != "world":
            self.cmb_pin.setCurrentIndex(self.cmb_pin.findData("world"))
        self._place_overlay()

    @staticmethod
    def _prop(w, **props):
        """Set style properties and re-apply the stylesheet so the colour changes at once."""
        for k, v in props.items():
            w.setProperty(k, v)
        w.style().unpolish(w)
        w.style().polish(w)
        w.update()

    def _sync_simple(self):
        running = self.pipe.running
        loading = self._loading
        self.s_start.setText("◌  Loading…" if loading else "●  Listening" if running else "▶  Start")
        self.s_start.setToolTip("Tap to stop" if running else "")
        self.s_start.setEnabled(not loading)
        self._prop(self.s_start, running="true" if running else "false", state="loading" if loading else "")
        chat_on = self.chk_live.isChecked() and self.s.screen.mode == "chatbox"
        for b, on in ((self.s_tts, self.chk_tts.isChecked()), (self.s_overlay, self.chk_overlay.isChecked()),
                      (self.s_simul, self.chk_simul.isChecked()), (self.s_chat, chat_on),
                      (self.s_duck, self.s.tts.duck_game)):
            self._prop(b, on="true" if on else "false")
        tts = self.chk_tts.isChecked()
        for w in (self.sld_volume, self.s_duck):
            w.setEnabled(tts)
        self.lbl_volume.setEnabled(tts)
        single = self.cmb_voice.currentData() == "single"
        self.cmb_single.setVisible(single)
        self.single_row_label.setVisible(single)

    def _set_mode(self, mode: str, save: bool = True):
        adv = mode == "advanced"
        self.btn_settings.blockSignals(True)
        self.btn_settings.setChecked(adv)
        self.btn_settings.blockSignals(False)
        self._prop(self.btn_settings, on="true" if adv else "false")
        self.side.setCurrentIndex(1 if adv else 0)
        self.delegate.set_scale(self.s.ui.font_scale * 1.2)
        self.feed.scheduleDelayedItemsLayout()
        self.feed.viewport().update()
        if save:
            self.s.ui.mode = "advanced" if adv else "simple"
            self._save_timer.start()

    def _wire(self):
        b = self.bridge
        b.line.connect(self._on_line)
        b.speakers.connect(self._on_speakers)
        b.status.connect(self._on_status)
        b.stats.connect(self._on_stats)
        b.chat.connect(lambda who, text: self.pipe.add_chat(who, text))
        b.models_done.connect(self._models_done)
        b.screen_result.connect(self._on_screen_result)
        b.chat_result.connect(self._on_chat_result)
        b.hotkey.connect(self._snap)
        b.players.connect(self._on_players)
        self.feed.clicked.connect(self._feed_clicked)
        self.overlay_panel.name_picked.connect(self._vr_name_picked)
        self.overlay_panel.on_change = lambda: self.overlay_service.mark_dirty() if self.overlay_service else None
        self.cmb_pin.currentIndexChanged.connect(self._overlay_cfg_changed)
        self.cmb_vsize.currentIndexChanged.connect(self._overlay_cfg_changed)
        self.cmb_vlines.currentIndexChanged.connect(self._overlay_cfg_changed)
        self.chk_fade.toggled.connect(self._overlay_cfg_changed)
        self.btn_place.clicked.connect(self._place_overlay)
        self.btn_ocr.clicked.connect(self._ocr_clicked)
        self.btn_snap.clicked.connect(self._snap)
        self.chk_live.toggled.connect(self._live_toggled)
        self.cmb_ocr_dev.currentIndexChanged.connect(
            lambda *_: (setattr(self.s.screen, "device", self.cmb_ocr_dev.currentData()), self._save_timer.start()))
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_models.clicked.connect(self._models_clicked)
        self.btn_file.clicked.connect(self._choose_file)
        self.btn_clear.clicked.connect(self.feed_model.clear)
        self.cmb_source.currentIndexChanged.connect(self._source_changed)
        self.cmb_device.currentIndexChanged.connect(self._ui_to_settings)
        self.cmb_target.currentIndexChanged.connect(self._ui_to_settings)
        self.cmb_mt.currentIndexChanged.connect(self._ui_to_settings)
        self.cmb_voice.currentIndexChanged.connect(self._ui_to_settings)
        self.cmb_single.currentIndexChanged.connect(self._ui_to_settings)
        self.btn_import_voice.clicked.connect(self._import_voice)
        sp_ = self.speakers
        sp_.voice_changed.connect(self._voice_assigned)
        for c in (self.chk_speakers, self.chk_split, self.chk_translate, self.chk_partials, self.chk_tts, self.chk_simul,
                  self.chk_orig, self.chk_learn):
            c.toggled.connect(self._ui_to_settings)
        self.chk_overlay.toggled.connect(self._toggle_overlay)
        sp = self.speakers
        sp.renamed.connect(self._rename)
        sp.save_print.connect(self._save_print)
        sp.forget_print.connect(self._forget_print)
        sp.merge.connect(self._merge)
        self.feed_model.changed.connect(lambda: self._overlay_timer.start())

    # -- settings <-> ui -----------------------------------------------------------------------------
    def _populate_devices(self):
        try:
            self.devices = audio_mod.list_devices()
        except Exception as exc:
            self.devices = {"loopback": [], "input": [], "output": []}
            self._on_status(f"Audio device scan failed: {exc}")
        self._fill_device_combo()

    def _fill_device_combo(self):
        kind = self.cmb_source.currentData() or "loopback"
        self.cmb_device.blockSignals(True)
        self.cmb_device.clear()
        if kind == "file":
            self.cmb_device.addItem(self.s.audio.file_path or "(no file chosen)", self.s.audio.file_path)
        else:
            self.cmb_device.addItem("Default device", "")
            for d in self.devices.get(kind, []):
                label = audio_mod._strip_loopback(d.name) if kind == "loopback" else d.name
                self.cmb_device.addItem(label, d.name)
            i = self.cmb_device.findData(self.s.audio.device)
            if i < 0 and self.s.audio.device:
                i = self.cmb_device.findText(audio_mod._strip_loopback(self.s.audio.device))
            self.cmb_device.setCurrentIndex(max(0, i))
        self.btn_file.setVisible(kind == "file")
        self.cmb_device.blockSignals(False)

    def _apply_settings_to_ui(self):
        s = self.s
        widgets = (self.cmb_source, self.cmb_target, self.cmb_mt, self.cmb_voice, self.chk_speakers, self.chk_split,
                   self.chk_translate, self.chk_partials, self.chk_tts, self.chk_simul, self.chk_orig, self.chk_learn)
        for w in widgets:
            w.blockSignals(True)
        self.cmb_source.setCurrentIndex(max(0, self.cmb_source.findData(s.audio.source)))
        self.cmb_target.setCurrentIndex(max(0, self.cmb_target.findData(s.translation.target)))
        self.cmb_mt.setCurrentIndex(max(0, self.cmb_mt.findData(s.translation.model)))
        self.cmb_voice.setCurrentIndex(max(0, self.cmb_voice.findData(s.tts.voice_mode)))
        self.cmb_single.blockSignals(True)
        self.cmb_single.clear()
        for v in self.pipe.voices.library:
            self.cmb_single.addItem(v.name, v.id)
        self.cmb_single.setCurrentIndex(max(0, self.cmb_single.findData(s.tts.single_voice)))
        self.cmb_single.setVisible(s.tts.voice_mode == "single")
        self.cmb_single.blockSignals(False)
        self.speakers.set_voices(self.pipe.voices.library, s.tts.voice_mode in ("manual", "own"))
        o = s.overlay
        for w in (self.cmb_pin, self.cmb_vsize, self.cmb_vlines, self.chk_fade):
            w.blockSignals(True)
        self.cmb_pin.setCurrentIndex(max(0, self.cmb_pin.findData(o.origin_mode)))
        self.cmb_vsize.setCurrentIndex(min(range(self.cmb_vsize.count()),
                                           key=lambda i: abs(self.cmb_vsize.itemData(i) - o.width_meters)))
        self.cmb_vlines.setCurrentIndex(max(0, self.cmb_vlines.findData(o.lines)))
        self.chk_fade.setChecked(o.look_at_fade_enabled)
        for w in (self.cmb_pin, self.cmb_vsize, self.cmb_vlines, self.chk_fade):
            w.blockSignals(False)
        self.btn_place.setEnabled(o.origin_mode == "world")
        self.cmb_ocr_engine.blockSignals(True)
        self.cmb_ocr_engine.setCurrentIndex(max(0, self.cmb_ocr_engine.findData(s.screen.engine)))
        self.cmb_ocr_engine.blockSignals(False)
        self.cmb_ocr_dev.setEnabled(s.screen.engine == "rapidocr")
        self.cmb_screen_mode.blockSignals(True)
        self.cmb_screen_mode.setCurrentIndex(max(0, self.cmb_screen_mode.findData(s.screen.mode)))
        self.cmb_screen_mode.blockSignals(False)
        self.cmb_ocr_dev.blockSignals(True)
        self.cmb_ocr_dev.setCurrentIndex(max(0, self.cmb_ocr_dev.findData(s.screen.device)))
        self.cmb_ocr_dev.blockSignals(False)
        self.chk_speakers.setChecked(s.speakers.enabled)
        self.chk_split.setChecked(s.speakers.overlap_split)
        self.chk_learn.setChecked(s.speakers.learn_voiceprints)
        self.chk_translate.setChecked(s.translation.enabled)
        self.chk_partials.setChecked(s.asr.partials)
        self.chk_tts.setChecked(s.tts.enabled)
        self.chk_simul.setChecked(s.tts.simultaneous)
        self.chk_orig.setChecked(s.ui.show_original)
        self.chk_overlay.blockSignals(True)
        self.chk_overlay.setChecked(s.overlay.enabled)
        self.chk_overlay.blockSignals(False)
        for w in widgets:
            w.blockSignals(False)
        self._fill_device_combo()

    def _ui_to_settings(self, *_):
        s = self.s
        s.audio.source = self.cmb_source.currentData()
        if s.audio.source != "file":
            s.audio.device = self.cmb_device.currentData() or ""
        s.translation.target = self.cmb_target.currentData()
        if s.translation.model != self.cmb_mt.currentData():
            s.translation.model = self.cmb_mt.currentData()
            if self.pipe.models_ready:
                model = s.translation.model
                threading.Thread(target=lambda: self._switch_mt(model), daemon=True).start()
        s.tts.voice_mode = self.cmb_voice.currentData()
        s.tts.single_voice = self.cmb_single.currentData() or ""
        self.cmb_single.setVisible(s.tts.voice_mode == "single")
        self.speakers.set_voices(self.pipe.voices.library, s.tts.voice_mode in ("manual", "own"))
        s.speakers.enabled = self.chk_speakers.isChecked()
        s.speakers.overlap_split = self.chk_split.isChecked()
        s.speakers.learn_voiceprints = self.chk_learn.isChecked()
        s.translation.enabled = self.chk_translate.isChecked()
        s.asr.partials = self.chk_partials.isChecked()
        s.tts.enabled = self.chk_tts.isChecked()
        s.tts.simultaneous = self.chk_simul.isChecked()
        if s.ui.show_original != self.chk_orig.isChecked():
            s.ui.show_original = self.chk_orig.isChecked()
            s.overlay.show_original = s.ui.show_original
            self.delegate.show_original = s.ui.show_original
            self.feed.scheduleDelayedItemsLayout()
            self.feed.viewport().update()
        self._save_timer.start()

    def _switch_mt(self, model: str):
        try:
            info = self.pipe.hub.load(model, switch=True)
            self.bridge.status.emit(f"Translator switched to {info.get('model')}.")
        except Exception as exc:
            self.bridge.status.emit(f"Translator switch failed: {exc}")

    def _save(self):
        try:
            self.store.save(self.s)
        except Exception as exc:
            self._on_status(f"Could not save settings: {exc}")

    def _source_changed(self, *_):
        self._fill_device_combo()
        self._ui_to_settings()

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Test audio", self.s.audio.file_path or "",
                                              "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.webm)")
        if path:
            self.s.audio.file_path = path
            self._fill_device_combo()
            self._save_timer.start()

    # -- actions -----------------------------------------------------------------------------------
    def _models_clicked(self):
        if self.pipe.models_ready:
            self.pipe.unload_models()
            self.btn_models.setText("Load models")
            self._set_running(False)
        else:
            self._load_models_async()

    def _load_models_async(self, then_start: bool = False):
        if self._loading:
            return
        self._loading = True
        self.btn_models.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Loading…")
        self._prop(self.btn_start, state="loading")
        self._on_status("Loading speech models… (first start takes a few seconds)")
        self._sync_simple()

        def work():
            try:
                self.pipe.load_models()
                self.bridge.models_done.emit(then_start, "")
            except Exception as exc:
                self.bridge.models_done.emit(False, str(exc))
        threading.Thread(target=work, daemon=True).start()

    def _models_done(self, then_start: bool, error: str):
        self._loading = False
        self.btn_models.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Stop" if self.pipe.running else "Start")
        self._prop(self.btn_start, state="")
        self._sync_simple()
        if error:
            self._on_status(f"Model load failed: {error}")
            return
        self.btn_models.setText("Unload models")
        if then_start:
            self._start_clicked()

    def _start_clicked(self):
        if self.pipe.running:
            self.pipe.stop()
            self._set_running(False)
            return
        if not self.pipe.models_ready:
            self._load_models_async(then_start=True)
            return
        self._ui_to_settings()
        try:
            self.pipe.start()
        except Exception as exc:
            self._on_status(f"Start failed: {exc}")
            return
        self._set_running(True)

    def _set_running(self, running: bool):
        QTimer.singleShot(0, self._sync_simple)
        self.btn_start.setText("● Listening — Stop" if running else "Start")
        self._prop(self.btn_start, running="true" if running else "false")
        self._on_status("Listening." if running else "Stopped.")
        if running:
            self._meter.start()
        else:
            self._meter.stop()
            self.s_start.setStyleSheet("")
            self.meter.setValue(0)

    def _update_meter(self):
        self.meter.setValue(int(min(1.0, self.pipe.level * 1.6) * 100))
        if self.pipe.running:      # the big Start button breathes while listening, brighter when it hears speech
            import math
            a = 0.5 + 0.5 * math.sin(time.monotonic() * 3.2)
            lv = min(1.0, self.pipe.level * 2.5)
            v = min(1.0, 0.42 + 0.14 * a + 0.3 * lv)
            from PyQt6.QtGui import QColor
            c = QColor.fromHsvF(0.40, 0.68, v).name()
            self.s_start.setStyleSheet(f"QPushButton#start {{ background:{c}; }}")
        elif self.s_start.styleSheet():
            self.s_start.setStyleSheet("")
        src = self.pipe.source
        if self.pipe.running and src is not None and not src.is_alive() and self.s.audio.source == "file":
            rb = self.pipe.readback
            speaking = rb is not None and (rb.busy or not rb.q.empty())
            if self.pipe.idle() and not speaking:      # wait for translations and read-aloud to finish
                self._idle_ticks = getattr(self, "_idle_ticks", 0) + 1
                if self._idle_ticks >= 15:               # 1.5 s of nothing left to do
                    self._idle_ticks = 0
                    self.pipe.stop()
                    self._set_running(False)
                    if self.exit_when_done:
                        QTimer.singleShot(2500, self.close)
            else:
                self._idle_ticks = 0

    # -- pipeline callbacks (UI thread) --------------------------------------------------------------
    def _on_line(self, line):
        self.feed_model.upsert(line)

    def _speaker_names(self):
        return {sid: (sp.label, sp.color) for sid, sp in self.pipe.spk.speakers.items()}

    def _on_speakers(self):
        self.feed_model.refresh_names(self._speaker_names())
        sps = list(self.pipe.spk.speakers.values())
        voices = {}
        for sp in sps:
            key = sp.name or sp.id
            voices[sp.id] = self.s.tts.manual.get(key, "")
        self.speakers.update_speakers(sps, voices)

    def _on_status(self, msg: str):
        first = msg.splitlines()[0] if msg else ""
        if hasattr(self, "pills"):
            self.pills.on_status(first)
        self.lbl_status.setText(first[:180])
        if hasattr(self, "s_status"):
            self.s_status.setText(first[:120])
        stamp = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.log.appendPlainText(stamp)
        try:
            from ..settings import APP_DIR
            path = APP_DIR / "app.log"
            if path.exists() and path.stat().st_size > 2_000_000:
                path.replace(path.with_suffix(".old.log"))
            with open(path, "a", encoding="utf-8") as f:
                f.write(stamp + "\n")
        except OSError:
            pass

    def _on_stats(self, st: dict):
        self.lbl_stats.setText(f"speech→text {st.get('lag_ms', 0):.0f} ms · asr {st.get('asr_ms', 0):.0f} ms · "
                               f"mt {st.get('mt_ms', 0):.0f} ms · lines {st.get('lines', 0)} · "
                               f"backlog {st.get('backlog', 0)}")

    def _import_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import a voice clip", "",
                                              "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if not path:
            return
        from PyQt6.QtWidgets import QInputDialog
        import os
        name, ok = QInputDialog.getText(self, "Voice name", "Name for this voice:",
                                        text=os.path.splitext(os.path.basename(path))[0])
        if not ok or not name.strip():
            return
        text, _ = QInputDialog.getText(self, "Transcript (optional)",
                                       "What is said in the clip? (optional, improves cloning)")
        from ..core.voices import import_voice
        v = import_voice(path, name.strip(), text.strip())
        self.pipe.voices.reload()
        self._apply_settings_to_ui()
        self._on_speakers()
        self._on_status(f"Imported voice '{v.name}'. Pick it for a speaker in the Read-aloud voice column.")

    def _voice_assigned(self, sid, voice_id):
        sp = self.pipe.spk.speakers.get(sid)
        self.pipe.voices.set_manual(sp.name if sp and sp.name else sid, voice_id)
        self._save_timer.start()

    def _rename(self, sid, name):
        self.pipe.spk.rename(sid, name)
        self._on_speakers()
        self._push_overlay()

    def _save_print(self, sid):
        sp = self.pipe.spk.speakers.get(sid)
        if sp and not sp.name:
            self._on_status("Give the speaker a name first (double-click the name).")
            return
        ok = self.pipe.spk.save_voiceprint(sid)
        self._on_status(f"Voiceprint saved for {sp.label}." if ok else "Not enough audio for a voiceprint yet.")
        self._on_speakers()

    def _forget_print(self, sid):
        self.pipe.spk.forget_voiceprint(sid)
        self._on_speakers()

    def _merge(self, keep, drop):
        self.pipe.spk.merge(keep, drop)
        for ln in self.feed_model.rows:
            if ln.speaker_id == drop:
                ln.speaker_id = keep
        self._on_speakers()

    # -- players / VR overlay -------------------------------------------------------------------------
    def _on_players(self, names, world, me):
        self.players_list = list(names)
        self.speakers.set_players(names, world)
        self.overlay_panel.set_players(names)

    # -- one-tap speaker naming (desktop) ------------------------------------------------------------
    def _feed_clicked(self, index):
        ln = index.data(Qt.ItemDataRole.UserRole)
        if ln is None or ln.source != "voice" or not ln.speaker_id:
            return
        self._name_menu(ln.speaker_id)

    def _name_menu(self, sid: str):
        from PyQt6.QtGui import QCursor
        from PyQt6.QtWidgets import QInputDialog, QMenu
        sp = self.pipe.spk.speakers.get(sid)
        if sp is None:
            return
        m = QMenu(self)
        m.setStyleSheet("QMenu{font-size:15pt;background:#1b1e2b;color:#e6e9f5;border:1px solid #30365a;}"
                        "QMenu::item{padding:10px 28px;} QMenu::item:selected{background:#3d59a1;}")
        head = m.addAction(f"Who is {sp.label}?")
        head.setEnabled(False)
        m.addSeparator()
        taken = {o.name for o in self.pipe.spk.speakers.values() if o.id != sid and o.name}
        for name in getattr(self, "players_list", []):
            a = m.addAction(name + ("   (already used)" if name in taken else ""))
            a.setData(name)
        if not getattr(self, "players_list", []):
            na = m.addAction("(no players found in the VRChat log)")
            na.setEnabled(False)
        m.addSeparator()
        typed = m.addAction("Type a name…")
        clear = m.addAction("Clear name")
        act = m.exec(QCursor.pos())
        if act is None:
            return
        if act is typed:
            name, ok = QInputDialog.getText(self, "Name this speaker", f"Name for {sp.label}:", text=sp.name)
            if ok:
                self._rename(sid, name.strip())
        elif act is clear:
            self._rename(sid, "")
        elif act.data():
            self._rename(sid, str(act.data()))

    def _vr_name_picked(self, sid: str, name: str):
        self._rename(sid, name)
        self._on_status(f"Named {sid} '{name}' from VR." if name else f"Cleared name for {sid}.")

    def _overlay_cfg_changed(self, *_):
        o = self.s.overlay
        o.origin_mode = self.cmb_pin.currentData()
        o.width_meters = float(self.cmb_vsize.currentData())
        o.lines = int(self.cmb_vlines.currentData())
        o.look_at_fade_enabled = self.chk_fade.isChecked()
        if o.origin_mode == "right_wrist":
            o.x_offset, o.y_offset, o.z_offset, o.pitch, o.yaw, o.roll = 0.0, 0.03, 0.10, -45.0, 0.0, 0.0
        elif o.origin_mode == "left_wrist":
            o.x_offset, o.y_offset, o.z_offset, o.pitch, o.yaw, o.roll = 0.0, 0.03, 0.10, -45.0, 0.0, 0.0
        elif o.origin_mode == "head":
            o.x_offset, o.y_offset, o.z_offset, o.pitch, o.yaw, o.roll = 0.0, -0.22, -0.9, 0.0, 0.0, 0.0
        self.btn_place.setEnabled(o.origin_mode == "world")
        if self.overlay_service:
            self.overlay_service.update_settings(o)
        self._push_overlay()
        self._save_timer.start()

    def _place_overlay(self):
        system = self._vr_system()
        if system is None:
            self._on_status("Turn on the VR overlay (SteamVR running) first.")
            return
        try:
            import math
            import openvr
            pose = system.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, 1)[0]
            m = pose.mDeviceToAbsoluteTracking.m
            pos = [m[0][3], m[1][3], m[2][3]]
            fwd = [-m[0][2], 0.0, -m[2][2]]
            n = math.hypot(fwd[0], fwd[2]) or 1.0
            fwd = [fwd[0] / n, 0.0, fwd[2] / n]
            o = self.s.overlay
            o.world_x, o.world_y, o.world_z = pos[0] + fwd[0] * 0.8, pos[1] - 0.15, pos[2] + fwd[2] * 0.8
            o.world_yaw = math.degrees(math.atan2(-fwd[0], -fwd[2]))
            o.world_pitch = o.world_roll = 0.0
            self.overlay_service.update_settings(o)
            self._save_timer.start()
            self._on_status("VR panel placed in front of you.")
        except Exception as exc:
            self._on_status(f"Could not place panel: {exc}")

    # -- VR overlay ----------------------------------------------------------------------------------
    def _toggle_overlay(self, on: bool):
        self.s.overlay.enabled = on
        self.s.overlay.interactive = True       # laser taps on names rename speakers
        self._save_timer.start()
        if on:
            try:
                from .vr_overlay import OpenVrOverlayService
                if self.overlay_service is None:
                    self.overlay_service = OpenVrOverlayService(self.overlay_panel, self.s.overlay)
                    self.overlay_service.status_changed.connect(self._on_status)
                if not self.overlay_service.start():
                    self.chk_overlay.setChecked(False)
                self._push_overlay()
            except Exception as exc:
                self._on_status(f"VR overlay unavailable: {exc}")
        elif self.overlay_service:
            self.overlay_service.stop()

    def _push_overlay(self):
        if not self.overlay_service or not self.overlay_service.is_active():
            return
        names = self._speaker_names()
        lines = []
        for ln in self.feed_model.recent(self.s.overlay.lines):
            if ln.source == "voice":
                name, color = names.get(ln.speaker_id, ("…", "#8089a8"))
            elif ln.source == "screen":
                name, color = "Screen", "#e0af68"
            elif ln.source == "chatbox":
                name, color = f"{ln.speaker_id} · chatbox", "#73daca"
            else:
                name, color = f"{ln.speaker_id} · chat", "#c0caf5"
            lines.append((name, color, ln.text, ln.translation, ln.speaker_id if ln.source == "voice" else ""))
        self.overlay_panel.set_lines(lines)
        self.overlay_service.mark_dirty()

    # -- screen translation ------------------------------------------------------------------------
    def _ocr_clicked(self):
        if self.screen.loaded:
            self.screen.unload()
            self.btn_ocr.setText("Load screen OCR")
            self.chk_live.setChecked(False)
            self._vr_timer.stop()
            return
        self.btn_ocr.setEnabled(False)
        self._on_status("Loading screen OCR…")

        def work():
            try:
                self.screen.load()
            except Exception as exc:
                self.bridge.status.emit(f"Screen OCR load failed: {exc}")
            finally:
                QTimer.singleShot(0, self._ocr_loaded_ui)
        threading.Thread(target=work, daemon=True).start()

    def _ocr_loaded_ui(self):
        self.btn_ocr.setEnabled(True)
        if self.screen.loaded:
            self.btn_ocr.setText("Unload screen OCR")
            self._vr_timer.start()

    def _snap(self):
        self._on_status("Reading screen…")
        self.screen.snapshot_async()

    def _live_toggled(self, on):
        if on and not self.screen.loaded:
            self._ocr_clicked()
        self.screen.set_live(on)

    def _screen_engine_changed(self, *_):
        new = self.cmb_ocr_engine.currentData()
        if new == self.s.screen.engine:
            return
        was = self.screen.loaded
        if was:
            self.screen.unload()
        self.s.screen.engine = new
        self.cmb_ocr_dev.setEnabled(new == "rapidocr")
        self._save_timer.start()
        if was:
            self._ocr_clicked()

    def _screen_mode_changed(self, *_):
        self.s.screen.mode = self.cmb_screen_mode.currentData()
        self._sync_simple()
        self._save_timer.start()

    def _toggle_chatbox_reader(self):
        on = not (self.chk_live.isChecked() and self.s.screen.mode == "chatbox")
        self.cmb_screen_mode.setCurrentIndex(self.cmb_screen_mode.findData("chatbox"))
        self.chk_live.setChecked(on)
        self._sync_simple()

    def _on_chat_result(self, msgs, info):
        import hashlib
        for name, text, tr in msgs:
            lid = "cbx-" + hashlib.md5(f"{name}|{text}".encode("utf-8")).hexdigest()[:10]
            ln = Line(id=lid, source="chatbox", speaker_id=name, text=text, translation=tr, final=True)
            ln.timings = {"asr": info.get("ocr_ms", 0), "mt": info.get("mt_ms", 0)}
            self.feed_model.upsert(ln)
            if self.s.tts.enabled and self.s.screen.read_chat and tr:
                self.pipe.speak_chat(name, tr if tr else text)
        if msgs or not info.get("live"):
            self._on_status(f"Chatbox: {len(msgs)} new of {info.get('bubbles', 0)} bubbles · OCR {info.get('ocr_ms')} ms"
                            f" · translate {info.get('mt_ms')} ms")

    def _on_screen_result(self, pairs, info):
        import hashlib
        for orig, tr in pairs:
            lid = "scr-" + hashlib.md5(orig.encode("utf-8")).hexdigest()[:10]
            ln = Line(id=lid, source="screen", speaker_id="Screen", text=orig, translation=tr, final=True)
            ln.timings = {"asr": info.get("ocr_ms", 0), "mt": info.get("mt_ms", 0)}
            if lid in self.feed_model.index_of:
                if info.get("live"):
                    continue          # already on screen; live mode only adds new text
                gone = Line(id=lid, source="screen", speaker_id="Screen", dropped="moved")
                self.feed_model.upsert(gone)   # snapshot: move it to the bottom
            self.feed_model.upsert(ln)
        self._on_status(f"Screen: {len(pairs)} text blocks · OCR {info.get('ocr_ms')} ms · "
                        f"translate {info.get('mt_ms')} ms ({info.get('src')})")

    def _start_hotkey(self):
        key = (self.s.screen.hotkey or "").lower()
        if not key:
            return
        try:
            from pynput import keyboard
            target = getattr(keyboard.Key, key, None)

            def on_press(k):
                if k == target:
                    self.bridge.hotkey.emit()
            self._kb = keyboard.Listener(on_press=on_press, daemon=True)
            self._kb.start()
        except Exception as exc:
            self._on_status(f"Hotkey unavailable: {exc}")

    def _vr_system(self):
        if not self.overlay_service or not self.overlay_service.is_active():
            return None
        return self.overlay_service._vr_system

    def _vr_poll(self):
        system = self._vr_system()
        if system is None or not self.s.screen.vr_chord:
            return
        try:
            import openvr
            idx = system.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_LeftHand)
            if idx == openvr.k_unTrackedDeviceIndexInvalid:
                return
            got, state = system.getControllerState(idx)
            down = bool(got) and bool(state.ulButtonPressed & (1 << 2)) and bool(state.ulButtonPressed & (1 << 1))
        except Exception:
            return
        if down and not self._chord_prev:
            self._snap()
        self._chord_prev = down

    def _head_turning(self) -> bool:
        system = self._vr_system()
        if system is None:
            return False
        try:
            import openvr
            pose = system.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, 1)[0]
            w = pose.vAngularVelocity.v
            return (w[0] ** 2 + w[1] ** 2 + w[2] ** 2) ** 0.5 > 1.0   # rad/s (~57 deg/s)
        except Exception:
            return False

    def closeEvent(self, ev):
        try:
            self.screen.unload()
            self.players.stop()
            self.pipe.stop()
            self.intake.stop()
            if self.overlay_service:
                self.overlay_service.stop()
            self._save()
        finally:
            super().closeEvent(ev)
