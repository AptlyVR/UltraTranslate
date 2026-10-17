"""Settings: plain dataclasses, tolerant load (unknown/missing keys never wipe the file), atomic save."""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APP_DIR = Path(os.environ.get("APPDATA", Path.home())) / "UltraTranslate"
SETTINGS_PATH = APP_DIR / "settings.json"
SPEAKERS_PATH = APP_DIR / "voiceprints.json"
HISTORY_DIR = APP_DIR / "history"


@dataclass
class AudioSettings:
    source: str = "loopback"          # loopback | input | file
    device: str = ""                  # device name; "" = default output (loopback) / default mic
    file_path: str = ""               # for source == file (also played to monitor_device)
    monitor_file: bool = True         # play file audio out while processing it
    gain: float = 1.0


@dataclass
class VadSettings:
    start_prob: float = 0.55
    end_prob: float = 0.35
    end_silence_ms: int = 450          # pause that always ends a line
    check_ms: int = 96                 # pause length where Smart Turn + speculative ASR run
    smart_turn: bool = True            # commit confident sentence ends early (~112 ms)
    smart_threshold: float = 0.85
    preroll_ms: int = 400
    min_speech_ms: int = 250
    soft_max_s: float = 6.0           # look for a pause to split after this
    hard_max_s: float = 10.0          # continuous speech/noise (no pause): cut at the quietest point
    partial_interval_ms: int = 600


@dataclass
class AsrSettings:
    model: str = "large-v3-turbo"     # large-v3-turbo | large-v3
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1               # measured: beam 2/3 = same text, +20-40 ms
    partials: bool = True
    fixed_language: str = ""          # "" = per-speaker automatic
    background_gate: bool = True      # drop quiet, low-confidence distant chatter


@dataclass
class SpeakerSettings:
    enabled: bool = True
    overlap_split: bool = True        # split segments where the local diarizer sees a speaker change
    session_threshold: float = 0.32   # cosine to join a session speaker (ECAPA; measured diff p99 0.23)
    change_dip: float = 0.20          # 1.5 s neighbour windows below this = candidate speaker change
    change_confirm: float = 0.30      # 3 s left/right below this = confirmed change (same p1 0.35)
    identity_threshold: float = 0.50  # cosine to auto-name from a saved voiceprint
    learn_voiceprints: bool = False   # auto-refine saved voiceprints from confident matches
    max_speakers: int = 12
    new_min_s: float = 0.9            # a new person is only founded from a turn at least this long
    merge_threshold: float = 0.50     # two speakers whose voice centroids are this similar are one person
    minor_total_s: float = 3.0        # a label with less total talk than this after minor_age_s of speech joins
    minor_age_s: float = 30.0         # the closest voice (total-talk rule; replay 11.5 -> 10.6 %)
    fuse_redimnet: bool = True        # decide "who is this" on ECAPA + ReDimNet-b5 together (replay: 12.7 -> 11.5 %)
    fused_session_threshold: float = 0.36
    fused_merge_threshold: float = 0.35
    device: str = "auto"              # auto | cuda | cpu (auto = cpu when VRAM is tight)


@dataclass
class TranslationSettings:
    enabled: bool = True
    target: str = "en"
    hub_url: str = "http://127.0.0.1:8097"
    model: str = "best"               # best (Hy-MT2 7B) | fast (1.8B)
    context_lines: int = 2
    live: bool = True                 # translate the live caption while they are still talking (display only)
    translate_chat: bool = True


@dataclass
class TtsSettings:
    enabled: bool = False
    url: str = "http://127.0.0.1:8090/v1/tts"   # VoxCPM2 TTS server
    output_device: str = ""
    volume: float = 0.9
    voice_mode: str = "own"           # own (clone each speaker; a per-speaker pick overrides) | single | auto | manual
    single_voice: str = ""            # voice id for single mode ("" = first in pool)
    voice_pool: list = field(default_factory=list)   # voice ids auto mode may use ([] = all)
    manual: dict = field(default_factory=dict)       # player name or speaker id -> voice id
    read_chat: bool = True
    max_wait_s: float = 8.0           # drop lines that waited longer than this
    skip_own_language: bool = True    # don't read lines already in the target language
    simultaneous: bool = False        # optional interpreter mode: speak clauses while they talk (rougher, sooner)
    interpreter_policy: str = "agree" # agree = LocalAgreement-2 with prefix-forced re-translation | chunks (older)
    duck_game: bool = True            # turn VRChat down while the translation speaks (capture is boosted back)
    duck_level: float = 0.3           # VRChat volume while ducked, fraction of your own setting
    echo_cancel: bool = True          # remove our own read-aloud voice from the loopback the translator hears


@dataclass
class IntakeSettings:
    vrcx_ws: bool = True
    vrcx_ws_url: str = "ws://127.0.0.1:34582/"
    vrcx_sqlite: bool = True
    vrcx_sqlite_path: str = ""


@dataclass
class OverlaySettings:
    enabled: bool = False
    origin_mode: str = "left_wrist"   # left_wrist | right_wrist | head | world
    width_meters: float = 0.26
    opacity: float = 0.95
    x_offset: float = 0.0
    y_offset: float = 0.03
    z_offset: float = 0.10
    pitch: float = -45.0
    yaw: float = 0.0
    roll: float = 0.0
    world_x: float = 0.0
    world_y: float = 1.5
    world_z: float = -1.0
    world_pitch: float = 0.0
    world_yaw: float = 0.0
    world_roll: float = 0.0
    follow_head_position_x: bool = False
    follow_head_position_y: bool = False
    follow_head_position_z: bool = False
    follow_head_rotation_pitch: bool = False
    follow_head_rotation_yaw: bool = False
    follow_head_rotation_roll: bool = False
    look_at_fade_enabled: bool = False
    look_at_distance: float = 0.0
    look_at_sensitivity: float = 0.4
    look_at_min_opacity: float = 0.15
    hide_when_not_looked_at: bool = False
    interactive: bool = True          # point the laser at a name on the panel to rename that speaker
    lines: int = 6
    font_px: int = 22
    show_original: bool = True


@dataclass
class ScreenSettings:
    python: str = ""                  # OCR venv python; "" = this python
    device: str = "cuda"              # cuda (~0.5 GB VRAM) | cpu (no VRAM)
    window: str = "VRChat"            # capture this window's client area; "" = primary monitor
    crop: float = 1.0                 # keep the central fraction of the capture
    model: str = "fast"               # TranslateHub model for screen text
    live_fps: float = 1.5
    hotkey: str = "f8"
    vr_chord: bool = True             # hold left grip + press left B
    mode: str = "chatbox"             # chatbox (players' chat bubbles + who wrote them) | text (all screen text)
    engine: str = "vlm"               # vlm (Qwen3-VL-4B: 24/24 bubbles, 0.6% CER, ~0.7 s, 5.5 GB) | rapidocr (0.2 s, <1 GB)
    read_chat: bool = True            # read chatbox messages aloud (if read-aloud is on)


@dataclass
class UiSettings:
    font_scale: float = 1.0
    show_original: bool = True
    max_feed_items: int = 400
    window_geometry: str = ""
    mode: str = "simple"              # simple (big VR-friendly controls) | advanced


@dataclass
class Settings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    asr: AsrSettings = field(default_factory=AsrSettings)
    speakers: SpeakerSettings = field(default_factory=SpeakerSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    intake: IntakeSettings = field(default_factory=IntakeSettings)
    overlay: OverlaySettings = field(default_factory=OverlaySettings)
    screen: ScreenSettings = field(default_factory=ScreenSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    version: int = 2


def _fill(cls, data: Any):
    """Build dataclass `cls` from dict, ignoring unknown keys and keeping defaults for bad values."""
    obj = cls()
    if not isinstance(data, dict):
        return obj
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        current = getattr(obj, f.name)
        value = data[f.name]
        if dataclasses.is_dataclass(current):
            setattr(obj, f.name, _fill(type(current), value))
        elif isinstance(current, bool):
            if isinstance(value, bool):
                setattr(obj, f.name, value)
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(obj, f.name, type(current)(value))
        elif isinstance(current, str):
            if isinstance(value, str):
                setattr(obj, f.name, value)
        else:
            setattr(obj, f.name, value)
    return obj


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH):
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> Settings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Settings()
        except (OSError, ValueError):
            # Corrupt file: keep it for inspection, never silently overwrite it.
            try:
                self.path.replace(self.path.with_suffix(".corrupt.json"))
            except OSError:
                pass
            return Settings()
        if isinstance(raw, dict) and raw.get("version") != 2:
            return _migrate_v1(raw)
        return _fill(Settings, raw)

    def save(self, settings: Settings) -> None:
        with self._lock:
            atomic_write_json(self.path, dataclasses.asdict(settings))


def _migrate_v1(raw: dict) -> Settings:
    """Carry over the few legacy values that still mean the same thing."""
    s = Settings()
    try:
        audio = raw.get("audio") or {}
        dev = audio.get("capture_device_name") or audio.get("capture_device") or ""
        if isinstance(dev, str):
            s.audio.device = dev
        overlay = raw.get("overlay") or {}
        s.overlay = _fill(OverlaySettings, overlay)
        tts = raw.get("tts") or {}
        if isinstance(tts.get("output_device"), str):
            s.tts.output_device = tts["output_device"]
    except Exception:
        pass
    return s
