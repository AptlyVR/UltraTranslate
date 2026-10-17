"""Read-aloud voice library and per-speaker assignment.

Voices: cloned voice references from an optional voice library JSON (one per voice folder)
plus designed voices (VoxCPM2 voice_design prompts, no reference needed).
Modes: single (everyone one voice) | auto (distinct voice per speaker from a pool) |
       manual (you pick per speaker; unpicked fall back to auto) | own (clone each speaker's own voice).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

CUSTOM_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "UltraTranslate" / "voices"
CUSTOM_INDEX = CUSTOM_DIR / "index.json"
# optional voice library: a JSON {"voices": {path: {"display_name", "text"}}, "current_voice_filepath": path}
VOICE_LIBRARY = Path(os.environ.get("ULTRATRANSLATE_VOICE_LIBRARY", str(Path.home() / ".ultratranslate_voices.json")))

DESIGNED = [
    ("Calm man", "a calm adult male voice, clear and neutral"),
    ("Bright woman", "a bright young female voice, friendly and clear"),
    ("Deep man", "a deep male voice, relaxed and warm"),
    ("Soft woman", "a soft female voice, gentle and calm"),
    ("Upbeat young man", "an energetic young male voice, upbeat"),
    ("Confident woman", "a mature female voice, confident and clear"),
    ("Casual man", "a light male voice, casual and quick"),
    ("Cheerful woman", "a warm female voice, cheerful"),
]


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    kind: str            # clone | design
    path: str = ""
    text: str = ""
    control: str = ""


def _clean(name: str) -> str:
    for junk in ("source_datasets", "approved_dataset", "clips"):
        name = name.replace(junk, "")
    return name.replace("_", " ").strip(" -") or "Voice"


def load_custom() -> list[Voice]:
    try:
        items = json.loads(CUSTOM_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [Voice(id="custom:" + i["path"], name=i["name"] + " (imported)", kind="clone", path=i["path"],
                  text=i.get("text", "")) for i in items if os.path.isfile(i.get("path", ""))]


def import_voice(src: str, name: str, text: str = "") -> Voice:
    """Copy an audio file into the app's voice folder and register it."""
    import shutil
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    dst = CUSTOM_DIR / Path(src).name
    if Path(src).resolve() != dst.resolve():
        shutil.copy2(src, dst)
    try:
        items = json.loads(CUSTOM_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        items = []
    items = [i for i in items if i.get("path") != str(dst)] + [{"name": name, "path": str(dst), "text": text}]
    from ..settings import atomic_write_json
    atomic_write_json(CUSTOM_INDEX, items)
    return Voice(id="custom:" + str(dst), name=name + " (imported)", kind="clone", path=str(dst), text=text)


def load_library() -> list[Voice]:
    voices: list[Voice] = load_custom()
    try:
        data = json.loads(VOICE_LIBRARY.read_text(encoding="utf-8"))
        entries = data.get("voices") or {}
        current = data.get("current_voice_filepath") or ""
    except (OSError, ValueError):
        entries, current = {}, ""
    groups: dict[str, tuple[str, dict]] = {}
    for path, e in entries.items():
        if not isinstance(e, dict) or not os.path.isfile(path):
            continue
        p = Path(path)
        folder = p.parent
        while folder.name.lower() in ("clips", "approved_dataset"):
            folder = folder.parent
        key = str(folder) if folder.name.lower() not in ("imports", "cloned_voices") else path
        if key not in groups or path == current:
            groups[key] = (path, e)
    for key, (path, e) in groups.items():
        p = Path(path)
        folder = Path(key)
        name = e.get("display_name") if key == path else folder.name
        if path == current:
            name = (e.get("display_name") or "My voice") + " (current)"
        text = e.get("text") or e.get("transcription") or e.get("transcript") or ""
        voices.append(Voice(id="clone:" + path, name=_clean(str(name))[:48], kind="clone", path=path, text=text))
    custom = [v for v in voices if v.id.startswith("custom:")]
    rest = sorted((v for v in voices if not v.id.startswith("custom:")),
                  key=lambda v: (not v.name.endswith("(current)"), v.name.lower()))
    voices = custom + rest
    voices += [Voice(id="design:" + n, name=n + " (designed)", kind="design", control=c) for n, c in DESIGNED]
    return voices


class VoiceAssigner:
    """Decides which voice reads each speaker's lines, per the current mode."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.library = load_library()
        self.by_id = {v.id: v for v in self.library}
        self.session: dict[str, str] = {}       # speaker id -> voice id (auto picks, this session)

    def reload(self):
        self.library = load_library()
        self.by_id = {v.id: v for v in self.library}

    def pool(self) -> list[Voice]:
        chosen = [self.by_id[i] for i in self.cfg.voice_pool if i in self.by_id]
        return chosen or [v for v in self.library if v.kind == "design"] + [v for v in self.library if v.kind == "clone"]

    def voice_for(self, speaker_id: str, speaker_name: str = "") -> Voice | None:
        mode = self.cfg.voice_mode
        if mode == "own":
            # each speaker in their own cloned voice, unless you picked a voice (e.g. an imported one) for them
            vid = self.cfg.manual.get(speaker_name) or self.cfg.manual.get(speaker_id)
            return self.by_id.get(vid) if vid else None
        if mode == "single":
            return self.by_id.get(self.cfg.single_voice) or self.pool()[0]
        if mode == "manual":
            vid = self.cfg.manual.get(speaker_name) or self.cfg.manual.get(speaker_id)
            if vid in self.by_id:
                return self.by_id[vid]
        vid = self.session.get(speaker_id)
        if vid not in self.by_id:
            used = set(self.session.values())
            pool = self.pool()
            free = [v for v in pool if v.id not in used]
            vid = free[0].id if free else pool[len(self.session) % len(pool)].id
            self.session[speaker_id] = vid
        return self.by_id[vid]

    def set_manual(self, key: str, voice_id: str) -> None:
        if voice_id:
            self.cfg.manual[key] = voice_id
        else:
            self.cfg.manual.pop(key, None)
