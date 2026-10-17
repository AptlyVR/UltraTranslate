"""Who is in the current VRChat instance, from VRChat's own output log (VRCX database as fallback).

Replays the newest log from the last world join, then tails it. Only complete lines are parsed.
on_change(players: list[str], world: str, me: str) is called whenever the roster changes.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from pathlib import Path

LOG_DIR = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "AppData" / "LocalLow" / "VRChat" / "VRChat"
VRCX_DB = Path.home() / "AppData" / "Roaming" / "VRCX" / "VRCX.sqlite3"

RE_JOIN = re.compile(r"\[Behaviour\] OnPlayerJoined (?P<name>.+?)(?: \((?P<uid>usr_[0-9a-f-]+)\))?\s*$")
RE_LEFT = re.compile(r"\[Behaviour\] OnPlayerLeft (?P<name>.+?)(?: \((?P<uid>usr_[0-9a-f-]+)\))?\s*$")
RE_ROOM = re.compile(r"\[Behaviour\] Entering Room: (?P<world>[^\r\n]+)")
RE_JOINING = re.compile(r"\[Behaviour\] Joining wrld_")
RE_ME = re.compile(r"User Authenticated: (?P<name>.+?) \(usr_")


class PlayerTracker:
    def __init__(self, on_change, on_status=lambda m: None):
        self.on_change = on_change
        self.on_status = on_status
        self.players: dict[str, str] = {}      # name -> user id
        self.world = ""
        self.me = ""
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._file: Path | None = None
        self._pos = 0
        self._partial = b""

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="vrc-players")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def names(self) -> list[str]:
        return sorted((n for n in self.players if n != self.me), key=str.lower)

    # ------------------------------------------------------------------------------------------
    def _latest(self) -> Path | None:
        try:
            files = list(LOG_DIR.glob("output_log_*.txt"))
        except OSError:
            return None
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def _emit(self) -> None:
        self.on_change(self.names(), self.world, self.me)

    def _handle(self, line: str) -> bool:
        if "[Behaviour]" not in line and "User Authenticated" not in line:
            return False
        m = RE_JOIN.search(line)
        if m:
            self.players[m["name"].strip()] = m["uid"] or ""
            return True
        m = RE_LEFT.search(line)
        if m:
            self.players.pop(m["name"].strip(), None)
            return True
        m = RE_ROOM.search(line)
        if m:
            self.world = m["world"]
            self.players.clear()
            return True
        if RE_JOINING.search(line):
            self.players.clear()
            return True
        m = RE_ME.search(line)
        if m and not self.me:
            self.me = m["name"].strip()
        return False

    def _replay(self, path: Path) -> None:
        data = path.read_bytes()
        # me: first authenticated line anywhere in the file
        m = RE_ME.search(data[:2_000_000].decode("utf-8", "replace"))
        if m:
            self.me = m["name"].strip()
        text = data.decode("utf-8", "replace")
        cut = max(text.rfind("[Behaviour] Entering Room:"), text.rfind("[Behaviour] Joining wrld_"))
        start = text.rfind("\n", 0, cut) + 1 if cut >= 0 else 0
        self.players.clear()
        rooms = list(RE_ROOM.finditer(text, max(0, start - 4000), start + 4000))
        if rooms:
            self.world = rooms[-1]["world"].strip()
        complete = text[start:text.rfind("\n") + 1] if "\n" in text[start:] else ""
        for line in complete.splitlines():
            self._handle(line)
        self._pos = len(data[:data.rfind(b"\n") + 1]) if b"\n" in data else 0
        self._partial = b""

    def _vrcx_fallback(self) -> bool:
        if not VRCX_DB.exists():
            return False
        try:
            with sqlite3.connect(f"file:{VRCX_DB.as_posix()}?mode=ro", uri=True, timeout=1) as con:
                rows = con.execute("SELECT type, display_name, location FROM gamelog_join_leave "
                                   "ORDER BY id DESC LIMIT 400").fetchall()
        except sqlite3.Error:
            return False
        if not rows:
            return False
        loc = rows[0][2]
        present: dict[str, str] = {}
        for typ, name, location in reversed([r for r in rows if r[2] == loc]):
            if typ == "OnPlayerJoined":
                present[name] = ""
            elif typ == "OnPlayerLeft":
                present.pop(name, None)
        self.players = present
        return True

    def _run(self) -> None:
        last_emit = None
        while not self.stop_event.is_set():
            try:
                path = self._latest()
                if path is None:
                    if self._vrcx_fallback():
                        snap = (tuple(self.names()), self.world)
                        if snap != last_emit:
                            last_emit = snap
                            self._emit()
                    else:
                        self.on_status("VRChat log not found; player list empty.")
                    self.stop_event.wait(10)
                    continue
                if path != self._file:
                    self._file = path
                    self._replay(path)
                    self.on_status(f"Player list: {len(self.names())} in {self.world or 'current world'}.")
                    last_emit = None
                size = path.stat().st_size
                if size < self._pos:          # truncated/rotated
                    self._replay(path)
                elif size > self._pos:
                    with path.open("rb") as f:
                        f.seek(self._pos)
                        chunk = f.read(size - self._pos)
                    self._pos = size
                    data = self._partial + chunk
                    nl = data.rfind(b"\n")
                    complete, self._partial = (data[:nl + 1], data[nl + 1:]) if nl >= 0 else (b"", data)
                    for line in complete.decode("utf-8", "replace").splitlines():
                        self._handle(line)
                snap = (tuple(self.names()), self.world)
                if snap != last_emit:
                    last_emit = snap
                    self._emit()
            except Exception as exc:
                self.on_status(f"Player list error: {exc}")
                self.stop_event.wait(5)
            self.stop_event.wait(1.0)
