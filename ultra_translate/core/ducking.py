"""Duck the game's own audio while the translated voice speaks, without starving the translator.

Windows per-app volume (the Volume Mixer slider) scales VRChat's stream BEFORE it is mixed, so the loopback the
translator listens to gets quieter too. That scale is exact digital gain in the float mixer, so the capture
multiplies by 1/level while ducked and the translator hears the original level. Touches only the per-app volume
of the target process (VRChat.exe), never the device volume, and always restores what was there before.
"""
from __future__ import annotations

import queue
import threading
import time


class GameDucker(threading.Thread):
    def __init__(self, process: str = "VRChat.exe", level: float = 0.3, log=print):
        super().__init__(daemon=True, name="ducker")
        self.process = process.lower()
        self.level = level                 # volume while ducked (fraction of the user's own setting)
        self.log = log
        self.want = 0.0                    # 0 = normal, 1 = ducked
        self.cur = 0.0
        self.comp = 1.0                    # capture gain compensation the audio source applies
        self.q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self._orig: dict = {}              # session -> user's volume before we touched it
        self.ok = True
        self.active = lambda: False        # polled: duck while this is True

    def duck(self, on: bool) -> None:
        self.want = 1.0 if on else 0.0

    def run(self) -> None:
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception as exc:
            self.ok = False
            self.log(f"Game ducking unavailable: {exc}")
            return
        last_scan = 0.0
        sessions = []
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                if now - last_scan > 3.0:
                    sessions = self._sessions()
                    last_scan = now
                self.want = 1.0 if self.active() else 0.0
                # fade: down in ~150 ms, back up in ~500 ms
                step = 0.34 if self.want > self.cur else 0.1
                if abs(self.want - self.cur) > 1e-3:
                    self.cur = min(self.want, self.cur + step) if self.want > self.cur else max(self.want, self.cur - step)
                    self._apply(sessions)
                time.sleep(0.05)
        finally:
            self.want = self.cur = 0.0
            self._apply(self._sessions(), restore=True)
            try:
                import comtypes
                comtypes.CoUninitialize()
            except Exception:
                pass

    def stop(self) -> None:
        self.stop_event.set()

    def _sessions(self):
        try:
            from pycaw.pycaw import AudioUtilities
            out = []
            for s in AudioUtilities.GetAllSessions():
                if s.Process is not None and s.Process.name().lower() == self.process:
                    out.append(s)
            return out
        except Exception:
            return []

    def _apply(self, sessions, restore: bool = False) -> None:
        factor = 1.0 - self.cur * (1.0 - self.level)
        for s in sessions:
            try:
                vol = s.SimpleAudioVolume
                key = s.ProcessId if hasattr(s, "ProcessId") else id(s)
                if key not in self._orig:
                    self._orig[key] = vol.GetMasterVolume()
                vol.SetMasterVolume(max(0.0, min(1.0, self._orig[key] * factor)), None)
                if restore or self.cur == 0.0:
                    vol.SetMasterVolume(self._orig.pop(key), None)
            except Exception:
                continue
        self.comp = 1.0 / max(0.05, factor) if sessions and not restore else 1.0
