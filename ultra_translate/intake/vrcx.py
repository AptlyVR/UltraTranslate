"""VRChat typed chat (ChatBoxMessage) from VRCX: overlay websocket and/or the VRCX SQLite gamelog.

Both sources feed one de-duplicator keyed on (speaker, text) inside a 20 s window, so a message seen by
both is shown, translated and spoken once. Missing DB / closed VRCX are retried, never fatal.
"""
from __future__ import annotations

import collections
import json
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_DB = Path.home() / "AppData" / "Roaming" / "VRCX" / "VRCX.sqlite3"


class ChatDeduper:
    def __init__(self, window_s: float = 20.0):
        self.window = window_s
        self.seen: collections.OrderedDict[tuple[str, str], float] = collections.OrderedDict()
        self.lock = threading.Lock()

    def fresh(self, speaker: str, text: str) -> bool:
        key = (speaker.strip().lower(), " ".join(text.split()).lower())
        now = time.monotonic()
        with self.lock:
            while self.seen and next(iter(self.seen.values())) < now - self.window:
                self.seen.popitem(last=False)
            if key in self.seen:
                return False
            self.seen[key] = now
            return True


class VrcxIntake:
    def __init__(self, cfg, on_chat, on_status):
        self.cfg = cfg
        self.on_chat = on_chat
        self.on_status = on_status
        self.dedupe = ChatDeduper()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self._ws = None
        self._last_status: dict[str, str] = {}

    def start(self) -> None:
        self.stop()
        self.stop_event.clear()
        if self.cfg.vrcx_ws:
            self.threads.append(threading.Thread(target=self._ws_loop, daemon=True, name="vrcx-ws"))
        if self.cfg.vrcx_sqlite:
            self.threads.append(threading.Thread(target=self._sql_loop, daemon=True, name="vrcx-sql"))
        for t in self.threads:
            t.start()

    def stop(self) -> None:
        self.stop_event.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        for t in self.threads:
            t.join(timeout=2)
        self.threads = []

    def _status(self, key: str, msg: str) -> None:
        if self._last_status.get(key) != msg:
            self._last_status[key] = msg
            self.on_status(msg)

    def _emit(self, speaker: str, text: str) -> None:
        text = (text or "").strip()
        if text and self.dedupe.fresh(speaker, text):
            self.on_chat(speaker or "VRChat", text)

    # -- websocket ---------------------------------------------------------------------------------
    def _ws_loop(self) -> None:
        try:
            import websocket
        except ImportError:
            self._status("ws", "VRCX websocket: websocket-client not installed.")
            return
        delay = 3.0
        while not self.stop_event.is_set():
            try:
                ws = websocket.create_connection(self.cfg.vrcx_ws_url, timeout=5)
                self._ws = ws
                ws.settimeout(1.0)
                ws.send(json.dumps({"Type": 0, "Data": "UltraTranslate Connected"}))
                self._status("ws", "VRCX overlay feed connected.")
                delay = 3.0
                while not self.stop_event.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        break
                    self._ws_message(raw)
            except Exception as exc:
                self._status("ws", f"VRCX overlay feed unavailable ({type(exc).__name__}); retrying.")
                if self.stop_event.wait(delay):
                    return
                delay = min(delay * 1.5, 30.0)

    def _ws_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if int(msg.get("Type", -1)) != 1 or not isinstance(msg.get("Data"), str):
            return
        fn = msg.get("FunctionName")
        try:
            data = json.loads(msg["Data"])
        except ValueError:
            return
        items = data if isinstance(data, list) else [data]
        if fn not in ("addEntryHudFeed", "wristFeedUpdate"):
            return
        # wristFeedUpdate resends the whole feed; dedupe window keeps it from replaying.
        for item in items:
            if isinstance(item, dict) and item.get("type") == "ChatBoxMessage":
                self._emit(str(item.get("displayName", "")), str(item.get("text", "")))

    # -- sqlite ------------------------------------------------------------------------------------
    def _sql_loop(self) -> None:
        path = Path(self.cfg.vrcx_sqlite_path or DEFAULT_DB)
        last_id = None
        while not self.stop_event.is_set():
            try:
                if not path.exists():
                    self._status("sql", f"VRCX database not found at {path}; waiting.")
                    if self.stop_event.wait(10):
                        return
                    continue
                uri = f"file:{path.as_posix()}?mode=ro"
                with sqlite3.connect(uri, uri=True, timeout=1) as con:
                    if last_id is None:
                        last_id = int(con.execute("SELECT COALESCE(MAX(id),0) FROM gamelog_event").fetchone()[0])
                        self._status("sql", "VRCX database feed active.")
                    rows = con.execute("SELECT id, data FROM gamelog_event WHERE id > ? ORDER BY id LIMIT 200",
                                       (last_id,)).fetchall()
                for rid, payload in rows:
                    last_id = max(last_id, int(rid))
                    try:
                        d = json.loads(payload)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(d, dict) and d.get("type") == "ChatBoxMessage":
                        self._emit(str(d.get("displayName", "")), str(d.get("text", "")))
            except sqlite3.OperationalError as exc:
                self._status("sql", f"VRCX database busy/unavailable: {exc}")
                if self.stop_event.wait(3):
                    return
            except Exception as exc:
                self._status("sql", f"VRCX database error: {exc}")
                if self.stop_event.wait(3):
                    return
            if self.stop_event.wait(0.7):
                return
