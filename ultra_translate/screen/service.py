"""Screen translation: snapshot (button / F8 / VR chord) or optional Live mode, via the isolated OCR worker.

Live mode captures ~1.5 fps, skips frames that did not change (160x90 thumbnail diff in the worker),
pauses while the head turns fast, and only translates text blocks it has not translated before.
"""
from __future__ import annotations

try:
    from .chatbox import split_status
except ImportError:          # loaded outside the package
    from chatbox import split_status
import json
import os
import sys
import queue
import subprocess
import threading
import time
import unicodedata
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "ocr_worker.py"
DEFAULT_PYTHON = os.environ.get("ULTRATRANSLATE_OCR_PYTHON", sys.executable)   # a python with rapidocr installed


def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).lower()


class ScreenTranslator:
    def __init__(self, cfg, hub, target_getter, on_result, on_status, on_chat=None, players=None):
        self.cfg = cfg
        self.hub = hub
        self.target = target_getter
        self.on_result = on_result       # (list[(original, translation)], info dict)
        self.on_status = on_status
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.cache: dict[tuple[str, str], str] = {}
        self.live = False
        self._live_thread: threading.Thread | None = None
        self._busy = threading.Lock()
        self.head_turning = lambda: False
        self.on_chat = on_chat or (lambda msgs, info: None)
        self.players = players or (lambda: [])
        self.seen_chat: dict[tuple[str, str], float] = {}
        from .vlm import VlmReader
        self.vlm = VlmReader(log=on_status)
        self._thumb = None

    # -- worker ------------------------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        if self.cfg.engine == "vlm":
            return self.vlm.loaded
        return self.proc is not None and self.proc.poll() is None

    def load(self) -> None:
        if self.loaded:
            return
        if self.cfg.engine == "vlm":
            self.vlm.load()
            self.on_status("Screen reader ready (Qwen3-VL-4B).")
            return
        py = self.cfg.python or DEFAULT_PYTHON
        if not Path(py).is_file():
            raise FileNotFoundError(f"OCR Python not found: {py}")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        self.proc = subprocess.Popen([py, "-u", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
                                     creationflags=flags, env=env, cwd=str(WORKER.parent))
        try:
            from ..core.childproc import kill_with_app
            kill_with_app(self.proc)
        except ImportError:
            pass
        r = self._call({"cmd": "load", "device": self.cfg.device}, timeout=120)
        if not r.get("ok"):
            self.unload()
            raise RuntimeError(r.get("error", "OCR load failed"))
        self.on_status(f"Screen OCR ready in {r.get('ms', 0) / 1000:.1f} s ({self.cfg.device}).")

    def unload(self) -> None:
        self.set_live(False)
        self.vlm.unload()
        proc, self.proc = self.proc, None
        if proc and proc.poll() is None:
            try:
                proc.stdin.write('{"cmd":"quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        self.on_status("Screen OCR unloaded.")

    def _call(self, req: dict, timeout: float = 30) -> dict:
        with self.lock:
            if not self.loaded:
                return {"ok": False, "error": "OCR not loaded"}
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            q: queue.Queue = queue.Queue()
            threading.Thread(target=lambda: q.put(self.proc.stdout.readline()), daemon=True).start()
            try:
                line = q.get(timeout=timeout)
            except queue.Empty:
                self.proc.kill()
                self.proc = None
                return {"ok": False, "error": "OCR worker timed out (restarted on next use)"}
        try:
            return json.loads(line) if line else {"ok": False, "error": "OCR worker exited"}
        except ValueError:
            return {"ok": False, "error": f"bad worker reply: {line[:80]}"}

    # -- capture + translate -------------------------------------------------------------------------
    def snapshot_async(self, mode: str = "text") -> None:
        """One-shot read (button / F8 / VR chord): all screen text by default (signs, posters)."""
        threading.Thread(target=self._snapshot, kwargs={"only_if_changed": False, "mode": mode}, daemon=True).start()

    def _snapshot(self, only_if_changed: bool, mode: str | None = None) -> bool:
        mode = mode or self.cfg.mode
        if not self._busy.acquire(blocking=False):
            return False
        try:
            if not self.loaded:
                self.load()
            if self.cfg.engine == "vlm":
                r = self._vlm_snap(mode, only_if_changed)
            else:
                r = self._call({"cmd": "snap", "window": self.cfg.window, "crop": self.cfg.crop, "mode": mode,
                                "players": list(self.players()),
                                "source": "auto", "only_if_changed": only_if_changed, "device": self.cfg.device})
            if not r.get("ok"):
                self.on_status(f"Screen OCR: {r.get('error')}")
                return False
            if not r.get("changed", True):
                return False
            t0 = time.perf_counter()
            target = self.target()
            if mode == "chatbox":
                self._chat(r, target, t0, only_if_changed)
                return True
            pairs = []
            for block in r.get("blocks", []):
                k = (_key(block), target)
                if k not in self.cache:
                    try:
                        self.cache[k] = self.hub.translate(block, target=target, model=self.cfg.model)
                    except Exception as exc:
                        self.on_status(f"Screen translate failed: {exc}")
                        self.cache[k] = ""
                pairs.append((block, self.cache[k]))
            if len(self.cache) > 2000:
                self.cache.clear()
            self.on_result(pairs, {"ocr_ms": r.get("ms"), "mt_ms": round((time.perf_counter() - t0) * 1000),
                                   "src": r.get("src"), "live": only_if_changed})
            return True
        except Exception as exc:
            self.on_status(f"Screen translate error: {exc}")
            return False
        finally:
            self._busy.release()

    def _vlm_snap(self, mode: str, only_if_changed: bool) -> dict:
        """Capture in-process, skip unchanged frames, read with Qwen3-VL."""
        import numpy as np
        from .capture import grab
        t0 = time.perf_counter()
        img, src = grab(self.cfg.window, self.cfg.crop)
        small = img[::12, ::12].mean(axis=2).astype(np.int16)
        changed = self._thumb is None or self._thumb.shape != small.shape or \
            float(np.mean(np.abs(small - self._thumb))) > 2.5
        if only_if_changed and not changed:
            return {"ok": True, "changed": False, "src": src}
        self._thumb = small
        resp = {"ok": True, "changed": True, "src": src}
        if mode == "chatbox":
            resp["chatboxes"] = self.vlm.read_chat(img)
            resp["blocks"] = []
        else:
            resp["blocks"] = self.vlm.read_text(img)
        resp["ms"] = round((time.perf_counter() - t0) * 1000)
        return resp

    def _snap_name(self, name: str) -> str:
        """OCR'd nametag -> the closest player in the instance (fixes small misreads)."""
        import difflib
        players = self.players()
        if not name or not players:
            return name
        if name in players:
            return name
        low = {p.lower(): p for p in players}
        m = difflib.get_close_matches(name.lower(), list(low), n=1, cutoff=0.6)
        return low[m[0]] if m else name

    def _chat(self, r: dict, target: str, t0: float, live: bool) -> None:
        now = time.monotonic()
        for k in [k for k, t in self.seen_chat.items() if now - t > 600]:
            self.seen_chat.pop(k)
        msgs = []
        for b in r.get("chatboxes", []):
            text = (b.get("text") or "").strip()
            if not b.get("status") and text:      # VLM bubbles: split OSC status lines (clock, song) off here
                msg, status = split_status([t for t in text.splitlines() if t.strip()])
                text = " ".join(msg)
            if len(text) < 1:
                continue                       # status-only bubble (clock / now playing): nothing to translate
            raw = (b.get("name") or "").strip()
            name = self._snap_name(raw)
            if b.get("name_guess") and self.players() and name not in self.players():
                name = ""                     # a guessed line that isn't a player in the instance: no name
            key = (name.lower(), _key(text))
            if key in self.seen_chat:          # a bubble stays up for a while: translate/read it once
                self.seen_chat[key] = now
                continue
            self.seen_chat[key] = now
            ck = (_key(text), target)
            if ck not in self.cache:
                try:
                    self.cache[ck] = self.hub.translate(text, target=target, model=self.cfg.model)
                except Exception as exc:
                    self.on_status(f"Chatbox translate failed: {exc}")
                    self.cache[ck] = ""
            msgs.append((name or "?", text, self.cache[ck]))
        if msgs or not live:
            self.on_chat(msgs, {"ocr_ms": r.get("ms"), "mt_ms": round((time.perf_counter() - t0) * 1000),
                                "src": r.get("src"), "live": live, "bubbles": len(r.get("chatboxes", []))})

    def set_live(self, on: bool) -> None:
        self.live = on
        if on and (self._live_thread is None or not self._live_thread.is_alive()):
            self._live_thread = threading.Thread(target=self._live_loop, daemon=True, name="screen-live")
            self._live_thread.start()

    def _live_loop(self) -> None:
        period = 1.0 / max(0.2, self.cfg.live_fps)
        while self.live:
            t0 = time.monotonic()
            if not self.head_turning():
                self._snapshot(only_if_changed=True)
            time.sleep(max(0.05, period - (time.monotonic() - t0)))
