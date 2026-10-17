"""Tiny client for TranslateHub (127.0.0.1:8097). Copy this file into any app that needs translation.

    from translatehub_client import TranslateHubClient
    hub = TranslateHubClient()
    hub.translate("こんにちは", target="en")   # starts the hub detached if it is not running
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HUB_URL = os.environ.get("TRANSLATEHUB_URL", "http://127.0.0.1:8097")
HUB_SCRIPT = Path(os.environ.get("TRANSLATEHUB_SCRIPT",
                                 str(Path(__file__).resolve().parents[2] / "translatehub" / "translate_hub.py")))


class TranslateHubError(RuntimeError):
    pass


class TranslateHubClient:
    def __init__(self, url: str = HUB_URL, autostart: bool = True, python: str | None = None):
        self.url = url.rstrip("/")
        self.autostart = autostart
        self.python = python or _pythonw()
        self._started_at = 0.0

    def _call(self, path: str, payload: dict | None = None, timeout: float = 30) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error")
            except Exception:
                detail = str(exc)
            raise TranslateHubError(detail) from exc

    def health(self, timeout: float = 0.5) -> dict | None:
        try:
            return self._call("/health", timeout=timeout)
        except (OSError, ValueError, TranslateHubError):
            return None

    def ensure_running(self, wait_s: float = 8.0) -> bool:
        if self.health():
            return True
        if not self.autostart or not HUB_SCRIPT.is_file():
            return False
        if time.monotonic() - self._started_at > 15:
            flags = 0
            if os.name == "nt":
                flags = (subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
                         | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            subprocess.Popen([self.python, str(HUB_SCRIPT)], cwd=str(HUB_SCRIPT.parent),
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=flags, close_fds=True)
            self._started_at = time.monotonic()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self.health():
                return True
            time.sleep(0.2)
        return False

    def load(self, model: str | None = None, switch: bool = False) -> dict:
        """Make sure a model is loaded. switch=True replaces a different loaded model (affects all apps)."""
        self.ensure_running()
        body = {"switch": switch}
        if model:
            body["model"] = model
        return self._call("/v1/load", body, timeout=90)

    def translate(self, text: str, target: str = "en", source: str | None = None,
                  context: list[str] | None = None, model: str | None = None, timeout: float = 90) -> str:
        if not self.ensure_running():
            raise TranslateHubError("TranslateHub is not running")
        body = {"text": text, "target": target, "source": source, "context": context or []}
        if model:
            body["model"] = model
        return str(self._call("/v1/translate", body, timeout=timeout).get("text") or "")


def _pythonw() -> str:
    exe = Path(sys.executable)
    candidate = exe.with_name("pythonw.exe")
    return str(candidate if candidate.is_file() else exe)
