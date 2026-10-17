"""Qwen3-VL-4B screen reader via llama.cpp (the accurate engine; see tools/score_ocr.py bench).

Bench on real VRChat screenshots (24 bubbles, 8 languages): 24/24 found, 0.6% character error, 23/24 names,
~0.7 s per 1920px frame, ~5.5 GB VRAM. Loaded on demand, freed on unload.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

HF = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
SERVER = Path(os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
             or Path(__file__).resolve().parents[2] / "bin" / "llama-server.exe")   # install.cmd puts it in bin/
REPO = "models--Qwen--Qwen3-VL-4B-Instruct-GGUF"
NEED_MIB = 5600
CHAT_PROMPT = ("This is a VRChat screenshot. Find every chat bubble (white text on a dark rounded box floating "
               "above a player). A player's nametag may be directly under the bubble, but nametags are often hidden: "
               "then use an empty name, never guess one. Some bubbles are status widgets that also show a clock, "
               "a now-playing song, song progress or heart rate: put those lines in \"status\" and only the "
               "person's actual message in \"text\" (empty if there is none). Ignore empty or '...' typing bubbles. "
               'Reply ONLY with JSON: [{"name": "<nametag text or empty>", "text": "<message exactly as written, '
               'original language>", "status": ["<status line>", ...]}]. Return [] if there are no chat bubbles.')
TEXT_PROMPT = ("Read all visible text in this VRChat screenshot (signs, posters, menus). Reply ONLY with JSON: "
               '["<text block 1>", "<text block 2>", ...] in reading order, original language. [] if none.')


def _find(pattern: str) -> Path | None:
    hits = sorted((HF / REPO / "snapshots").glob(f"*/{pattern}"))
    return hits[-1] if hits else None


class VlmReader:
    def __init__(self, log=print):
        self.log = log
        self.proc: subprocess.Popen | None = None
        self.port = 0

    @property
    def available(self) -> bool:
        return SERVER.is_file() and _find("Qwen3VL-4B-Instruct-Q4_K_M.gguf") is not None

    @property
    def loaded(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def load(self) -> None:
        if self.loaded:
            return
        model, mmproj = _find("Qwen3VL-4B-Instruct-Q4_K_M.gguf"), _find("mmproj-Qwen3VL-4B-Instruct-F16.gguf")
        if not model or not mmproj:
            raise FileNotFoundError("Qwen3-VL-4B GGUF not downloaded")
        from ..core.vram import free_vram_mib
        free = free_vram_mib()
        if free is not None and free < NEED_MIB + 3000:
            raise RuntimeError(f"Only {free} MiB GPU memory free; Qwen3-VL needs {NEED_MIB} + 3000 headroom. "
                               "Use the RapidOCR engine instead.")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(SERVER), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1", "--port", str(self.port),
             "-ngl", "99", "--ctx-size", "8192", "--parallel", "1", "--no-webui", "--flash-attn", "on"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        try:
            from ..core.childproc import kill_with_app
            kill_with_app(self.proc)          # never outlive the app (a crash once left 5.5 GB on the GPU)
        except ImportError:
            pass
        while time.perf_counter() - t0 < 120:
            if self.proc.poll() is not None:
                self.proc = None
                raise RuntimeError("Qwen3-VL server exited while loading")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        self.log(f"Qwen3-VL ready in {time.perf_counter() - t0:.1f} s")
                        return
            except Exception:
                pass
            time.sleep(0.2)
        self.unload()
        raise RuntimeError("Qwen3-VL server did not become ready")

    def unload(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()

    def _ask(self, img_bgr, prompt: str, max_side: int = 1920) -> str:
        from PIL import Image
        img = Image.fromarray(img_bgr[:, :, ::-1])
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        body = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": prompt}]}], "temperature": 0, "max_tokens": 500}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def read_chat(self, img_bgr) -> list[dict]:
        m = re.search(r"\[.*\]", self._ask(img_bgr, CHAT_PROMPT), re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except ValueError:
            return []
        return [{"name": str(i.get("name", "")).strip(), "text": str(i.get("text", "")).strip(),
                 "status": [str(x) for x in (i.get("status") or []) if str(x).strip()]}
                for i in items if isinstance(i, dict) and (i.get("text") or i.get("status"))]

    def read_text(self, img_bgr) -> list[str]:
        reply = self._ask(img_bgr, TEXT_PROMPT)
        m = re.search(r"\[.*\]", reply, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
            return [str(x).strip() for x in items if str(x).strip()]
        except ValueError:
            return [reply.strip()] if reply.strip() else []
