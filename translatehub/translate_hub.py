"""TranslateHub: one shared local Hy-MT2 translator for several apps.

A single process listens on 127.0.0.1:8097. The first request loads the model into a private
llama-server child; after `idle_unload_s` without requests the child is stopped and VRAM is freed.
Clients never spawn their own translator: they call this, and start it detached if it is not running.

Endpoints
  GET  /health                     -> {"status":"ok","loaded":bool,"model":"best","idle_s":..}
  POST /v1/translate               {"text","target","source"?,"context"?:[str],"model"?} -> {"text","ms","model"}
  POST /v1/translate/batch         {"items":[{"text","target",...}],"model"?} -> {"items":[{"text"}],"ms"}
  POST /v1/chat/completions        OpenAI-compatible passthrough (loads on demand)
  POST /v1/load  {"model"?}        preload;  POST /v1/unload   free VRAM now
Language codes: ISO-639-1 ("ja"), NLLB ("jpn_Jpan") or English names ("Japanese").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("TRANSLATEHUB_PORT", "8097"))
ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
HF_CACHE = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
DEFAULT_LLAMA = Path(os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
             or Path(__file__).resolve().parents[1] / "bin" / "llama-server.exe")   # install.cmd puts it in bin/

MODELS = {
    "best": ("models--tencent--Hy-MT2-7B-GGUF", "Hy-MT2-7B-Q4_K_M.gguf", "Hy-MT2 7B Q4"),
    "fast": ("models--tencent--Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-Q4_K_M.gguf", "Hy-MT2 1.8B Q4"),
}

LANG_NAMES = {
    "zh": "Chinese", "zh-hant": "Traditional Chinese", "yue": "Cantonese", "en": "English", "fr": "French",
    "pt": "Portuguese", "es": "Spanish", "ja": "Japanese", "tr": "Turkish", "ru": "Russian", "ar": "Arabic",
    "ko": "Korean", "th": "Thai", "it": "Italian", "de": "German", "vi": "Vietnamese", "ms": "Malay",
    "id": "Indonesian", "tl": "Filipino", "hi": "Hindi", "pl": "Polish", "cs": "Czech", "nl": "Dutch",
    "km": "Khmer", "my": "Burmese", "fa": "Persian", "gu": "Gujarati", "ur": "Urdu", "te": "Telugu",
    "mr": "Marathi", "he": "Hebrew", "bn": "Bengali", "ta": "Tamil", "uk": "Ukrainian", "bo": "Tibetan",
    "kk": "Kazakh", "mn": "Mongolian", "ug": "Uyghur",
}
NLLB_TO_ISO = {
    "zho_hans": "zh", "zho_hant": "zh-hant", "yue_hant": "yue", "eng_latn": "en", "fra_latn": "fr",
    "por_latn": "pt", "spa_latn": "es", "jpn_jpan": "ja", "tur_latn": "tr", "rus_cyrl": "ru", "arb_arab": "ar",
    "kor_hang": "ko", "tha_thai": "th", "ita_latn": "it", "deu_latn": "de", "vie_latn": "vi", "zsm_latn": "ms",
    "ind_latn": "id", "tgl_latn": "tl", "hin_deva": "hi", "pol_latn": "pl", "ces_latn": "cs", "nld_latn": "nl",
    "khm_khmr": "km", "mya_mymr": "my", "pes_arab": "fa", "guj_gujr": "gu", "urd_arab": "ur", "tel_telu": "te",
    "mar_deva": "mr", "heb_hebr": "he", "ben_beng": "bn", "tam_taml": "ta", "ukr_cyrl": "uk", "bod_tibt": "bo",
    "kaz_cyrl": "kk", "khk_cyrl": "mn", "uig_arab": "ug",
}
_NAME_TO_ISO = {v.lower(): k for k, v in LANG_NAMES.items()}


def lang_name(code: str | None) -> str | None:
    value = str(code or "").strip()
    if not value:
        return None
    low = value.lower().replace("_", "-")
    if low in LANG_NAMES:
        return LANG_NAMES[low]
    nllb = low.replace("-", "_")
    if nllb in NLLB_TO_ISO:
        return LANG_NAMES[NLLB_TO_ISO[nllb]]
    if low in _NAME_TO_ISO:
        return LANG_NAMES[_NAME_TO_ISO[low]]
    return LANG_NAMES.get(low.split("-", 1)[0])


def build_prompt(text: str, target: str, context: list[str] | None = None) -> str:
    # Hy-MT prompt templates (tencent/Hy-MT model card). Chinese-pair prompt when either side is Chinese.
    text = text.strip()
    ctx = "\n".join(c.strip() for c in (context or []) if c and c.strip())
    if target in ("Chinese", "Traditional Chinese", "Cantonese"):
        if ctx:
            return f"{ctx}\n参考上面的信息，把下面的文本翻译成{target}，注意不需要翻译上文，也不要额外解释：\n{text}"
        return f"将以下文本翻译为{target}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"
    if ctx:
        return (f"{ctx}\nReferring to the information above, translate the following text into {target}. "
                f"Do not translate the context above and do not add any explanation:\n{text}")
    return (f"Translate the following segment into {target}, without additional explanation.\n\n{text}")


NEED_MIB = {"best": 5800, "fast": 2200}     # measured-ish VRAM per model incl. 2 x 2048 ctx
HEADROOM_MIB = int(os.environ.get("TRANSLATEHUB_HEADROOM_MIB", "3000"))   # PC freezes if VRAM overflows


def free_vram_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class Engine:
    def __init__(self, model: str, idle_unload_s: float, gpu_layers: str, parallel: int):
        self.model = model
        self.idle_unload_s = idle_unload_s
        self.gpu_layers = gpu_layers
        self.parallel = max(1, parallel)
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.loaded_model: str | None = None
        self.last_used = time.monotonic()
        self.inflight = 0
        self.log_handle = None
        self.load_ms = 0.0
        self.loading = False
        threading.Thread(target=self._idle_loop, daemon=True).start()

    @staticmethod
    def _model_path(key: str) -> Path:
        repo, filename, _ = MODELS[key]
        env = os.environ.get(f"TRANSLATEHUB_{key.upper()}_MODEL")
        if env and Path(env).is_file():
            return Path(env)
        found = sorted((HF_CACHE / repo / "snapshots").glob(f"*/{filename}"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise FileNotFoundError(f"{filename} not in {HF_CACHE / repo}")
        return found[-1]

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind((HOST, 0))
            return s.getsockname()[1]

    def _req(self, path: str, payload: dict | None = None, timeout: float = 60) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"http://{HOST}:{self.port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def ready(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.loaded_model is not None

    def ensure(self, model: str | None = None, switch: bool = False) -> None:
        """Load `model` if nothing is loaded. A different loaded model is reused (no ping-pong between
        apps preferring different sizes) unless switch=True (explicit /v1/load)."""
        want = model if model in MODELS else self.model
        with self.lock:
            if self.ready() and (self.loaded_model == want or not switch):
                return
            self._stop()
            self.loading = True
            try:
                self._load(want)
            finally:
                self.loading = False

    def _load(self, want: str) -> None:
        if True:
            free = free_vram_mib()
            if free is not None:
                if free < NEED_MIB[want] + HEADROOM_MIB and want == "best" and free >= NEED_MIB["fast"] + HEADROOM_MIB:
                    log(f"only {free} MiB VRAM free; loading the 1.8B model instead of 7B")
                    want = "fast"
                if free < NEED_MIB[want] + HEADROOM_MIB:
                    raise RuntimeError(f"not enough free VRAM for {MODELS[want][2]}: {free} MiB free, "
                                       f"need {NEED_MIB[want] + HEADROOM_MIB} MiB (incl. safety headroom)")
            server = Path(os.environ.get("TRANSLATEHUB_LLAMA_SERVER") or DEFAULT_LLAMA)
            if not server.is_file():
                raise FileNotFoundError(f"llama-server not found: {server}")
            LOG_DIR.mkdir(exist_ok=True)
            self.log_handle = open(LOG_DIR / "llama-server.log", "ab", buffering=0)
            self.port = self._free_port()
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            cmd = [str(server), "--model", str(self._model_path(want)), "--host", HOST, "--port", str(self.port),
                   "--ctx-size", str(2048 * self.parallel), "--parallel", str(self.parallel),
                   "--gpu-layers", self.gpu_layers, "--flash-attn", "on", "--no-webui"]
            started = time.perf_counter()
            self.proc = subprocess.Popen(cmd, cwd=str(server.parent), stdin=subprocess.DEVNULL,
                                         stdout=self.log_handle, stderr=self.log_handle, creationflags=flags)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    code = self.proc.returncode
                    self._stop()
                    raise RuntimeError(f"llama-server exited ({code}); see {LOG_DIR / 'llama-server.log'}")
                try:
                    if self._req("/health", timeout=1).get("status") == "ok":
                        self.loaded_model = want
                        self.load_ms = (time.perf_counter() - started) * 1000
                        log(f"loaded {MODELS[want][2]} in {self.load_ms:.0f} ms on port {self.port}")
                        return
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(0.15)
            self._stop()
            raise RuntimeError("llama-server did not become ready in 60 s")

    def _stop(self) -> None:
        proc, self.proc = self.proc, None
        self.loaded_model = None
        self.port = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None

    def unload(self) -> None:
        with self.lock:
            if self.proc is not None:
                log("unloading model")
            self._stop()

    def _idle_loop(self) -> None:
        while True:
            time.sleep(5)
            if self.idle_unload_s <= 0 or not self.ready() or self.inflight:
                continue
            if time.monotonic() - self.last_used > self.idle_unload_s:
                with self.lock:
                    if not self.inflight and time.monotonic() - self.last_used > self.idle_unload_s:
                        log(f"idle {self.idle_unload_s:.0f} s")
                        self._stop()

    def chat(self, payload: dict, model: str | None = None, timeout: float = 120, no_load: bool = False) -> dict:
        if no_load and not self.ready():
            raise NotLoaded("TranslateHub model is not loaded (no_load requested)")
        self.last_used = time.monotonic()
        self.inflight += 1
        try:
            self.ensure(model)
            port = self.port
            data = json.dumps(payload).encode()
            req = urllib.request.Request(f"http://{HOST}:{port}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        finally:
            self.inflight -= 1
            self.last_used = time.monotonic()

    def translate(self, text: str, target: str, context: list[str] | None = None, model: str | None = None,
                  no_load: bool = False) -> str:
        name = lang_name(target)
        if not name:
            raise ValueError(f"unsupported target language: {target}")
        if not text.strip():
            return ""
        out = self.chat({
            "messages": [{"role": "user", "content": build_prompt(text, name, context)}],
            "temperature": 0.0, "top_k": 1, "repeat_penalty": 1.05,
            "max_tokens": min(512, 64 + len(text) * 4), "cache_prompt": True,
        }, model=model, no_load=no_load)
        choices = out.get("choices") or [{}]
        return str((choices[0].get("message") or {}).get("content") or "").strip()


class NotLoaded(RuntimeError):
    pass


ENGINE: Engine | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        LOG_DIR.mkdir(exist_ok=True)
        with open(LOG_DIR / "hub.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # quiet
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            e = ENGINE
            self._send(200, {"status": "ok", "service": "translatehub", "loaded": e.ready(), "loading": e.loading,
                             "model": e.loaded_model or e.model, "default_model": e.model,
                             "idle_s": round(time.monotonic() - e.last_used, 1),
                             "idle_unload_s": e.idle_unload_s, "load_ms": round(e.load_ms),
                             "languages": sorted(LANG_NAMES)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        e = ENGINE
        try:
            body = self._body()
            path = self.path.rstrip("/")
            t0 = time.perf_counter()
            if path == "/v1/translate":
                no_load = bool(body.get("no_load")) or self.headers.get("X-No-Load") == "1"
                text = e.translate(str(body.get("text") or ""), body.get("target") or "en",
                                   body.get("context"), body.get("model"), no_load=no_load)
                self._send(200, {"text": text, "ms": round((time.perf_counter() - t0) * 1000, 1),
                                 "model": e.loaded_model})
            elif path == "/v1/translate/batch":
                items = [{"text": e.translate(str(i.get("text") or ""), i.get("target") or "en",
                                              i.get("context"), body.get("model"))}
                         for i in body.get("items") or []]
                self._send(200, {"items": items, "ms": round((time.perf_counter() - t0) * 1000, 1)})
            elif path == "/v1/chat/completions":
                no_load = bool(body.pop("no_load", False)) or self.headers.get("X-No-Load") == "1"
                self._send(200, e.chat(body, model=body.pop("hub_model", None), no_load=no_load))
            elif path == "/v1/load":
                e.last_used = time.monotonic()
                e.ensure(body.get("model"), switch=bool(body.get("switch")))
                self._send(200, {"loaded": True, "model": e.loaded_model, "load_ms": round(e.load_ms)})
            elif path == "/v1/unload":
                e.unload()
                self._send(200, {"loaded": False})
            else:
                self._send(404, {"error": "not found"})
        except NotLoaded as exc:
            self._send(409, {"error": str(exc), "loaded": False, "loading": e.loading})
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # report, never hide
            log(f"error {self.path}: {exc}")
            self._send(503, {"error": str(exc)})


def main() -> int:
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TRANSLATEHUB_MODEL", "best"), choices=sorted(MODELS))
    ap.add_argument("--idle-unload", type=float, default=float(os.environ.get("TRANSLATEHUB_IDLE_S", "900")))
    ap.add_argument("--gpu-layers", default="all")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--preload", action="store_true")
    args = ap.parse_args()
    ENGINE = Engine(args.model, args.idle_unload, args.gpu_layers, args.parallel)
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError:
        print(f"TranslateHub already running on {HOST}:{PORT}", flush=True)
        return 0
    server.daemon_threads = True
    log(f"TranslateHub listening on http://{HOST}:{PORT} (model={args.model}, idle unload {args.idle_unload:.0f} s)")
    if args.preload:
        threading.Thread(target=ENGINE.ensure, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        ENGINE.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
