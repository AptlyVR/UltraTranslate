"""UltraTranslate installer (Windows + NVIDIA GPU). Run by install.cmd; safe to re-run (finished steps are skipped).

    python install.py [--no-voice] [--no-screen] [--fast-translator] [--yes]

Steps:
  1  check Windows, Python 3.12, NVIDIA GPU, disk space
  2  create .venv (everything installs there; your system Python is not touched)
  3  install PyTorch (CUDA 12.8) and the app's packages (+ the voice server's)
  4  download llama.cpp (CUDA build) into bin/ for the translator and screen reader
  5  download models: Smart Turn, Whisper large-v3-turbo, Hy-MT2 translator, speaker models,
     optional Qwen3-VL screen reader and VoxCPM2 voice model
     (pyannote's segmentation model needs a free Hugging Face account: the installer opens the page for you)
  6  create "Start UltraTranslate.cmd" and "Start voice server.cmd"
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / "Scripts" / "python.exe"
BIN = ROOT / "bin"
MODELS = ROOT / "models"
ARGS = set(sys.argv[1:])
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

G, Y, R, B, D, X = "\033[92m", "\033[93m", "\033[91m", "\033[94m", "\033[90m", "\033[0m"
os.system("")       # enable ANSI colours in the Windows console
for _stream in (sys.stdout, sys.stderr):  # old consoles use cp1252 and can't print ✓ / ✕: never crash on that
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def step(n, title):
    print(f"\n{B}[{n}/6] {title}{X}")


def ok(msg):
    print(f"  {G}✓{X} {msg}")


def warn(msg):
    print(f"  {Y}!{X} {msg}")


def fail(msg):
    print(f"  {R}✕ {msg}{X}")
    if "--yes" not in ARGS:
        input("\nPress Enter to close.")
    sys.exit(1)


def run(cmd, **kw):
    print(f"  {D}> {' '.join(str(c) for c in cmd)}{X}")
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        fail(f"command failed ({r.returncode}): {cmd[0]} {cmd[1] if len(cmd) > 1 else ''}")
    return r


def ask(q, default=True):
    if "--yes" in ARGS:
        print(f"  {q} -> {'yes' if default else 'no'} (--yes)")
        return default
    a = input(f"  {q} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not a else a.startswith("y")


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        name, mem = out[0].rsplit(",", 1)
        return name.strip(), int(mem)
    except Exception:
        return None, 0


# -- 1 -----------------------------------------------------------------------------------------------------
def check():
    step(1, "Checking this PC")
    if platform.system() != "Windows":
        fail("UltraTranslate needs Windows 10/11 (WASAPI audio, SteamVR overlay).")
    build = int((platform.version().split(".") + ["0", "0", "0"])[2] or 0)
    ok(f"Windows {'11' if build >= 22000 else platform.release()}")
    if sys.version_info[:2] != (3, 12):
        fail(f"Python 3.12 is required (this is {platform.python_version()}). install.cmd can install it for you.")
    ok(f"Python {platform.python_version()}")
    name, mem = gpu_info()
    if not name:
        try:
            found = subprocess.run(["powershell", "-NoProfile", "-Command",
                                    "(Get-CimInstance Win32_VideoController).Name -join ', '"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            found = ""
        fail("UltraTranslate needs an NVIDIA GPU (CUDA)" + (f"; this PC has: {found}" if found else "") +
             ". If it does have one, install the latest NVIDIA driver and run install.cmd again.")
    ok(f"{name}, {mem // 1024} GB VRAM")
    if mem < 12000:
        warn("Under 12 GB VRAM: using the small 1.8B translator; the screen reader may not fit.")
        ARGS.add("--fast-translator")
    free = shutil.disk_usage(ROOT).free // 2 ** 30
    need = 15 if PY.is_file() else 30          # packages already installed on a re-run: only models are left
    if free < need:
        fail(f"Only {free} GB free on this drive; about {need} GB is needed (Python packages + models).")
    ok(f"{free} GB free disk")
    return mem


# -- 2 -----------------------------------------------------------------------------------------------------
def venv():
    step(2, "Creating the Python environment (.venv)")
    if PY.is_file():
        ok("already there")
        return
    run([sys.executable, "-m", "venv", VENV])
    run([PY, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    ok("created")


# -- 3 -----------------------------------------------------------------------------------------------------
def packages():
    step(3, "Installing packages (PyTorch + app). This is the long part: ~5 GB, 5-15 minutes")
    run([PY, "-m", "pip", "install", "-q", "torch", "torchaudio", "--index-url", TORCH_INDEX])
    ok("PyTorch (CUDA)")
    run([PY, "-m", "pip", "install", "-q", "-r", ROOT / "requirements.txt", "--extra-index-url", TORCH_INDEX])
    ok("app packages")
    if "--no-voice" not in ARGS:
        run([PY, "-m", "pip", "install", "-q", "-r", ROOT / "requirements-voice.txt", "--extra-index-url", TORCH_INDEX])
        ok("voice server packages")
    r = subprocess.run([PY, "-c", "import torch;print(torch.cuda.is_available())"], capture_output=True, text=True)
    if "True" not in r.stdout:
        fail("PyTorch can't see the GPU. Update the NVIDIA driver and run install.cmd again.")
    ok("PyTorch sees the GPU")


# -- 4 -----------------------------------------------------------------------------------------------------
def llama():
    step(4, "Downloading llama.cpp (translator / screen-reader engine)")
    if (BIN / "llama-server.exe").is_file():
        ok("already in bin/")
        return
    # "latest" is sometimes a tagged release without binaries: take the newest one that has the CUDA build
    rels = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15",
                                            timeout=30))
    rel, want, assets = None, [], {}
    for r in rels:
        assets = {a["name"]: a["browser_download_url"] for a in r.get("assets", [])}
        want = [n for n in assets if n.startswith("llama-") and n.endswith("-bin-win-cuda-12.4-x64.zip")]
        want += [n for n in assets if n.startswith("cudart-llama-bin-win-cuda-12.4-x64")]
        if len(want) >= 2:
            rel = r
            break
    if rel is None:
        fail("couldn't find the llama.cpp CUDA build in the latest release; download it manually into bin/")
    BIN.mkdir(exist_ok=True)
    for n in want:
        z = BIN / n
        print(f"  {D}downloading {n}{X}")
        urllib.request.urlretrieve(assets[n], z)
        with zipfile.ZipFile(z) as f:
            for m in f.namelist():
                if not m.endswith("/"):
                    (BIN / Path(m).name).write_bytes(f.read(m))     # flatten into bin/
        z.unlink()
    ok(f"llama.cpp {rel['tag_name']} in bin/")


# -- 5 -----------------------------------------------------------------------------------------------------
HF_SNIPPET = r'''
import sys, os
from huggingface_hub import hf_hub_download, snapshot_download
kind, repo, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
if kind == "file":
    p = hf_hub_download(repo, rest[0], local_dir=rest[1] if len(rest) > 1 else None)
else:
    p = snapshot_download(repo, allow_patterns=rest or None)
print(p)
'''


def hf(kind, repo, *rest):
    return subprocess.run([PY, "-c", HF_SNIPPET, kind, repo, *rest], capture_output=True, text=True)


def hf_logged_in():
    r = subprocess.run([PY, "-c", "from huggingface_hub import whoami;print(whoami()['name'])"],
                       capture_output=True, text=True)
    return r.returncode == 0


def models(mem):
    step(5, "Downloading models (about 9-20 GB depending on options)")
    MODELS.mkdir(exist_ok=True)
    if not (MODELS / "smart-turn-v3.2-cpu.onnx").is_file():
        r = hf("file", "pipecat-ai/smart-turn-v3", "smart-turn-v3.2-cpu.onnx", str(MODELS))
        if r.returncode:
            fail("Smart Turn download failed:\n" + r.stderr[-500:])
    ok("Smart Turn v3.2 (turn-taking)")
    for repo, label in (("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "Whisper large-v3-turbo (speech)"),
                        ("speechbrain/spkrec-ecapa-voxceleb", "ECAPA (speaker voiceprints)")):
        r = hf("snap", repo)
        if r.returncode:
            fail(f"{label} download failed:\n" + r.stderr[-500:])
        ok(label)
    fast = "--fast-translator" in ARGS
    repo, fname = (("tencent/Hy-MT2-1.8B-GGUF", "Hy-MT2-1.8B-Q4_K_M.gguf") if fast else
                   ("tencent/Hy-MT2-7B-GGUF", "Hy-MT2-7B-Q4_K_M.gguf"))
    r = hf("snap", repo, fname)
    if r.returncode:
        fail("translator download failed:\n" + r.stderr[-500:])
    ok(f"Hy-MT2 {'1.8B' if fast else '7B'} translator")
    # ReDimNet (second speaker model) comes through torch.hub
    r = subprocess.run([PY, "-c", "import torch;torch.hub.load('IDRnD/ReDimNet','ReDimNet',model_name='b5',"
                        "train_type='ptn',dataset='vox2',trust_repo=True)"], capture_output=True, text=True)
    ok("ReDimNet-b5 (speaker clustering)") if r.returncode == 0 else warn("ReDimNet failed; the app falls back to ECAPA alone")
    # pyannote segmentation: gated, needs a (free) Hugging Face login + accepting its terms
    while True:
        r = hf("snap", "pyannote/segmentation-3.0")
        if r.returncode == 0:
            ok("pyannote segmentation-3.0 (who is talking when)")
            break
        print(f"\n  {Y}pyannote's speaker-segmentation model needs a free Hugging Face account:{X}")
        print("   1. Sign in or sign up, then press 'Agree and access repository' on the page that opens.")
        print("   2. Create a READ token on the second page and paste it below.")
        webbrowser.open("https://huggingface.co/pyannote/segmentation-3.0")
        webbrowser.open("https://huggingface.co/settings/tokens")
        if not hf_logged_in():
            subprocess.run([PY, "-c", "from huggingface_hub import login;login()"])
        if not ask("Try the download again?", True):
            warn("skipped - speaker labels need this model; run install.cmd again later")
            break
    if "--no-screen" not in ARGS and mem >= 16000 and ask("Download the Qwen3-VL screen reader (~3.3 GB, reads chat bubbles)?", True):
        r = hf("snap", "Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-F16.gguf")
        ok("Qwen3-VL-4B screen reader") if r.returncode == 0 else warn("Qwen3-VL download failed; RapidOCR still works")
    if "--no-voice" not in ARGS and ask("Download the VoxCPM2 voice model (~4.7 GB, speaks in each person's own voice)?", True):
        r = hf("snap", "openbmb/VoxCPM2")
        ok("VoxCPM2 voice model") if r.returncode == 0 else warn("VoxCPM2 download failed; it downloads on first use instead")


# -- 6 -----------------------------------------------------------------------------------------------------
def shortcuts():
    step(6, "Creating start files")
    (ROOT / "Start UltraTranslate.cmd").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\nstart "" ".venv\\Scripts\\pythonw.exe" -m ultra_translate\r\n', encoding="utf-8")
    ok("Start UltraTranslate.cmd")
    if "--no-voice" not in ARGS:
        (ROOT / "Start voice server.cmd").write_text(
            '@echo off\r\ncd /d "%~dp0"\r\ntitle UltraTranslate voice server\r\n'
            '".venv\\Scripts\\python.exe" -m tts_server.voxcpm_server --port 8090 --eager-load\r\n', encoding="utf-8")
        ok("Start voice server.cmd (keep its window open while you use read-aloud)")


def main():
    print(f"{B}UltraTranslate installer{X}  {D}(re-run any time; finished steps are skipped){X}")
    mem = check()
    venv()
    packages()
    llama()
    models(mem)
    shortcuts()
    print(f"\n{G}All set.{X} Double-click {B}Start voice server.cmd{X} (for read-aloud), then {B}Start UltraTranslate.cmd{X}"
          " and press Start.")
    if "--yes" not in ARGS:
        input("\nPress Enter to close.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
