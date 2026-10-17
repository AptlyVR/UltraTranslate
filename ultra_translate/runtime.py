from __future__ import annotations

import ctypes
import csv
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path


_RUNTIME_READY = False
_INTEGRITY_SIDS = {
    "S-1-16-4096": "Low",
    "S-1-16-8192": "Medium",
    "S-1-16-8448": "MediumPlus",
    "S-1-16-12288": "High",
    "S-1-16-16384": "System",
}


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("Sid", ctypes.c_void_p),
        ("Attributes", ctypes.c_uint32),
    ]


class _TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", _SID_AND_ATTRIBUTES)]


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_INTEGRITY_LEVEL = 25

_ADVAPI32 = ctypes.WinDLL("advapi32", use_last_error=True)
_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)

_ADVAPI32.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
_ADVAPI32.OpenProcessToken.restype = ctypes.c_int
_ADVAPI32.GetTokenInformation.argtypes = [
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),
]
_ADVAPI32.GetTokenInformation.restype = ctypes.c_int
_ADVAPI32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_ADVAPI32.ConvertSidToStringSidW.restype = ctypes.c_int
_KERNEL32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
_KERNEL32.OpenProcess.restype = ctypes.c_void_p
_KERNEL32.CloseHandle.argtypes = [ctypes.c_void_p]
_KERNEL32.CloseHandle.restype = ctypes.c_int
_KERNEL32.LocalFree.argtypes = [ctypes.c_void_p]
_KERNEL32.LocalFree.restype = ctypes.c_void_p


def _integrity_from_pid(pid: int) -> str | None:
    process = _KERNEL32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not process:
        return None
    token = ctypes.c_void_p()
    try:
        if not _ADVAPI32.OpenProcessToken(process, _TOKEN_QUERY, ctypes.byref(token)):
            return None
        needed = ctypes.c_uint32()
        _ADVAPI32.GetTokenInformation(token, _TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(needed))
        if not needed.value:
            return None
        buffer = ctypes.create_string_buffer(needed.value)
        if not _ADVAPI32.GetTokenInformation(
            token,
            _TOKEN_INTEGRITY_LEVEL,
            buffer,
            needed.value,
            ctypes.byref(needed),
        ):
            return None
        label = ctypes.cast(buffer, ctypes.POINTER(_TOKEN_MANDATORY_LABEL)).contents
        sid_ptr = ctypes.c_void_p()
        if not _ADVAPI32.ConvertSidToStringSidW(label.Label.Sid, ctypes.byref(sid_ptr)):
            return None
        try:
            sid = ctypes.wstring_at(sid_ptr)
        finally:
            _KERNEL32.LocalFree(sid_ptr)
        return _INTEGRITY_SIDS.get(sid, sid)
    finally:
        if token.value:
            _KERNEL32.CloseHandle(token.value)
        _KERNEL32.CloseHandle(process)


def current_integrity_level() -> str | None:
    if os.name != "nt":
        return None
    return _integrity_from_pid(os.getpid())


def process_integrity_level(image_name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        output = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH"],
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    normalized = image_name.lower()
    for row in csv.reader(output.splitlines()):
        if len(row) < 2:
            continue
        if row[0].strip().lower() != normalized:
            continue
        try:
            return _integrity_from_pid(int(row[1]))
        except ValueError:
            return None
    return None


def relaunch_limited_if_needed(module_name: str = "ultra_translate") -> bool:
    if os.name != "nt":
        return False
    if os.environ.get("ULTRA_TRANSLATE_LIMITED_RELAUNCH") == "1":
        return False
    if current_integrity_level() != "High":
        return False

    task_name = f"UltraTranslateLimited-{os.getpid()}"
    launcher_path = Path(tempfile.gettempdir()) / f"{task_name}.cmd"
    launcher_path.write_text(
        "\n".join(
            [
                "@echo off",
                "set ULTRA_TRANSLATE_LIMITED_RELAUNCH=1",
                f'cd /d "{Path.cwd()}"',
                f'"{sys.executable}" -m {module_name}',
                f'schtasks /Delete /TN "{task_name}" /F >nul 2>nul',
                'del "%~f0" >nul 2>nul',
            ]
        ),
        encoding="ascii",
    )
    run_time = (datetime.now() + timedelta(minutes=1)).strftime("%H:%M")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.run(
            [
                "schtasks",
                "/Create",
                "/TN",
                task_name,
                "/SC",
                "ONCE",
                "/ST",
                run_time,
                "/TR",
                str(launcher_path),
                "/RL",
                "LIMITED",
                "/F",
            ],
            check=True,
            capture_output=True,
            text=True,
            creationflags=flags,
        )
        subprocess.run(
            ["schtasks", "/Run", "/TN", task_name],
            check=True,
            capture_output=True,
            text=True,
            creationflags=flags,
        )
        return True
    except Exception:
        try:
            launcher_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def prepare_runtime() -> None:
    global _RUNTIME_READY
    if _RUNTIME_READY or os.name != "nt":
        return
    _RUNTIME_READY = True

    conda_icu = Path(sys.prefix) / "Library" / "bin" / "icuuc.dll"
    system_icu = Path(r"C:\Windows\System32\icuuc.dll")
    if not conda_icu.exists() or not system_icu.exists():
        return

    # Conda environments often expose an ICU build with version-suffixed exports that
    # PyQt6's Qt6Core.dll cannot bind against. Preloading the Windows ICU keeps Qt stable
    # without mutating the user's ComfyUI environment.
    ctypes.WinDLL(str(system_icu))
