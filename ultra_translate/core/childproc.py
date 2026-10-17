"""Make helper processes (Qwen3-VL llama-server, OCR worker) die with the app, even on a crash or force-close.

Windows job object with KILL_ON_JOB_CLOSE: when our process ends, the OS closes the job handle and kills every
process in it. Without this, a killed app left a 5.5 GB model server running on the GPU.
"""
from __future__ import annotations

import ctypes
import sys

_job = None


def _job_handle():
    global _job
    if _job is not None or sys.platform != "win32":
        return _job
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IOC(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IOC), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    h = k32.CreateJobObjectW(None, None)
    if not h:
        return None
    info = EXT()
    info.Basic.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info)):   # ExtendedLimitInformation
        return None
    _job = (k32, h)
    return _job


def kill_with_app(proc) -> bool:
    """Tie a subprocess.Popen to this process's lifetime. Returns True when it worked."""
    try:
        j = _job_handle()
        if not j:
            return False
        k32, h = j
        from ctypes import wintypes
        k32.OpenProcess.restype = wintypes.HANDLE
        ph = k32.OpenProcess(0x0001 | 0x0100, False, proc.pid)      # PROCESS_TERMINATE | PROCESS_SET_QUOTA
        if not ph:
            return False
        ok = bool(k32.AssignProcessToJobObject(h, ph))
        k32.CloseHandle(ph)
        return ok
    except Exception:
        return False
