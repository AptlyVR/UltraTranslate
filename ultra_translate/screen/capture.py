"""Screen capture: the VRChat window's client area (its desktop mirror shows the
VR view), falling back to the primary monitor."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

import mss
import numpy as np

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # real pixel coordinates
except Exception:
    pass

_sct = None


def _grabber():
    global _sct
    if _sct is None:
        _sct = mss.mss()
    return _sct


def find_window(title_part: str) -> int:
    """First visible top-level window whose title contains title_part."""
    found = []
    needle = title_part.lower()
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if needle in buf.value.lower():
                found.append(hwnd)
                return False
        return True

    user32.EnumWindows(proc(cb), 0)
    return found[0] if found else 0


def _client_rect(hwnd: int) -> dict | None:
    r = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(r)):
        return None
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    w, h = r.right - r.left, r.bottom - r.top
    if w < 50 or h < 50:
        return None
    return {"left": pt.x, "top": pt.y, "width": w, "height": h}


def grab(window_title: str = "VRChat", crop_center: float = 1.0) -> tuple[np.ndarray, str]:
    """Returns (BGR image, source description). crop_center < 1 keeps only the
    central fraction (useful in VR: text you look at is near the middle)."""
    sct = _grabber()
    region, src = None, "monitor"
    if window_title:
        hwnd = find_window(window_title)
        if hwnd:
            region = _client_rect(hwnd)
            if region:
                src = f"window '{window_title}'"
    if region is None:
        region = sct.monitors[1]
    if 0 < crop_center < 1:
        cw, ch = int(region["width"] * crop_center), int(region["height"] * crop_center)
        region = {"left": region["left"] + (region["width"] - cw) // 2,
                  "top": region["top"] + (region["height"] - ch) // 2,
                  "width": cw, "height": ch}
    shot = sct.grab(region)
    img = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
    return np.ascontiguousarray(img), src
