from __future__ import annotations

import argparse
import os
import sys


_MUTEX = None


def _single_instance() -> bool:
    """One UltraTranslate at a time: a second launch brings the running window to the front and exits."""
    global _MUTEX
    if os.name != "nt":
        return True
    import ctypes
    k = ctypes.windll.kernel32
    _MUTEX = k.CreateMutexW(None, False, r"Local\UltraTranslate.SingleInstance")
    if k.GetLastError() != 183:          # ERROR_ALREADY_EXISTS
        return True
    u = ctypes.windll.user32
    hwnd = u.FindWindowW(None, "UltraTranslate")
    if hwnd:
        u.ShowWindow(hwnd, 9)            # SW_RESTORE
        u.SetForegroundWindow(hwnd)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(prog="ultra_translate")
    ap.add_argument("--file", default="", help="play this audio file through the pipeline (visible test)")
    ap.add_argument("--autostart", action="store_true", help="start listening immediately")
    ap.add_argument("--exit-when-done", action="store_true", help="close after a --file test has finished")
    ap.add_argument("--demo-turns", action="store_true",
                    help="slowed demo: pause the test file after each line until it has been read aloud")
    args = ap.parse_args()
    if not _single_instance():
        return 0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from ultra_translate.runtime import prepare_runtime, relaunch_limited_if_needed
    prepare_runtime()
    if relaunch_limited_if_needed():
        return 0
    from PyQt6.QtWidgets import QApplication
    from ultra_translate.ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("UltraTranslate")
    win = MainWindow(autostart_file=args.file, autostart=args.autostart, exit_when_done=args.exit_when_done,
                     demo_turns=args.demo_turns)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
