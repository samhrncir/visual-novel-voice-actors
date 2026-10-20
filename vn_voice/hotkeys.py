"""System-wide hotkeys (they work while the game has focus), via the Win32 RegisterHotKey API."""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
from typing import Callable

log = logging.getLogger(__name__)

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_NOREPEAT = 0x4000
MODIFIERS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8, "meta": 0x8}
KEYS = {
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    **{chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
    **{f"num{d}": 0x60 + d for d in range(10)},
    "space": 0x20, "pause": 0x13, "scrolllock": 0x91, "insert": 0x2D, "delete": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22, "`": 0xC0,
}


def parse_hotkey(text: str) -> tuple[int, int]:
    """'Ctrl+Alt+R' -> (modifier flags, virtual-key code). Raises ValueError if unknown."""
    parts = [p.strip().lower() for p in text.replace("-", "+").split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    mods = 0
    for part in parts[:-1]:
        if part not in MODIFIERS:
            raise ValueError(f"unknown modifier {part!r}")
        mods |= MODIFIERS[part]
    key = parts[-1]
    if key not in KEYS:
        raise ValueError(f"unknown key {key!r}")
    return mods, KEYS[key]


class GlobalHotkeys:
    """Registers hotkeys on a private thread and calls ``callback(action)`` when one is pressed."""

    def __init__(self, bindings: dict[str, str], callback: Callable[[str], None]):
        self._bindings = dict(bindings)
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self.failed: list[str] = []  # human-readable problems (bad key name, key taken by another app)

    def start(self) -> None:
        if sys.platform != "win32":
            return
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2)

    def stop(self) -> None:
        if self._thread and self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        actions: dict[int, str] = {}
        for hotkey_id, (action, combo) in enumerate(self._bindings.items(), start=1):
            if not combo:
                continue
            try:
                mods, vk = parse_hotkey(combo)
            except ValueError as e:
                self.failed.append(f"{combo}: {e}")
                continue
            if user32.RegisterHotKey(None, hotkey_id, mods | MOD_NOREPEAT, vk):
                actions[hotkey_id] = action
            else:
                self.failed.append(f"{combo} is already used by another program")
        self._ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in actions:
                    try:
                        self._callback(actions[msg.wParam])
                    except Exception:
                        log.exception("Hotkey handler failed")
        finally:
            for hotkey_id in actions:
                user32.UnregisterHotKey(None, hotkey_id)
