"""Monitor discovery, screen grabbing and "has the dialogue changed?" detection."""
from __future__ import annotations

import ctypes
import io
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

# A pixel counts as changed when its brightest channel moves by more than this.
PIXEL_THRESHOLD = 48


def enable_dpi_awareness() -> None:
    """Work in physical pixels so screen grabs line up on scaled (125%/150%) displays."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):  # PER_MONITOR_AWARE_V2
            return
    except AttributeError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except (AttributeError, OSError):
        pass


# --------------------------------------------------------------------------- monitors

@dataclass(frozen=True)
class Monitor:
    name: str  # Windows device name, e.g. \\.\DISPLAY2 (Qt screens are matched by position instead)
    left: int
    top: int
    width: int
    height: int
    primary: bool = False

    def label(self, index: int) -> str:
        tag = " (main)" if self.primary else ""
        return f"Screen {index + 1}: {self.width}x{self.height}{tag}"


def list_monitors() -> list[Monitor]:
    if sys.platform != "win32":
        return _list_monitors_mss()

    from ctypes import wintypes

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
    )
    user32 = ctypes.windll.user32
    found: list[Monitor] = []

    def _callback(hmonitor, _hdc, _rect, _data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            r = info.rcMonitor
            found.append(
                Monitor(info.szDevice, r.left, r.top, r.right - r.left, r.bottom - r.top, bool(info.dwFlags & 1))
            )
        return True

    user32.EnumDisplayMonitors(None, None, enum_proc(_callback), 0)
    found.sort(key=lambda m: (m.left, m.top))
    return found


def _list_monitors_mss() -> list[Monitor]:
    import mss

    with mss.mss() as sct:
        return [
            Monitor(f"monitor{i}", m["left"], m["top"], m["width"], m["height"], i == 1)
            for i, m in enumerate(sct.monitors[1:], start=1)
        ]


def find_monitor(name: str, monitors: list[Monitor] | None = None) -> Monitor | None:
    monitors = monitors if monitors is not None else list_monitors()
    return next((m for m in monitors if m.name == name), None)


def clamp_region(region: list[int] | tuple[int, ...], monitor: Monitor) -> tuple[int, int, int, int] | None:
    """Monitor-relative region clipped to the monitor; None if nothing is left."""
    left, top, width, height = (int(v) for v in region)
    left, top = max(0, left), max(0, top)
    width = min(width, monitor.width - left)
    height = min(height, monitor.height - top)
    if width < 8 or height < 8:
        return None
    return left, top, width, height


# --------------------------------------------------------------------------- grabbing

class Grabber:
    """Thin wrapper over mss. Create and use it on a single thread (mss handles are per-thread)."""

    def __init__(self) -> None:
        import mss

        self._sct = mss.mss()

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Absolute desktop coordinates in, H x W x 3 RGB uint8 array out."""
        shot = self._sct.grab({"left": left, "top": top, "width": width, "height": height})
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    def grab_monitor(self, monitor: Monitor) -> np.ndarray:
        return self.grab(monitor.left, monitor.top, monitor.width, monitor.height)

    def grab_region(self, monitor: Monitor, region: tuple[int, int, int, int]) -> np.ndarray:
        left, top, width, height = region
        return self.grab(monitor.left + left, monitor.top + top, width, height)

    def close(self) -> None:
        self._sct.close()


def _fit(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    return img


def encode_png(rgb: np.ndarray, max_side: int = 1568) -> bytes:
    buf = io.BytesIO()
    _fit(Image.fromarray(rgb), max_side).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def encode_jpeg(rgb: np.ndarray, max_side: int = 1024, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    _fit(Image.fromarray(rgb), max_side).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --------------------------------------------------------------------------- change detection

def signature(rgb: np.ndarray) -> np.ndarray:
    """Cheap fingerprint: every 2nd pixel, brightest channel (keeps coloured text visible)."""
    return rgb[::2, ::2].max(axis=2)


def changed_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIXEL_THRESHOLD


def changed_fraction(a: np.ndarray, b: np.ndarray, ignore: np.ndarray | None = None) -> float:
    """Share of sampled pixels that changed, optionally skipping pixels marked in ``ignore``."""
    if a.shape != b.shape:
        return 1.0
    changed = changed_mask(a, b)
    if ignore is not None:
        changed &= ~ignore
    return float(np.count_nonzero(changed)) / a.size


def looks_empty(sig: np.ndarray) -> bool:
    """A flat, featureless box (hidden or blank) has nothing worth sending to Claude."""
    return float(sig.std()) < 3.0


class ChangeDetector:
    """Decides when the dialogue box shows a *new, finished* line.

    - Pixels that keep flipping (a blinking continue-arrow, sparkles, an animated
      background) are learned and ignored, so even small real changes register.
    - A line is *finished* once everything else has stopped changing for
      ``settle_s``, so typewriter text has fully appeared.
    - It is *new* when it differs from the last accepted line by more than
      ``sensitivity`` (share of sampled pixels).
    - If the box never settles, a read is forced after ``force_after_s``;
      ``penalize()`` stretches that wait when forced reads keep finding the same text.
    """

    HALF_LIFE_S = 2.7  # how fast a pixel's recent-change count fades
    MOTION_IGNORE = 2.5  # changed ~3+ times recently: animated, not text appearing
    NEW_LINE_IGNORE = 1.8  # changed ~2+ times recently: don't let it count as a new line
    MAX_IGNORED = 0.5  # if over half the box animates, stop ignoring and rely on force_after

    def __init__(
        self,
        settle_s: float = 0.45,
        sensitivity: float = 0.0012,
        force_after_s: float = 3.0,
        motion_eps: float = 0.0002,
    ) -> None:
        self.settle_s = settle_s
        self.sensitivity = sensitivity
        self.force_after_s = force_after_s
        self.motion_eps = motion_eps
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._prev: np.ndarray | None = None
        self._prev_t = 0.0
        self._activity: np.ndarray | None = None
        self._reference: np.ndarray | None = None
        self._last_motion = float("-inf")
        self._motion_start: float | None = None
        self._backoff = 1.0
        self._force_next = False

    # -- control
    def reset_reference(self) -> None:
        """Forget the last accepted line so the current one is read again."""
        self._reference = None

    def force_next(self) -> None:
        self._force_next = True

    def penalize(self) -> None:
        self._backoff = min(self._backoff * 2, 8.0)

    def reward(self) -> None:
        self._backoff = 1.0

    # -- main entry point
    def update(self, rgb: np.ndarray, now: float) -> bool:
        sig = signature(rgb)
        if self._prev is None or sig.shape != self._prev.shape:
            if self._prev is not None:
                self._reference = None  # the box was resized
            self._history.clear()
            self._history.append((now, sig))
            self._activity = np.zeros(sig.shape, dtype=np.float32)
            self._prev, self._prev_t = sig, now
            self._last_motion = now
            return self._take_forced(sig)

        step = changed_mask(sig, self._prev)
        self._activity *= np.float32(0.5 ** (max(0.0, now - self._prev_t) / self.HALF_LIFE_S))
        self._activity += step
        motion_ignore = self._ignored(self.MOTION_IGNORE)
        self._prev, self._prev_t = sig, now

        self._history.append((now, sig))
        # Keep exactly one frame that is at least settle_s old at the front.
        while len(self._history) > 2 and self._history[1][0] <= now - self.settle_s:
            self._history.popleft()

        if motion_ignore is not None:
            step &= ~motion_ignore
        moving = np.count_nonzero(step) / sig.size > self.motion_eps
        old_t, old_sig = self._history[0]
        if not moving and now - old_t >= self.settle_s:
            # Catches slow typewriter text whose frame-to-frame steps are tiny.
            moving = changed_fraction(sig, old_sig, motion_ignore) > self.motion_eps

        if self._take_forced(sig):
            return True

        if moving:
            self._last_motion = now
            if self._motion_start is None:
                self._motion_start = now
            limit = self.force_after_s * self._backoff
            if self.force_after_s > 0 and now - self._motion_start >= limit and self._is_new(sig):
                self._accept(sig)
                self._motion_start = now
                return True
            return False

        if now - self._last_motion < self.settle_s:
            return False
        self._motion_start = None
        if self._is_new(sig):
            self._accept(sig)
            return True
        return False

    # -- internals
    def _ignored(self, limit: float) -> np.ndarray | None:
        ignore = self._activity > limit
        return None if ignore.mean() > self.MAX_IGNORED else ignore

    def _take_forced(self, sig: np.ndarray) -> bool:
        if not self._force_next:
            return False
        self._force_next = False
        self._accept(sig)
        return True

    def _is_new(self, sig: np.ndarray) -> bool:
        if looks_empty(sig):
            self._reference = sig  # remember the blank box so its return isn't "new"
            return False
        if self._reference is None:
            return True
        return changed_fraction(sig, self._reference, self._ignored(self.NEW_LINE_IGNORE)) > self.sensitivity

    def _accept(self, sig: np.ndarray) -> None:
        self._reference = sig

