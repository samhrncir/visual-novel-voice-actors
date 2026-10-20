"""Change detection against rendered dialogue boxes: typewriter text, blinking arrows, blanks, animation."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from vn_voice.screen import ChangeDetector, changed_fraction, clamp_region, Monitor, signature

W, H = 1500, 300
STEP = 0.2  # seconds between captures (the default poll interval)


def _font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


FONT, NAME_FONT = _font(30), _font(32)


def render(text: str, name: str = "", arrow: bool = False, arrow_size: int = 30) -> np.ndarray:
    img = Image.new("RGB", (W, H), (28, 26, 40))
    draw = ImageDraw.Draw(img)
    if name:
        draw.text((40, 20), name, font=NAME_FONT, fill=(255, 200, 120))
    draw.multiline_text((40, 80), text, font=FONT, fill=(240, 240, 240), spacing=10)
    if arrow:
        x, y, s = 1470 - arrow_size, 280 - arrow_size, arrow_size
        draw.polygon([(x, y), (x + s, y), (x + s // 2, y + s)], fill=(255, 255, 255))
    return np.asarray(img)


class Sim:
    """Feeds frames to a detector at the poll interval and records when it fires."""

    def __init__(self, **kwargs):
        self.detector = ChangeDetector(**kwargs)
        self.t = 0.0
        self.fired: list[tuple[float, str]] = []

    def frame(self, rgb: np.ndarray, label: str = "") -> bool:
        hit = self.detector.update(rgb, self.t)
        if hit:
            self.fired.append((self.t, label))
        self.t += STEP
        return hit

    def type_line(self, text: str, name: str = "", cps: float = 40):
        shown = 0.0
        while shown < len(text):
            self.frame(render(text[: int(shown)], name), f"partial:{text[: int(shown)]}")
            shown += cps * STEP
        self.hold(text, name, seconds=0.0)

    def hold(self, text: str, name: str = "", seconds: float = 2.0, blink: bool = False, arrow_size: int = 30):
        start, end = self.t, self.t + seconds
        while True:
            arrow = blink and int((self.t - start) / 0.5) % 2 == 0  # arrow shows as soon as text completes
            self.frame(render(text, name, arrow, arrow_size), f"full:{text}")
            if self.t >= end:
                break


LINE_A = "I can't believe you actually came all this way just to see me...\nYou're such an idiot, you know that?"
LINE_B = "Hey, wait up! You forgot your bag at the station."


def test_typewriter_line_fires_once_after_it_finishes():
    sim = Sim()
    sim.type_line(LINE_A, "Yuki")
    sim.hold(LINE_A, "Yuki", seconds=3)
    assert [label for _, label in sim.fired] == [f"full:{LINE_A}"]


def test_slow_typewriter_does_not_fire_mid_line():
    sim = Sim()
    sim.type_line(LINE_B, "Haru", cps=20)
    sim.hold(LINE_B, "Haru", seconds=2)
    assert [label for _, label in sim.fired] == [f"full:{LINE_B}"]


@pytest.mark.parametrize("arrow_size", [20, 30, 50])
def test_blinking_arrow_is_ignored(arrow_size):
    sim = Sim()
    sim.type_line(LINE_A, "Yuki")
    sim.hold(LINE_A, "Yuki", seconds=8, blink=True, arrow_size=arrow_size)
    assert [label for _, label in sim.fired] == [f"full:{LINE_A}"]


@pytest.mark.parametrize("arrow_size", [20, 30, 50])
def test_each_new_line_fires_once(arrow_size):
    sim = Sim()
    lines = [(LINE_A, "Yuki"), (LINE_B, "Haru"), ("...", "Yuki"), ("Really?", "Haru"), ("Yes.", "Haru"),
             ("No.", "Haru")]
    for text, name in lines:
        sim.type_line(text, name)
        sim.hold(text, name, seconds=3, blink=True, arrow_size=arrow_size)
    assert [label for _, label in sim.fired] == [f"full:{text}" for text, _ in lines]


def test_instant_text_fires_after_settling():
    sim = Sim(settle_s=0.45)
    sim.hold(LINE_B, "Haru", seconds=2)
    assert len(sim.fired) == 1
    assert sim.fired[0][0] <= 0.8


def test_blank_box_never_fires():
    sim = Sim()
    blank = np.full((H, W, 3), 28, dtype=np.uint8)
    for _ in range(20):
        sim.frame(blank)
    assert sim.fired == []


def _with_noise(frame: np.ndarray, rows: slice, rng) -> np.ndarray:
    frame = frame.copy()
    height = len(range(*rows.indices(H)))
    frame[rows, :, :] = rng.integers(0, 255, size=(height, W, 3), dtype=np.uint8)
    return frame


def test_partly_animated_box_is_learned_and_lines_still_fire():
    rng = np.random.default_rng(1)
    sim = Sim(force_after_s=2.0)
    for text, name in [(LINE_B, "Haru"), ("Fine, I'll come with you.", "Yuki")]:
        shown = 0.0
        while shown < len(text):
            sim.frame(_with_noise(render(text[: int(shown)], name), slice(220, 300), rng), "partial")
            shown += 40 * STEP
        for _ in range(int(5 / STEP)):
            sim.frame(_with_noise(render(text, name), slice(220, 300), rng), f"full:{text}")
    assert [label for _, label in sim.fired] == [f"full:{LINE_B}", "full:Fine, I'll come with you."]


def test_mostly_animated_box_forces_reads_with_backoff():
    rng = np.random.default_rng(0)
    sim = Sim(force_after_s=2.0)
    base = render(LINE_A, "Yuki")

    def animated():
        return _with_noise(base, slice(60, 300), rng)  # 80% of the box keeps changing

    seen = 0
    for _ in range(int(16 / STEP)):
        sim.frame(animated())
        if len(sim.fired) > seen:
            seen = len(sim.fired)
            sim.detector.penalize()  # the reader keeps finding the same text
    times = [t for t, _ in sim.fired]
    assert len(times) >= 2, "should force reads when the box never stops moving"
    assert times[0] == pytest.approx(2.0, abs=0.5)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps[0] == pytest.approx(4.0, abs=0.5), "after a penalty the next forced read waits twice as long"
    assert all(later > earlier for earlier, later in zip(gaps, gaps[1:]))


def test_force_next_and_reset_reference():
    sim = Sim()
    sim.hold(LINE_B, "Haru", seconds=2)
    assert len(sim.fired) == 1
    sim.detector.force_next()
    assert sim.frame(render(LINE_B, "Haru"))
    sim.detector.reset_reference()
    sim.hold(LINE_B, "Haru", seconds=1)
    assert len(sim.fired) == 3


def test_signature_and_fraction_basics():
    a = render(LINE_A, "Yuki")
    assert signature(a).shape == (H // 2, W // 2)
    assert changed_fraction(signature(a), signature(a)) == 0.0
    assert changed_fraction(signature(a), signature(render(LINE_B, "Haru"))) > 0.01


def test_picker_double_click_confirms_the_existing_box():
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from vn_voice.ui.region_picker import RegionPicker

    app = QApplication.instance() or QApplication([])
    image = QImage(1920, 1080, QImage.Format_RGB888)
    image.fill(0)
    picker = RegionPicker(image, None, [160, 760, 1600, 300])
    picker.resize(960, 540)  # half size, like a scaled display
    got = []
    picker.picked.connect(got.append)
    picker.show()
    app.processEvents()
    QTest.mouseClick(picker, Qt.LeftButton, pos=QPoint(400, 450))  # a stray click must not erase the box
    QTest.mouseDClick(picker, Qt.LeftButton, pos=QPoint(400, 450))
    app.processEvents()
    assert got and all(abs(a - b) <= 2 for a, b in zip(got[0], [160, 760, 1600, 300]))


@pytest.mark.parametrize(
    "region, expected",
    [([0, 0, 100, 50], (0, 0, 100, 50)), ([1900, 1000, 100, 100], (1900, 1000, 20, 80)), ([5000, 0, 10, 10], None)],
)
def test_clamp_region(region, expected):
    monitor = Monitor("\\\\.\\DISPLAY2", 1920, 0, 1920, 1080)
    assert clamp_region(region, monitor) == expected
