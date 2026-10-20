"""Wires screen capture -> Claude -> casting -> voice on background threads, and reports to the UI."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PySide6.QtCore import QObject, Signal

from .audio import AudioPlayer, Recording
from .casting import NARRATOR_KEY, Cast
from .config import Profile, Settings, get_secret, norm_name
from .reader import ClaudeReader, LineRead, ReadContext, ReaderError, ReaderRefusal
from .screen import ChangeDetector, Grabber, clamp_region, encode_jpeg, encode_png, find_monitor
from .tts import EngineError, SpeechRequest, TTSEngine, Voice, create_engine

log = logging.getLogger(__name__)

MAX_RETRIES = 2  # re-read a line this many times after a temporary Claude failure


@dataclass
class Frame:
    crop: np.ndarray
    scene: np.ndarray | None
    manual: bool
    at: float
    seq: int = 0


@dataclass
class LineRecord:
    line: LineRead
    key: str
    character: str
    voice: Voice | None
    request: SpeechRequest | None
    engine_id: str
    spoken: bool
    note: str
    read_seconds: float
    cost: float
    at: str
    recording: Recording = field(default_factory=Recording)


class LatestFrame:
    """Holds only the newest frame, so lines the player has already clicked past are skipped."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._seq = 0

    def put(self, frame: Frame) -> None:
        with self._cond:
            self._seq += 1
            frame.seq = self._seq
            self._frame = frame
            self._cond.notify_all()

    def take(self, timeout: float) -> Frame | None:
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame

    def newer_than(self, seq: int) -> bool:
        with self._cond:
            return self._seq > seq


def normalize_text(text: str) -> str:
    """Letters and digits only (any script), for spotting the same line read twice."""
    return re.sub(r"[\W_]+", "", text.casefold())


def dedupe_key(line: LineRead) -> str:
    # Lines like "..." or "?!" have no letters at all; compare those as written.
    return f"{norm_name(line.speaker)}|{normalize_text(line.text) or line.text.strip()}"


class Pipeline(QObject):
    status = Signal(str)
    error = Signal(str)
    line_ready = Signal(object)  # LineRecord
    cast_changed = Signal()
    preview = Signal(object)  # RGB numpy array of the box that was just read
    speaking = Signal(bool)
    engine_state = Signal(str, str)  # "loading" | "ready" | "error", message
    stats_changed = Signal()
    auto_changed = Signal(bool)

    def __init__(self, settings: Settings, profile: Profile):
        super().__init__()
        self.settings = settings
        self.profile = profile
        self.cast = Cast(profile)
        self.history: deque[LineRecord] = deque(maxlen=300)
        self.session_cost = 0.0
        self.lines_read = 0
        self.engine: TTSEngine | None = None
        self.engine_ready = threading.Event()
        self.engine_error = ""
        self.auto = False
        self._reader: ClaudeReader | None = None
        self._reader_sig: tuple | None = None
        self._frames = LatestFrame()
        self._capture_cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._manual = threading.Event()
        self._geometry_version = 0
        self._stop = threading.Event()
        self._last_key = ""
        self._retries = 0
        self.player = AudioPlayer(on_error=self.error.emit, on_speaking=self.speaking.emit)
        self.player.volume = settings.volume
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._reader_loop, name="reader", daemon=True),
        ]

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        for thread in self._threads:
            thread.start()
        self.load_engine()

    def shutdown(self) -> None:
        self._stop.set()
        engine, self.engine = self.engine, None
        if engine is not None:
            self.player.call(engine.close)
        self.player.shutdown()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2)

    # ------------------------------------------------------------------ controls (UI thread)
    def set_auto(self, on: bool) -> None:
        if on and not self._has_region():
            self.auto_changed.emit(False)
            self.error.emit("Select the dialogue box first (Game screen → Select dialogue box…).")
            return
        self.auto = on
        if on:
            # Read whatever is on screen right now, then follow along.
            self._capture_cmds.put(lambda d: d.reset_reference())
        self.auto_changed.emit(on)
        self.status.emit("Following the game…" if on else "Paused. Press Start (F9) to follow the game.")

    def read_now(self) -> None:
        if not self._has_region():
            self.error.emit("Select the dialogue box first.")
            return
        self._manual.set()

    def stop_audio(self) -> None:
        self.player.stop()

    def replay_last(self) -> None:
        record = next((r for r in reversed(list(self.history)) if r.request is not None), None)
        if record is None:
            self.status.emit("Nothing to replay yet.")
            return
        self.replay(record)

    def replay(self, record: LineRecord) -> None:
        """Play a past line again: from memory if nothing changed, otherwise with the current voice."""
        engine = self.engine
        if record.request is None or engine is None:
            return
        voice = self.cast.voice_for(record.key, engine) if self.engine_ready.is_set() else None
        unchanged = record.engine_id == engine.id and (voice is None or record.voice == voice)
        if record.recording.complete and unchanged:
            self.player.play_recording(record.recording)
            return
        if voice is None:
            self.error.emit("The voice engine isn't ready yet.")
            return
        record.voice, record.engine_id = voice, engine.id
        record.request = SpeechRequest(**{**record.request.__dict__, "voice": voice})
        record.recording = Recording()
        self.player.speak(engine, record.request, record.recording, still_valid=lambda: engine is self.engine)

    def preview_voice(self, key: str) -> None:
        engine = self.engine
        if engine is None or not self.engine_ready.is_set():
            self.error.emit("The voice engine isn't ready yet.")
            return
        voice = self.cast.voice_for(key, engine)
        character = self.cast.get(key)
        if voice is None or character is None:
            return
        last = next((r for r in reversed(list(self.history)) if r.key == key and r.request is not None), None)
        if last is not None:
            request = SpeechRequest(**{**last.request.__dict__, "voice": voice})
        else:
            name = character.name
            text = "Once upon a time, our story began." if key == NARRATOR_KEY else f"Hi, I'm {name}. Nice to meet you!"
            request = SpeechRequest(text=text, voice=voice, emotion="happy", delivery="friendly and natural",
                                    character_description=character.description)
        self.player.speak(engine, request, still_valid=lambda: engine is self.engine)

    def apply_settings(self, settings: Settings, engine_changed: bool) -> None:
        self.settings = settings
        self.player.volume = settings.volume
        # Retune the change detector in place; rebuilding it would re-read the current line.
        settle, sensitivity, force = settings.settle_ms / 1000, settings.sensitivity, settings.force_after_s

        def retune(d: ChangeDetector) -> None:
            d.settle_s, d.sensitivity, d.force_after_s = settle, sensitivity, force

        self._capture_cmds.put(retune)
        if engine_changed:
            self.load_engine()
        self.stats_changed.emit()

    def set_profile(self, profile: Profile) -> None:
        self.player.stop()
        self.profile = profile
        self.cast = Cast(profile)
        self.history.clear()
        self._last_key = ""
        self._geometry_version += 1
        self.cast_changed.emit()

    def geometry_changed(self) -> None:
        """Call after changing the profile's monitor or region."""
        self.cast.save()
        self._geometry_version += 1

    def load_engine(self) -> None:
        old = self.engine
        self.engine_ready.clear()
        self.engine_error = ""
        self.player.stop()
        try:
            engine = create_engine(self.settings.engine, self.settings)
        except EngineError as e:
            self.engine = None
            self._engine_failed(str(e))
            if old is not None:
                self.player.call(old.close)
            return
        self.engine = engine
        self.engine_state.emit("loading", self._loading_message(engine))

        def job() -> None:
            if old is not None:
                old.close()
            if self.engine is not engine:
                return  # replaced again before it even started loading
            try:
                engine.prepare()
            except EngineError as e:
                if self.engine is engine:
                    self._engine_failed(str(e))
                return
            except Exception as e:  # unexpected library failure; keep the app usable
                log.exception("Voice engine failed to load")
                if self.engine is engine:
                    self._engine_failed(f"The voice engine failed to load: {e}")
                return
            if self.engine is engine:
                self.engine_ready.set()
                self.engine_state.emit("ready", f"{engine.display_name}: {len(engine.voices())} voices ready")
                self.cast_changed.emit()
                self.stats_changed.emit()

        self.player.call(job)

    def _engine_failed(self, message: str) -> None:
        self.engine_error = message
        self.engine_state.emit("error", message)

    @staticmethod
    def _loading_message(engine: TTSEngine) -> str:
        is_downloaded = getattr(engine, "is_downloaded", None)
        if is_downloaded is not None:
            if not is_downloaded():
                return "Downloading the local voice model (about 4.5 GB, first time only)…"
            return "Loading the local voice model onto your GPU (about 15 seconds)…"
        return f"Connecting to {engine.display_name}…"

    # ------------------------------------------------------------------ capture thread
    def _has_region(self) -> bool:
        return bool(self.profile.monitor and self.profile.region)

    def _resolve_geometry(self):
        if not self._has_region():
            return None, None
        monitor = find_monitor(self.profile.monitor)
        if monitor is None:
            return None, None
        return monitor, clamp_region(self.profile.region, monitor)

    def _new_detector(self) -> ChangeDetector:
        s = self.settings
        return ChangeDetector(settle_s=s.settle_ms / 1000, sensitivity=s.sensitivity, force_after_s=s.force_after_s)

    def _capture_loop(self) -> None:
        try:
            grabber = Grabber()
        except Exception as e:
            self.error.emit(f"Screen capture isn't available: {e}")
            return
        detector = self._new_detector()
        monitor = region = None
        version = -1
        failures = 0
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    while True:
                        try:
                            self._capture_cmds.get_nowait()(detector)
                        except queue.Empty:
                            break
                    if version != self._geometry_version:
                        version = self._geometry_version
                        monitor, region = self._resolve_geometry()
                        detector = self._new_detector()
                    if monitor is None or region is None:
                        self._manual.clear()
                        self._stop.wait(0.5)
                        continue
                    manual = self._manual.is_set()
                    if manual:
                        self._manual.clear()
                        detector.force_next()
                    crop = grabber.grab_region(monitor, region)
                    if detector.update(crop, started) and (self.auto or manual):
                        scene = grabber.grab_monitor(monitor) if self.settings.include_scene else None
                        self._frames.put(Frame(crop=crop, scene=scene, manual=manual, at=started))
                    failures = 0
                except Exception as e:
                    # A monitor unplugged or a capture glitch: report it once, keep trying.
                    failures += 1
                    log.warning("Screen capture problem: %s", e)
                    if failures == 3:
                        self.error.emit(f"Can't capture the game screen: {e}")
                    version = -1  # look the monitor up again
                    self._stop.wait(1.0)
                    continue
                elapsed = time.monotonic() - started
                self._stop.wait(max(0.02, self.settings.poll_interval_ms / 1000 - elapsed))
        finally:
            grabber.close()

    # ------------------------------------------------------------------ reader thread
    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._frames.take(timeout=0.5)
            if frame is None:
                continue
            try:
                self._process(frame)
            except Exception as e:
                log.exception("Processing a line failed")
                self.error.emit(f"Unexpected problem: {e}")

    def _get_reader(self) -> ClaudeReader | None:
        key = get_secret("anthropic")
        sig = (key, self.settings.claude_model, self.settings.claude_effort)
        if sig != self._reader_sig:
            self._reader_sig = sig
            self._reader = ClaudeReader(key, self.settings.claude_model, self.settings.claude_effort) if key else None
        return self._reader

    def _recent_context(self) -> list[str]:
        n = max(0, self.settings.context_lines)
        recent = list(self.history)[-n:] if n else []
        return [f"{r.character or 'Narration'} [{r.line.emotion}]: {r.line.text[:200]}" for r in recent]

    def _should_speak(self, line: LineRead) -> tuple[bool, str]:
        if line.kind == "narration" and not self.settings.read_narration:
            return False, "Narration (muted in Settings)"
        if line.kind == "thought" and not self.settings.read_thoughts:
            return False, "Inner thought (muted in Settings)"
        return True, ""

    def _wait_for_engine(self) -> TTSEngine | None:
        if self.engine_ready.is_set():
            return self.engine
        if self.engine is None or self.engine_error:
            self.error.emit(self.engine_error or "The voice engine isn't set up. Check Settings.")
            return None
        self.status.emit("Waiting for the voice engine to finish loading…")
        deadline = time.monotonic() + 600
        while not self.engine_ready.wait(0.5):
            if self._stop.is_set() or self.engine_error or time.monotonic() > deadline:
                return None
        return self.engine

    def _retry_later(self, frame: Frame) -> None:
        """After a temporary failure, read the same box again unless the game has moved on."""
        if frame.manual or self._retries >= MAX_RETRIES:
            return
        self._retries += 1
        self._stop.wait(2.0)
        if not self._frames.newer_than(frame.seq):
            self._capture_cmds.put(lambda d: d.reset_reference())

    def _process(self, frame: Frame) -> None:
        reader = self._get_reader()
        if reader is None:
            self.error.emit("Add your Anthropic API key in Settings so Claude can read the screen.")
            return
        engine = self._wait_for_engine()
        if engine is None:
            return
        cast, profile = self.cast, self.profile  # one consistent game even if the user switches mid-line

        self.preview.emit(frame.crop)
        dialogue_png = encode_png(frame.crop)
        scene_jpeg = encode_jpeg(frame.scene) if frame.scene is not None else None
        context = ReadContext(
            game=profile.name,
            roster=cast.roster(),
            recent=self._recent_context(),
            want_performance_text=engine.uses_audio_tags,
        )
        self.status.emit("Reading the dialogue…")
        try:
            result = reader.read(dialogue_png, scene_jpeg, context)
        except ReaderRefusal as e:
            self.status.emit(str(e))
            return
        except ReaderError as e:
            self.error.emit(str(e))
            if e.retryable:
                self._retry_later(frame)
            return
        self._retries = 0
        self.session_cost += result.cost_usd
        self.lines_read += 1
        self.stats_changed.emit()

        line = result.line
        if not line.has_dialogue or not line.text:
            self.status.emit("No dialogue on screen.")
            if not frame.manual:
                self._capture_cmds.put(lambda d: d.penalize())
            return
        key_text = dedupe_key(line)
        if not frame.manual and key_text == self._last_key:
            self._capture_cmds.put(lambda d: d.penalize())
            self.status.emit("Same line as before.")
            return
        self._capture_cmds.put(lambda d: d.reward())
        if self.settings.interrupt_on_advance and not frame.manual and self._frames.newer_than(frame.seq):
            log.info("Skipped a line the player already clicked past: %r", line.text[:60])
            return
        if cast is not self.cast:
            return  # the user switched games while Claude was reading
        self._last_key = key_text

        previous = self.history[-1] if self.history else None
        key, character, is_new = cast.character_for(line)
        voice = cast.voice_for(key, engine)
        speak, note = self._should_speak(line)
        request = None
        if voice is None:
            speak, note = False, "No voice available for this engine"
        elif engine is not self.engine:
            speak, note = False, "Voice engine changed while reading"
        if voice is not None:
            request = SpeechRequest(
                text=line.text,
                voice=voice,
                performance_text=line.performance_text,
                emotion=line.emotion,
                intensity=line.intensity,
                volume=line.volume,
                pace=line.pace,
                delivery=line.delivery,
                kind=line.kind,
                speed=character.speed,
                character_description=character.description,
                previous_text=previous.line.text if previous is not None and previous.key == key else "",
            )
        record = LineRecord(
            line=line, key=key, character=character.name if key != NARRATOR_KEY else "", voice=voice,
            request=request, engine_id=engine.id, spoken=speak, note=note,
            read_seconds=result.seconds, cost=result.cost_usd, at=datetime.now().strftime("%H:%M:%S"),
        )
        if speak and request is not None:
            self.player.speak(engine, request, record.recording, interrupt=self.settings.interrupt_on_advance,
                              still_valid=lambda: engine is self.engine)
        self.history.append(record)
        self.line_ready.emit(record)
        if is_new:
            self.cast_changed.emit()
        self.status.emit("Speaking…" if speak else note)
        cast.save()
