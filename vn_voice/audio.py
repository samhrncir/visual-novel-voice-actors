"""One background worker that generates and plays speech, one line at a time.

All voice-engine work (model loading and synthesis) runs on this single thread,
which keeps GPU engines happy (they aren't safe to call from several threads).

Cancelling works with a generation counter: stop() bumps it, and any speech job
from an older generation is skipped (or cut off mid-line). Setup jobs submitted
with call() always run, in order.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .tts.base import AudioStream, EngineError, SpeechRequest, TTSEngine

log = logging.getLogger(__name__)

BLOCK = 1024  # frames per write; small so "stop" takes effect within ~40 ms


@dataclass
class Recording:
    """Audio kept from a line that played to the end, for free replays."""

    sample_rate: int = 0
    chunks: list[np.ndarray] = field(default_factory=list)
    complete: bool = False


@dataclass
class _Job:
    run: Callable[[], None]
    generation: int
    speech: bool  # speech can be cancelled by stop(); setup jobs can't


class AudioPlayer:
    def __init__(
        self,
        on_error: Callable[[str], None] = lambda msg: None,
        on_speaking: Callable[[bool], None] = lambda speaking: None,
    ) -> None:
        self.volume = 1.0
        self._on_error = on_error
        self._on_speaking = on_speaking
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._running_generation = -1
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()

    # -- public API (any thread)
    def call(self, fn: Callable[[], None]) -> None:
        """Run fn on the audio thread, after anything already queued. Never cancelled."""
        self._submit(fn, speech=False)

    def speak(self, engine: TTSEngine, request: SpeechRequest, recording: Recording | None = None,
              interrupt: bool = True, still_valid: Callable[[], bool] = lambda: True) -> None:
        """Synthesize and play a line. ``still_valid`` is checked on the audio thread right before
        synthesis, so a line queued for an engine that has since been replaced is dropped."""
        if interrupt:
            self.stop()

        def job() -> None:
            if still_valid():
                self._play(engine.synthesize(request), recording)

        self._submit(job, speech=True)

    def play_recording(self, recording: Recording, interrupt: bool = True) -> None:
        if interrupt:
            self.stop()
        stream = AudioStream(recording.sample_rate, iter(list(recording.chunks)))
        self._submit(lambda: self._play(stream, None), speech=True)

    def stop(self) -> None:
        """Cut off the current line and drop any speech queued so far."""
        with self._lock:
            self._generation += 1

    def shutdown(self) -> None:
        self.stop()
        self._jobs.put(None)
        self._thread.join(timeout=3)

    # -- worker
    def _submit(self, fn: Callable[[], None], speech: bool) -> None:
        with self._lock:
            self._jobs.put(_Job(fn, self._generation, speech))

    def _cancelled(self) -> bool:
        return self._running_generation != self._generation

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            if job.speech and job.generation != self._generation:
                continue  # stopped before it started
            self._running_generation = job.generation if job.speech else self._generation
            try:
                job.run()
            except EngineError as e:
                self._on_error(str(e))
            except Exception as e:  # never let one bad line kill the audio thread
                log.exception("Audio job failed")
                self._on_error(f"Audio problem: {e}")
            finally:
                self._on_speaking(False)

    def _play(self, stream: AudioStream, recording: Recording | None) -> None:
        import sounddevice as sd

        chunks = stream.chunks
        interrupted = False
        kept: list[np.ndarray] = []
        try:
            first = next(chunks, None)  # wait for audio before opening the device
            if first is None or self._cancelled():
                return
            with sd.OutputStream(samplerate=stream.sample_rate, channels=1, dtype="float32") as out:
                self._on_speaking(True)
                chunk: np.ndarray | None = first
                while chunk is not None:
                    kept.append(chunk)
                    scaled = np.clip(chunk * self.volume, -1.0, 1.0).astype(np.float32).reshape(-1, 1)
                    for start in range(0, len(scaled), BLOCK):
                        if self._cancelled():
                            interrupted = True
                            break
                        out.write(scaled[start:start + BLOCK])
                    if interrupted:
                        out.abort()
                        break
                    chunk = next(chunks, None)
                else:
                    out.stop()  # let the buffered tail finish playing
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        if recording is not None and not interrupted and kept:
            recording.sample_rate = stream.sample_rate
            recording.chunks = kept
            recording.complete = True
