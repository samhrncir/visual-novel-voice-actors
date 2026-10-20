"""Shared types for voice engines, plus helpers that turn Claude's direction into engine input."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    gender: str = "unknown"  # female / male / neutral / unknown
    age: str = "unknown"  # child / teen / young_adult / adult / middle_aged / elderly / unknown
    accent: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()

    def label(self) -> str:
        bits = [
            self.gender if self.gender not in ("", "unknown") else "",
            self.age.replace("_", " ") if self.age not in ("", "unknown") else "",
            self.accent,
        ]
        bits = [b for b in bits if b]
        return f"{self.name} ({', '.join(bits)})" if bits else self.name


@dataclass
class SpeechRequest:
    text: str
    voice: Voice
    performance_text: str = ""  # text with [audio tags], for engines that support them
    emotion: str = "neutral"
    intensity: int = 1
    volume: str = "normal"
    pace: str = "normal"
    delivery: str = ""
    kind: str = "dialogue"
    speed: float = 1.0  # per-character speed multiplier
    character_description: str = ""  # Claude's voice description for this character
    previous_text: str = ""


@dataclass
class AudioStream:
    sample_rate: int
    chunks: Iterator[np.ndarray]  # float32 mono, -1..1


class EngineError(Exception):
    """A voice engine problem worth showing to the user."""


class TTSEngine(ABC):
    id: str = ""
    display_name: str = ""

    @property
    def uses_audio_tags(self) -> bool:
        """True when the engine wants Claude's [tagged] performance_text."""
        return False

    def prepare(self) -> None:
        """Slow one-time setup (model load, voice list). Runs on the audio thread."""

    @abstractmethod
    def voices(self) -> list[Voice]:
        """Voices available for casting (may be empty until prepare() has run)."""

    @abstractmethod
    def synthesize(self, request: SpeechRequest) -> AudioStream:
        """Start generating speech. Runs on the audio thread."""

    def usage_text(self) -> str:
        return ""

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- tone helpers

_TAG_RE = re.compile(r"\[[^\]]{1,40}\]")


def strip_tags(text: str) -> str:
    return " ".join(_TAG_RE.sub(" ", text).split())


def safe_performance_text(request: SpeechRequest) -> str:
    """Claude's tagged text, unless it changed the actual words (then fall back to auto tags)."""
    tagged = request.performance_text.strip()
    if tagged and _words(strip_tags(tagged)) == _words(request.text):
        return tagged
    return auto_tagged_text(request)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


_EMOTION_TAGS = {
    "happy": "happily", "excited": "excited", "playful": "playfully", "teasing": "mischievously",
    "affectionate": "warmly", "shy": "shyly", "embarrassed": "flustered", "sad": "sad", "hurt": "hurt",
    "crying": "crying", "angry": "angry", "annoyed": "annoyed", "scared": "scared", "nervous": "nervous",
    "worried": "worried", "surprised": "surprised", "confused": "confused", "serious": "serious",
    "determined": "determined", "sarcastic": "sarcastic", "smug": "smug", "bored": "bored",
    "tired": "tired", "pained": "pained", "relieved": "relieved",
}
_VOLUME_TAGS = {"whisper": "whispers", "soft": "softly", "loud": "loudly", "shout": "shouts"}


def auto_tagged_text(request: SpeechRequest) -> str:
    """Fallback audio tags built from the emotion labels."""
    tags = []
    if request.volume in _VOLUME_TAGS:
        tags.append(_VOLUME_TAGS[request.volume])
    if request.emotion in _EMOTION_TAGS and (request.intensity >= 2 or not tags):
        tags.append(_EMOTION_TAGS[request.emotion])
    prefix = "".join(f"[{t}] " for t in tags)
    return f"{prefix}{request.text}"


_INTENSITY_WORDS = {1: "slightly", 2: "", 3: "very"}
_VOLUME_WORDS = {"whisper": "whispering", "soft": "speaking softly", "loud": "speaking loudly", "shout": "shouting"}
_PACE_WORDS = {"slow": "slow pace", "fast": "fast pace"}


def acting_instruction(request: SpeechRequest) -> str:
    """Natural-language direction for instruction-following engines, e.g. 'Very scared, whispering.'"""
    parts: list[str] = []
    if request.emotion and request.emotion != "neutral":
        word = _INTENSITY_WORDS.get(request.intensity, "")
        parts.append(f"{word} {request.emotion}".strip())
    if request.delivery:
        parts.append(request.delivery.rstrip("."))
    if request.volume in _VOLUME_WORDS:
        parts.append(_VOLUME_WORDS[request.volume])
    if request.pace in _PACE_WORDS:
        parts.append(_PACE_WORDS[request.pace])
    said = " ".join(parts).casefold()
    if request.kind == "thought" and "monologue" not in said:
        parts.append("quiet, introspective inner monologue")
    elif request.kind == "narration" and "narrat" not in said and "storytell" not in said:
        parts.append("clear storytelling narration")
    if not parts:
        parts.append("natural, conversational")
    text = ", ".join(parts)
    return text[0].upper() + text[1:] + "."


def speed_for(request: SpeechRequest, low: float = 0.7, high: float = 1.2) -> float:
    pace = {"slow": 0.92, "fast": 1.08}.get(request.pace, 1.0)
    return round(min(high, max(low, request.speed * pace)), 2)


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class Pcm16Joiner:
    """Re-aligns a byte stream into whole 16-bit samples (network chunks can split a sample)."""

    leftover: bytes = field(default=b"")

    def feed(self, data: bytes) -> np.ndarray | None:
        data = self.leftover + data
        usable = len(data) - (len(data) % 2)
        self.leftover = data[usable:]
        return pcm16_to_float(data[:usable]) if usable else None
