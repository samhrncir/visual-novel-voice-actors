"""Gives every character a fitting, distinct voice and remembers it per game."""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import Counter
from datetime import datetime

from .config import AGES, Character, Profile, norm_name, save_profile
from .reader import LineRead
from .tts.base import TTSEngine, Voice

log = logging.getLogger(__name__)

NARRATOR_KEY = ""
# The game text is English, so native-English voices sound most natural.
_ENGLISH_ACCENTS = {"american", "british", "english", "australian", "canadian", "irish", "scottish",
                    "us", "uk", "standard", "transatlantic", "new zealand"}
_STOP_WORDS = {"a", "an", "and", "the", "with", "voice", "of", "in", "her", "his", "slightly", "very"}
_NARRATOR_HINTS = {"narrat", "story", "audiobook", "calm", "warm", "clear", "measured", "deep"}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.casefold()) if w not in _STOP_WORDS and len(w) > 2}


def _age_index(age: str) -> int | None:
    return AGES.index(age) if age in AGES else None


def _tiebreak(name: str, voice_id: str) -> float:
    # Deterministic, so re-casting the same character gives the same answer.
    return hashlib.md5(f"{name}|{voice_id}".encode()).digest()[0] / 2550.0


def score_voice(character: Character, voice: Voice, uses: int, narrator_voice_id: str | None,
                is_narrator: bool = False) -> float:
    score = 0.0
    if character.gender in ("female", "male"):
        if voice.gender == character.gender:
            score += 4.0
        elif voice.gender == "neutral":
            score += 1.0
        elif voice.gender in ("female", "male"):
            score -= 6.0

    char_age, voice_age = _age_index(character.age), _age_index(voice.age)
    if char_age is not None and voice_age is not None:
        score += max(-2.0, 2.0 - abs(char_age - voice_age))
    elif is_narrator and voice.age in ("adult", "middle_aged"):
        score += 1.0

    if voice.accent.casefold() in _ENGLISH_ACCENTS:
        score += 0.8

    voice_words = _words(f"{voice.description} {' '.join(voice.tags)} {voice.name}")
    overlap = _words(character.description) & voice_words
    score += min(1.6, 0.4 * len(overlap))

    tags = " ".join(voice.tags).casefold()
    if is_narrator:
        score += 0.6 * sum(1 for hint in _NARRATOR_HINTS if hint in voice.description.casefold() or hint in tags)
    else:
        if "character" in tags or "animation" in tags:
            score += 0.5
        if "narrat" in tags:
            score -= 0.3
        if voice.id == narrator_voice_id:
            score -= 3.0
    if "auto" in voice.tags:
        score -= 1.5  # a valid choice, but prefer the named voices

    score -= 5.0 * uses
    return score + _tiebreak(character.name, voice.id)


def pick_voice(character: Character, voices: list[Voice], usage: Counter,
               narrator_voice_id: str | None = None, is_narrator: bool = False) -> Voice | None:
    if not voices:
        return None
    return max(voices, key=lambda v: score_voice(character, v, usage.get(v.id, 0), narrator_voice_id, is_narrator))


class Cast:
    """The current game's characters and their voices. Safe to use from several threads."""

    def __init__(self, profile: Profile):
        self.profile = profile
        self._lock = threading.RLock()
        self._tried: dict[tuple[str, str], set[str]] = {}  # voices already offered by "pick another"

    # -- lookups
    def get(self, key: str) -> Character | None:
        with self._lock:
            return self.profile.narrator if key == NARRATOR_KEY else self.profile.characters.get(key)

    def keys(self) -> list[str]:
        with self._lock:
            return [NARRATOR_KEY, *self.profile.characters]

    def roster(self) -> list[tuple[str, str, str]]:
        with self._lock:
            return [(c.name, c.gender, c.age) for c in self.profile.characters.values()]

    # -- updates
    def character_for(self, line: LineRead) -> tuple[str, Character, bool]:
        """(key, character, is_new) for the line's speaker; narration goes to the narrator."""
        with self._lock:
            if line.is_narration:
                return NARRATOR_KEY, self.profile.narrator, False
            key = norm_name(line.speaker)
            character = self.profile.characters.get(key)
            is_new = character is None
            if is_new:
                character = Character(
                    name=line.speaker,
                    gender=line.speaker_gender if line.speaker_gender in ("female", "male") else "unknown",
                    age=line.speaker_age if line.speaker_age in AGES else "unknown",
                    description=line.speaker_voice,
                    first_seen=datetime.now().isoformat(timespec="seconds"),
                )
                self.profile.characters[key] = character
            else:
                # Fill in anything we didn't know yet; never overwrite what the user set.
                if character.gender == "unknown" and line.speaker_gender in ("female", "male"):
                    character.gender = line.speaker_gender
                if character.age == "unknown" and line.speaker_age in AGES:
                    character.age = line.speaker_age
                if not character.description and line.speaker_voice:
                    character.description = line.speaker_voice
            character.lines += 1
            return key, character, is_new

    def voice_for(self, key: str, engine: TTSEngine) -> Voice | None:
        """The character's voice on this engine, casting one if needed."""
        voices = engine.voices()
        if not voices:
            return None
        by_id = {v.id: v for v in voices}
        with self._lock:
            character = self.get(key)
            if character is None:
                return None
            current = by_id.get(character.voices.get(engine.id, ""))
            if current is not None:
                return current
            voice = self._cast(key, character, engine.id, voices)
            if voice is not None:
                character.voices[engine.id] = voice.id
                self.save()
            return voice

    def recast(self, key: str, engine: TTSEngine) -> Voice | None:
        """Give the character a voice it hasn't had yet; once all have been tried, start over."""
        voices = engine.voices()
        with self._lock:
            character = self.get(key)
            if character is None or not voices:
                return None
            previous = character.voices.pop(engine.id, None)
            tried = self._tried.setdefault((key, engine.id), set())
            if previous:
                tried.add(previous)
            choices = [v for v in voices if v.id not in tried]
            if not choices:
                tried.clear()
                tried.add(previous or "")
                choices = [v for v in voices if v.id != previous] or voices
            voice = self._cast(key, character, engine.id, choices)
            if voice is not None:
                character.voices[engine.id] = voice.id
                self.save()
            return voice

    def set_voice(self, key: str, engine_id: str, voice_id: str) -> None:
        with self._lock:
            character = self.get(key)
            if character is not None:
                character.voices[engine_id] = voice_id
                self.save()

    def update(self, key: str, **fields) -> None:
        with self._lock:
            character = self.get(key)
            if character is not None:
                for name, value in fields.items():
                    setattr(character, name, value)
                self.save()

    def remove(self, key: str) -> None:
        with self._lock:
            if key != NARRATOR_KEY:
                self.profile.characters.pop(key, None)
                self.save()

    def save(self) -> None:
        with self._lock:
            try:
                save_profile(self.profile)
            except OSError:
                # e.g. antivirus or OneDrive briefly locking the file; the next save catches up.
                log.warning("Couldn't save the cast for %s", self.profile.name, exc_info=True)

    # -- internals
    def _cast(self, key: str, character: Character, engine_id: str, voices: list[Voice]) -> Voice | None:
        usage: Counter = Counter()
        narrator_voice = self.profile.narrator.voices.get(engine_id)
        for other_key in self.keys():
            other = self.get(other_key)
            if other_key != key and other is not None and engine_id in other.voices:
                usage[other.voices[engine_id]] += 1
        return pick_voice(character, voices, usage, narrator_voice, is_narrator=key == NARRATOR_KEY)
