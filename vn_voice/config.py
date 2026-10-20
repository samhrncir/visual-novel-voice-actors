"""Settings, per-game profiles (cast lists) and API-key storage.

Everything lives under ``data/`` next to the app so it is easy to find, back up
or delete. API keys are kept in Windows Credential Manager (via ``keyring``),
never in the JSON files.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VNVA_DATA_DIR") or ROOT_DIR / "data")
PROFILES_DIR = DATA_DIR / "profiles"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_PATH = DATA_DIR / "settings.json"

KEYRING_SERVICE = "vn-voice-actors"
SECRET_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "elevenlabs": "ELEVENLABS_API_KEY"}

Gender = Literal["female", "male", "unknown"]
Age = Literal["child", "teen", "young_adult", "adult", "middle_aged", "elderly", "unknown"]
AGES: tuple[str, ...] = ("child", "teen", "young_adult", "adult", "middle_aged", "elderly")


class _Model(BaseModel):
    # Unknown keys are ignored so older/newer files still load.
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class Hotkeys(_Model):
    read_now: str = "F8"
    toggle: str = "F9"
    replay: str = "F10"
    stop: str = "F11"


class Settings(_Model):
    active_profile: str = "My Visual Novel"
    engine: str = "elevenlabs"

    # Claude (reads the screen and directs the tone)
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high"] = "low"
    include_scene: bool = True  # also send a small full-screen shot so Claude sees expressions
    context_lines: int = 6

    # What to voice
    read_narration: bool = True
    read_thoughts: bool = True
    interrupt_on_advance: bool = True  # stop the current line when the game moves on

    # Change detection
    start_on_launch: bool = False  # begin following the game as soon as the app opens
    poll_interval_ms: int = 200
    settle_ms: int = 450
    sensitivity: float = 0.0012  # share of the box that must change to count as a new line
    force_after_s: float = 3.0  # read anyway if the box never stops moving (0 = never)

    volume: float = 1.0
    hotkeys: Hotkeys = Field(default_factory=Hotkeys)

    # ElevenLabs
    elevenlabs_model: str = "eleven_v4"
    elevenlabs_stability: float = 0.5  # lower = more expressive, higher = steadier

    # Local GPU engine (Qwen3-TTS)
    local_mode: Literal["custom", "design"] = "custom"  # 9 built-in voices, or voices designed from text
    local_device: str = "cuda"


class Character(_Model):
    name: str
    gender: Gender = "unknown"
    age: Age = "unknown"
    description: str = ""  # short voice description from Claude, used for casting
    voices: dict[str, str] = Field(default_factory=dict)  # engine id -> voice id
    speed: float = 1.0
    lines: int = 0
    first_seen: str = ""  # ISO 8601 local time


def _default_narrator() -> Character:
    return Character(name="Narrator", age="adult", description="warm, clear storyteller")


class Profile(_Model):
    name: str
    monitor: str = ""  # Windows device name, e.g. \\.\DISPLAY2
    region: list[int] | None = None  # [left, top, width, height], physical pixels, monitor-relative
    narrator: Character = Field(default_factory=_default_narrator)
    characters: dict[str, Character] = Field(default_factory=dict)


# --------------------------------------------------------------------------- helpers

def norm_name(name: str) -> str:
    """Key used to match a speaker across lines ("YUKI " and "Yuki" are the same)."""
    return " ".join(name.casefold().split())


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "game"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


_io_lock = threading.Lock()


def load_settings() -> Settings:
    if SETTINGS_PATH.exists():
        try:
            return Settings.model_validate_json(SETTINGS_PATH.read_text(encoding="utf-8"))
        except ValueError:
            log.exception("settings.json is unreadable; using defaults")
    return Settings()


def save_settings(settings: Settings) -> None:
    with _io_lock:
        _atomic_write(SETTINGS_PATH, settings.model_dump_json(indent=2))


def profile_path(name: str) -> Path:
    return PROFILES_DIR / f"{slugify(name)}.json"


def list_profiles() -> list[str]:
    names = []
    for path in sorted(PROFILES_DIR.glob("*.json")):
        try:
            names.append(Profile.model_validate_json(path.read_text(encoding="utf-8")).name)
        except ValueError:
            log.warning("Skipping unreadable profile %s", path)
    return names


def load_profile(name: str) -> Profile:
    path = profile_path(name)
    if path.exists():
        try:
            return Profile.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            log.exception("Profile %s is unreadable; starting a fresh one", path)
    return Profile(name=name)


def save_profile(profile: Profile) -> None:
    with _io_lock:
        _atomic_write(profile_path(profile.name), profile.model_dump_json(indent=2))


# --------------------------------------------------------------------------- secrets

def get_secret(name: str) -> str | None:
    """API key from Windows Credential Manager, falling back to the environment."""
    try:
        import keyring

        value = keyring.get_password(KEYRING_SERVICE, name)
        if value:
            return value
    except Exception:  # keyring backend missing or locked
        log.debug("keyring lookup failed for %s", name, exc_info=True)
    env = SECRET_ENV_VARS.get(name)
    return os.environ.get(env) if env else None


def set_secret(name: str, value: str) -> None:
    import keyring

    if value:
        keyring.set_password(KEYRING_SERVICE, name, value)
    else:
        try:
            keyring.delete_password(KEYRING_SERVICE, name)
        except keyring.errors.PasswordDeleteError:
            pass
