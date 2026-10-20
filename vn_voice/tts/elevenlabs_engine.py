"""ElevenLabs voices (cloud). On v3/v4 models the line carries [audio tags] for the emotion."""
from __future__ import annotations

import json
import logging
import re
import threading

from ..config import CACHE_DIR
from .base import (
    AudioStream,
    EngineError,
    Pcm16Joiner,
    SpeechRequest,
    TTSEngine,
    Voice,
    safe_performance_text,
    speed_for,
    strip_tags,
)

log = logging.getLogger(__name__)

MODELS: dict[str, str] = {
    "eleven_v4": "Eleven v4 (most expressive, understands emotion tags)",
    "eleven_v3": "Eleven v3 (expressive, understands emotion tags)",
    "eleven_flash_v2_5": "Flash v2.5 (fastest and cheapest, no emotion tags)",
    "eleven_multilingual_v2": "Multilingual v2 (steady, no emotion tags)",
}
SAMPLE_RATE = 24000
OUTPUT_FORMAT = "pcm_24000"  # raw 16-bit PCM, available on every plan
VOICE_CACHE = CACHE_DIR / "elevenlabs_voices.json"

_AGES = {
    "child": "child", "kid": "child", "teen": "teen", "teenager": "teen", "young": "young_adult",
    "young_adult": "young_adult", "adult": "adult", "middle_aged": "middle_aged", "old": "elderly",
    "elderly": "elderly", "senior": "elderly",
}


def is_tag_model(model_id: str) -> bool:
    return model_id.startswith(("eleven_v3", "eleven_v4"))


def voice_from_api(v) -> Voice:
    labels = {str(k).lower(): str(val) for k, val in (getattr(v, "labels", None) or {}).items() if val}
    gender = labels.get("gender", "").strip().lower()
    if gender not in ("female", "male", "neutral"):
        gender = "unknown"
    age = _AGES.get(re.sub(r"[\s-]+", "_", labels.get("age", "").strip().lower()), "unknown")
    description = labels.get("description") or labels.get("descriptive") or getattr(v, "description", "") or ""
    tags = tuple(t for t in (labels.get("use_case") or labels.get("use case"), str(getattr(v, "category", "") or "")) if t)
    return Voice(
        id=v.voice_id,
        name=getattr(v, "name", None) or v.voice_id,
        gender=gender,
        age=age,
        accent=labels.get("accent", "").strip(),
        description=str(description)[:200],
        tags=tags,
    )


def _translate(error) -> EngineError:
    status = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, dict):
        code = str(detail.get("code") or detail.get("status") or "")
        message = str(detail.get("message") or "")
    else:
        code, message = "", str(detail or "")
    if status == 402 or "quota" in code or "credits" in code:
        return EngineError("Your ElevenLabs credits are used up for this billing period.")
    if status == 401:
        return EngineError("ElevenLabs rejected the API key (or the key lacks text-to-speech permission).")
    if status == 429:
        return EngineError("ElevenLabs is rate-limiting (too many requests at once). Try again in a moment.")
    if status == 404:
        return EngineError("That ElevenLabs voice no longer exists. Pick another voice for this character.")
    if status == 403:
        return EngineError(f"ElevenLabs refused access: {message or code}")
    return EngineError(f"ElevenLabs error {status}: {message or code or 'unknown'}")


class ElevenLabsEngine(TTSEngine):
    id = "elevenlabs"
    display_name = "ElevenLabs"

    def __init__(self, api_key: str | None, model_id: str = "eleven_v4", stability: float = 0.5, client=None):
        if client is None:
            if not api_key:
                raise EngineError("Add your ElevenLabs API key in Settings.")
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=api_key, timeout=60)
        self._client = client
        self.model_id = model_id or "eleven_v4"
        self.stability = min(1.0, max(0.0, stability))
        self._voices: list[Voice] = _load_cached_voices()
        self.chars_this_session = 0
        self.credits_left: int | None = None
        self._lines = 0

    @property
    def uses_audio_tags(self) -> bool:
        return is_tag_model(self.model_id)

    # -- setup
    def prepare(self) -> None:
        self._voices = self._fetch_voices()
        self.refresh_usage()

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def _fetch_voices(self) -> list[Voice]:
        from elevenlabs.core.api_error import ApiError

        try:
            found: list[Voice] = []
            token = None
            for _ in range(20):  # 2,000 voices is plenty
                page = self._client.voices.search(page_size=100, next_page_token=token)
                found.extend(voice_from_api(v) for v in page.voices)
                token = page.next_page_token
                if not page.has_more or not token:
                    break
        except ApiError as e:
            if e.status_code in (401, 403):
                raise _translate(e) from e
            return self._cached_or_raise(e)
        except Exception as e:  # network trouble
            return self._cached_or_raise(e)
        if not found:
            raise EngineError("Your ElevenLabs account has no voices. Add some in the ElevenLabs Voice Library.")
        try:
            VOICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            VOICE_CACHE.write_text(json.dumps([v.__dict__ for v in found], indent=1), encoding="utf-8")
        except OSError:
            log.warning("Couldn't write the ElevenLabs voice cache", exc_info=True)
        return found

    def _cached_or_raise(self, error: Exception) -> list[Voice]:
        if self._voices:
            log.warning("Using cached ElevenLabs voices (%s)", error)
            return self._voices
        raise EngineError(f"Couldn't load your ElevenLabs voices: {error}") from error

    def refresh_usage(self) -> None:
        try:
            sub = self._client.user.subscription.get()
            self.credits_left = max(0, int(sub.character_limit) - int(sub.character_count))
        except Exception:  # keys without the user_read permission can't see this; that's fine
            log.debug("ElevenLabs subscription lookup failed", exc_info=True)

    def usage_text(self) -> str:
        text = f"ElevenLabs: {self.chars_this_session:,} characters this session"
        if self.credits_left is not None:
            text += f", {self.credits_left:,} credits left"
        return text

    # -- speech
    def _voice_settings(self, request: SpeechRequest):
        from elevenlabs import VoiceSettings

        # v3/v4 use three stability steps (Creative 0.0, Natural 0.5, Robust 1.0); snap to the nearest.
        stepped = min((0.0, 0.5, 1.0), key=lambda step: abs(step - self.stability))
        if self.model_id.startswith("eleven_v3"):
            return VoiceSettings(stability=stepped)  # v3 only honours stability
        if self.model_id.startswith("eleven_v4"):
            return VoiceSettings(stability=stepped, similarity_boost=0.75)
        # Older models can't read tags, so let emotional lines vary more.
        stability = max(0.2, self.stability - 0.12 * (request.intensity - 1))
        return VoiceSettings(
            stability=round(stability, 2), similarity_boost=0.75, style=0.0,
            use_speaker_boost=True, speed=speed_for(request),
        )

    def text_for(self, request: SpeechRequest) -> str:
        return safe_performance_text(request) if self.uses_audio_tags else strip_tags(request.text)

    def synthesize(self, request: SpeechRequest) -> AudioStream:
        from elevenlabs.core.api_error import ApiError

        text = self.text_for(request)
        kwargs = {
            "voice_id": request.voice.id,
            "text": text,
            "model_id": self.model_id,
            "output_format": OUTPUT_FORMAT,
            "voice_settings": self._voice_settings(request),
        }
        if not self.uses_audio_tags and request.previous_text:
            kwargs["previous_text"] = request.previous_text  # smoother flow between lines
        log.info("ElevenLabs %s voice=%s text=%r", self.model_id, request.voice.name, text)

        def chunks():
            joiner = Pcm16Joiner()
            try:
                for data in self._client.text_to_speech.stream(**kwargs):
                    if isinstance(data, (bytes, bytearray)) and data:
                        samples = joiner.feed(bytes(data))
                        if samples is not None:
                            yield samples
            except ApiError as e:
                raise _translate(e) from e
            except EngineError:
                raise
            except Exception as e:
                raise EngineError(f"ElevenLabs request failed: {e}") from e
            self.chars_this_session += len(text)
            self._lines += 1
            if self._lines % 10 == 0:  # keep the credits counter roughly current, off the audio thread
                threading.Thread(target=self.refresh_usage, name="eleven-usage", daemon=True).start()

        return AudioStream(SAMPLE_RATE, chunks())


def _load_cached_voices() -> list[Voice]:
    try:
        raw = json.loads(VOICE_CACHE.read_text(encoding="utf-8"))
        return [Voice(**{**item, "tags": tuple(item.get("tags", ()))}) for item in raw]
    except (OSError, ValueError, TypeError):
        return []
