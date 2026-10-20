"""Claude reads the dialogue box and directs how the line should be performed."""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import anthropic
from pydantic import BaseModel, ValidationError, field_validator

log = logging.getLogger(__name__)

MODELS: dict[str, str] = {
    "claude-opus-5-5": "Claude Opus 5.5 (most accurate)",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (faster, about half the cost)",
    "claude-haiku-5-5": "Claude Haiku 5.5 (fastest, cheapest)",
}
# $ per million tokens: input, output, cache read, cache write (5 minute). Estimates for the cost meter.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
# Server-side refusal fallback is available for these models on the Claude API.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

EMOTIONS: tuple[str, ...] = (
    "neutral", "happy", "excited", "playful", "teasing", "affectionate", "shy", "embarrassed",
    "sad", "hurt", "crying", "angry", "annoyed", "scared", "nervous", "worried", "surprised",
    "confused", "serious", "determined", "sarcastic", "smug", "bored", "tired", "pained", "relieved",
)
VOLUMES = ("whisper", "soft", "normal", "loud", "shout")
PACES = ("slow", "normal", "fast")
GENDERS = ("female", "male", "unknown")
SPEAKER_AGES = ("child", "teen", "young_adult", "adult", "middle_aged", "elderly", "unknown")

SYSTEM_PROMPT = """\
You voice-direct a visual novel in real time. Each request shows the game's dialogue box (usually with the \
speaker's nameplate) and sometimes a smaller picture of the whole scene. Transcribe the line that is currently \
displayed and decide how a voice actor should perform it. This is latency-sensitive: answer directly without \
deliberating.

Fields:
- has_dialogue: false when there is no readable line: the box is empty or hidden, or the screen shows a menu, \
title screen, choice list, save/load screen or settings. When false, leave the text fields empty.
- speaker: the name on the nameplate, exactly as written. If it matches someone in the known-character list \
(ignoring case, honorifics or styling), use the list's spelling. Use "" for narration, or when no name is shown \
and the speaker is not obvious from context. Keep "???" if that is what the nameplate says.
- kind: "dialogue" for spoken lines, "thought" for inner monologue (often in parentheses, italics or a different \
colour), "narration" for descriptive text that nobody speaks aloud.
- text: the complete line as displayed, with wrapped lines joined by single spaces. Keep stutters, ellipses, \
dashes and punctuation. Leave out the speaker name, the quotation marks or brackets around the line \
(such as 「」『』“” or ""), interface buttons (Auto, Skip, Log, Save, Load, Config and the like) and the \
click-to-continue marker. If several lines are visible at once (full-screen text), return only the newest one.
- emotion, intensity (1 subtle, 2 clear, 3 strong), volume and pace: judge them from the wording and \
punctuation, the recent lines, and the speaker's expression and pose in the scene picture when there is one.
- delivery: a short acting direction of at most 12 words, for example "teasing, smug, drawn-out vowels" or \
"barely holding back tears".
- performance_text: only when the request asks for it, otherwise "". The same words prepared for an \
expressive TTS model: insert one to three audio tags in square brackets right before the words they affect, \
such as [whispers], [sighs], [laughs], [giggles], [nervous], [excited], [sarcastic], [angry], [shouts], \
[crying], [gasps], [hesitant], [softly], [playfully], [sad]. You may write one strongly stressed word in \
capitals. Never add, drop or reorder words, and never include anything except the line itself.
- speaker_gender, speaker_age, speaker_voice: your best guess about the speaker, from the name, how they talk, \
how others address them, and their sprite if visible. These pick a fitting voice. speaker_voice is a 3 to 10 \
word voice description such as "bright, bubbly teenage girl" or "gruff, weary middle-aged man". For narration, \
describe a narrator voice that suits the game. If you recognise the game, use what you know about its characters.
"""

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "has_dialogue": {"type": "boolean"},
        "speaker": {"type": "string"},
        "kind": {"type": "string", "enum": ["dialogue", "narration", "thought"]},
        "text": {"type": "string"},
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "intensity": {"type": "integer", "enum": [1, 2, 3]},
        "volume": {"type": "string", "enum": list(VOLUMES)},
        "pace": {"type": "string", "enum": list(PACES)},
        "delivery": {"type": "string"},
        "performance_text": {"type": "string"},
        "speaker_gender": {"type": "string", "enum": list(GENDERS)},
        "speaker_age": {"type": "string", "enum": list(SPEAKER_AGES)},
        "speaker_voice": {"type": "string"},
    },
    "required": [
        "has_dialogue", "speaker", "kind", "text", "emotion", "intensity", "volume", "pace",
        "delivery", "performance_text", "speaker_gender", "speaker_age", "speaker_voice",
    ],
    "additionalProperties": False,
}


class LineRead(BaseModel):
    """One line as Claude read and directed it."""

    has_dialogue: bool
    speaker: str = ""
    kind: str = "dialogue"  # dialogue / narration / thought
    text: str = ""
    emotion: str = "neutral"
    intensity: int = 1
    volume: str = "normal"
    pace: str = "normal"
    delivery: str = ""
    performance_text: str = ""
    speaker_gender: str = "unknown"
    speaker_age: str = "unknown"
    speaker_voice: str = ""

    # Be forgiving: a fallback model may stray from the enums, and a bad label shouldn't drop a line.
    @field_validator("intensity", mode="before")
    @classmethod
    def _clamp_intensity(cls, v):
        try:
            return min(3, max(1, int(v)))
        except (TypeError, ValueError):
            return 1

    @field_validator("kind", mode="before")
    @classmethod
    def _kind(cls, v):
        v = str(v or "").strip().lower()
        return v if v in ("dialogue", "narration", "thought") else "dialogue"

    @field_validator("emotion", "volume", "pace", "speaker_gender", "speaker_age", mode="before")
    @classmethod
    def _lower(cls, v):
        return str(v or "").strip().lower().replace(" ", "_")

    @field_validator("speaker", "text", "delivery", "performance_text", "speaker_voice", mode="before")
    @classmethod
    def _strip(cls, v):
        return " ".join(str(v or "").split())

    @property
    def is_narration(self) -> bool:
        return self.kind == "narration" or not self.speaker


@dataclass
class ReadContext:
    game: str = ""
    roster: Sequence[tuple[str, str, str]] = ()  # (name, gender, age)
    recent: Sequence[str] = ()  # already formatted, oldest first
    want_performance_text: bool = False


@dataclass
class ReadResult:
    line: LineRead
    model: str
    seconds: float
    cost_usd: float
    usage: dict = field(default_factory=dict)


class ReaderError(Exception):
    """Reading failed in a way the user should hear about (bad key, quota, network)."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable  # worth trying the same line again shortly


class ReaderRefusal(ReaderError):
    """Claude declined this line."""


def _image_block(data: bytes, media_type: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(data).decode()},
    }


def build_user_text(ctx: ReadContext, has_scene: bool) -> str:
    roster = "; ".join(f"{name} ({gender}, {age.replace('_', ' ')})" for name, gender, age in ctx.roster)
    recent = "\n".join(f"- {line}" for line in ctx.recent) or "- (none yet)"
    images = "The first image is the dialogue box; the second is the whole scene." if has_scene \
        else "The image is the dialogue box."
    tags = "Include performance_text." if ctx.want_performance_text else 'Set performance_text to "".'
    return (
        f"Game: {ctx.game or 'unknown'}\n"
        f"Known characters: {roster or 'none yet'}\n"
        f"Recent lines, oldest first:\n{recent}\n\n"
        f"{images} {tags} Read the current line."
    )


def build_request(model: str, effort: str, dialogue_png: bytes, scene_jpeg: bytes | None, ctx: ReadContext) -> dict:
    content: list[dict] = [_image_block(dialogue_png, "image/png")]
    if scene_jpeg:
        content.append(_image_block(scene_jpeg, "image/jpeg"))
    content.append({"type": "text", "text": build_user_text(ctx, scene_jpeg is not None)})
    return {
        "model": model,
        "max_tokens": 4096,  # room for brief adaptive thinking plus the ~200-token answer
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }


def estimate_cost(model: str, usage) -> float:
    prices = PRICES.get(model) or next((p for m, p in PRICES.items() if model.startswith(m)), None)
    if prices is None or usage is None:
        return 0.0
    p_in, p_out, p_read, p_write = prices
    return (
        (getattr(usage, "input_tokens", 0) or 0) * p_in
        + (getattr(usage, "output_tokens", 0) or 0) * p_out
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * p_read
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * p_write
    ) / 1_000_000


class ClaudeReader:
    def __init__(self, api_key: str | None, model: str = "claude-opus-5-5", effort: str = "low", client=None):
        self.model = model
        self.effort = effort
        if client is not None:
            self._client = client
        else:
            timeout = anthropic.Timeout(45.0, connect=10.0)
            # With no key the SDK falls back to ANTHROPIC_API_KEY or an `ant auth login` profile.
            kwargs = {"api_key": api_key} if api_key else {}
            self._client = anthropic.Anthropic(timeout=timeout, max_retries=2, **kwargs)

    def read(self, dialogue_png: bytes, scene_jpeg: bytes | None, ctx: ReadContext) -> ReadResult:
        request = build_request(self.model, self.effort, dialogue_png, scene_jpeg, ctx)
        started = time.perf_counter()
        try:
            if self.model in FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **request
                )
            else:
                response = self._client.messages.create(**request)
        except anthropic.AuthenticationError as e:
            raise ReaderError("Claude rejected the API key. Check it in Settings.") from e
        except anthropic.PermissionDeniedError as e:
            raise ReaderError(f"Claude API permission denied: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise ReaderError(f"Model {self.model} isn't available to this API key.") from e
        except anthropic.RateLimitError as e:
            raise ReaderError("Claude rate limit hit; slowing down.", retryable=True) from e
        except anthropic.BadRequestError as e:
            raise ReaderError(f"Claude rejected the request: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise ReaderError(f"Claude API error {e.status_code}: {e.message}", retryable=e.status_code >= 500) from e
        except anthropic.APIConnectionError as e:
            raise ReaderError("Can't reach the Claude API. Check your internet connection.", retryable=True) from e
        seconds = time.perf_counter() - started

        served_by = getattr(response, "model", None) or self.model
        cost = estimate_cost(served_by, response.usage)
        usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "cache_read": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        }
        if response.stop_reason == "refusal":
            raise ReaderRefusal("Claude declined to read this line.")
        if response.stop_reason == "max_tokens":
            raise ReaderError("Claude's answer was cut off (max_tokens).", retryable=True)
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            line = LineRead.model_validate_json(text)
        except ValidationError as e:
            log.warning("Unparseable reader output: %r", text[:500])
            raise ReaderError("Claude returned an unreadable answer.", retryable=True) from e
        log.info(
            "read %.2fs %s in=%s out=%s cache=%s $%.4f", seconds, served_by,
            usage["input"], usage["output"], usage["cache_read"], cost,
        )
        return ReadResult(line=line, model=served_by, seconds=seconds, cost_usd=cost, usage=usage)
