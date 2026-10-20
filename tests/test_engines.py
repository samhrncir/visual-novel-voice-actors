"""Voice engines against fakes: ElevenLabs request/stream handling and the local engine's prompts."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vn_voice.tts import elevenlabs_engine as el
from vn_voice.tts.base import EngineError, SpeechRequest, Voice


def api_voice(voice_id, name, **labels):
    return SimpleNamespace(voice_id=voice_id, name=name, labels=labels, description="", category="premade")


class FakeTTS:
    def __init__(self, payload: bytes, error=None):
        self.payload = payload
        self.error = error
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        # Deliberately split mid-sample to exercise re-alignment.
        for i in range(0, len(self.payload), 333):
            yield self.payload[i:i + 333]


class FakeVoices:
    def __init__(self, pages):
        self.pages = pages
        self.tokens: list = []

    def search(self, page_size, next_page_token=None):
        self.tokens.append(next_page_token)
        return self.pages[len(self.tokens) - 1]


def fake_client(payload=b"", error=None, pages=None):
    pages = pages or [SimpleNamespace(voices=[api_voice("v1", "Sarah", gender="female", age="young")],
                                      has_more=False, next_page_token=None)]
    return SimpleNamespace(
        text_to_speech=FakeTTS(payload, error),
        voices=FakeVoices(pages),
        user=SimpleNamespace(subscription=SimpleNamespace(
            get=lambda: SimpleNamespace(character_limit=100_000, character_count=12_345))),
    )


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "VOICE_CACHE", tmp_path / "voices.json")


PCM = (np.sin(np.linspace(0, 40, 2401)) * 20000).astype("<i2").tobytes()
VOICE = Voice("v1", "Sarah", "female", "young_adult")


def request(**kw):
    base = dict(text="I-I'm fine! Really!", voice=VOICE, performance_text="[nervous] I-I'm fine! REALLY!",
                emotion="embarrassed", intensity=2, pace="fast")
    return SpeechRequest(**{**base, **kw})


def test_voice_labels_are_normalised():
    v = el.voice_from_api(api_voice("x", "Bill", gender="Male", age="middle aged", accent="american",
                                    description="wise, mature", use_case="narrative_story"))
    assert (v.gender, v.age, v.accent, v.description) == ("male", "middle_aged", "american", "wise, mature")
    assert "narrative_story" in v.tags
    assert el.voice_from_api(api_voice("y", "Odd", gender="other", age="ancient")).gender == "unknown"


def test_prepare_pages_through_voices_and_reads_credits():
    pages = [
        SimpleNamespace(voices=[api_voice("a", "A", gender="female")], has_more=True, next_page_token="t2"),
        SimpleNamespace(voices=[api_voice("b", "B", gender="male")], has_more=False, next_page_token=None),
    ]
    client = fake_client(pages=pages)
    engine = el.ElevenLabsEngine(None, "eleven_v4", client=client)
    engine.prepare()
    assert [v.id for v in engine.voices()] == ["a", "b"]
    assert client.voices.tokens == [None, "t2"]
    assert engine.credits_left == 100_000 - 12_345
    # The list is cached so the app can start offline next time.
    assert [v.id for v in el._load_cached_voices()] == ["a", "b"]


def test_v4_streams_tagged_text_as_float_audio():
    client = fake_client(PCM)
    engine = el.ElevenLabsEngine(None, "eleven_v4", stability=0.4, client=client)
    stream = engine.synthesize(request())
    audio = np.concatenate(list(stream.chunks))
    assert stream.sample_rate == 24000 and audio.dtype == np.float32 and len(audio) == 2401
    call = client.text_to_speech.calls[0]
    assert call["text"] == "[nervous] I-I'm fine! REALLY!"
    assert call["output_format"] == "pcm_24000" and call["model_id"] == "eleven_v4"
    # v4 stability snaps to the nearest of its three steps (0.4 -> Natural 0.5).
    assert call["voice_settings"].model_dump(exclude_none=True) == {"stability": 0.5, "similarity_boost": 0.75}
    assert "previous_text" not in call
    assert engine.chars_this_session == len(call["text"])


@pytest.mark.parametrize("slider, sent", [(0.0, 0.0), (0.2, 0.0), (0.3, 0.5), (0.7, 0.5), (0.9, 1.0)])
def test_v3_only_sends_stepped_stability(slider, sent):
    client = fake_client(PCM)
    engine = el.ElevenLabsEngine(None, "eleven_v3", stability=slider, client=client)
    list(engine.synthesize(request()).chunks)
    assert client.text_to_speech.calls[0]["voice_settings"].model_dump(exclude_none=True) == {"stability": sent}


def test_flash_strips_tags_and_uses_settings_for_tone():
    client = fake_client(PCM)
    engine = el.ElevenLabsEngine(None, "eleven_flash_v2_5", client=client)
    list(engine.synthesize(request(text="[sighs] Fine.", previous_text="Are you okay?")).chunks)
    call = client.text_to_speech.calls[0]
    assert call["text"] == "Fine." and call["previous_text"] == "Are you okay?"
    settings = call["voice_settings"].model_dump(exclude_none=True)
    assert settings["speed"] == 1.08 and settings["stability"] < 0.5


def test_api_errors_become_friendly_messages():
    from elevenlabs.core.api_error import ApiError

    for status, body, expected in [
        (402, {"detail": {"code": "insufficient_credits", "message": "x"}}, "credits are used up"),
        (401, {"detail": {"code": "invalid_api_key", "message": "x"}}, "rejected the API key"),
        (404, {"detail": {"code": "voice_not_found", "message": "x"}}, "voice no longer exists"),
        (429, {"detail": {"code": "rate_limit_exceeded", "message": "x"}}, "rate-limiting"),
    ]:
        engine = el.ElevenLabsEngine(None, "eleven_v4", client=fake_client(error=ApiError(status_code=status,
                                                                                            body=body)))
        with pytest.raises(EngineError, match=expected):
            list(engine.synthesize(request()).chunks)


def test_missing_key_is_reported():
    with pytest.raises(EngineError, match="API key"):
        el.ElevenLabsEngine(None, "eleven_v4")


def test_local_engine_lists_voices_without_loading_the_model():
    from vn_voice.tts.local_engine import BUILTIN_VOICES, DESIGN_VOICES, QwenLocalEngine

    assert QwenLocalEngine("custom").voices() == BUILTIN_VOICES
    design = QwenLocalEngine("design").voices()
    assert design == DESIGN_VOICES and len({v.id for v in design}) == len(design)
    assert {v.gender for v in design} >= {"female", "male"}


def test_local_engine_prompts(monkeypatch):
    from vn_voice.tts import local_engine

    calls = []

    class FakeModel:
        def generate_custom_voice_streaming(self, **kw):
            calls.append(("custom", kw))
            yield np.zeros(10, np.float32), 24000, {}

        def generate_voice_design_streaming(self, **kw):
            calls.append(("design", kw))
            yield np.zeros(10, np.float32), 24000, {}

    engine = local_engine.QwenLocalEngine("custom")
    engine._model = FakeModel()
    req = SpeechRequest(text="[whispers] Wait... did you hear that?", voice=Voice("serena", "Serena"),
                        emotion="scared", intensity=3, volume="whisper", delivery="voice trembling")
    list(engine.synthesize(req).chunks)
    mode, kw = calls[-1]
    assert mode == "custom" and kw["speaker"] == "serena" and kw["text"] == "Wait... did you hear that?"
    assert kw["instruct"] == "Very scared, voice trembling, whispering."

    engine = local_engine.QwenLocalEngine("design")
    engine._model = FakeModel()
    auto = Voice("design:auto", "Auto", tags=("auto",))
    list(engine.synthesize(SpeechRequest(text="Hi.", voice=auto, character_description="gruff old sailor")).chunks)
    mode, kw = calls[-1]
    assert mode == "design" and kw["instruct"] == "gruff old sailor. Natural, conversational."
    assert engine.lines_spoken == 1
