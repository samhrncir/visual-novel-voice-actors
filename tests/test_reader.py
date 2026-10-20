"""Claude reader (with a fake client), tone helpers, audio player and the pipeline, without network or sound."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vn_voice import config
from vn_voice.reader import (
    OUTPUT_SCHEMA,
    SYSTEM_PROMPT,
    ClaudeReader,
    LineRead,
    ReadContext,
    ReaderError,
    ReaderRefusal,
    build_request,
    estimate_cost,
)
from vn_voice.tts.base import (
    AudioStream,
    Pcm16Joiner,
    SpeechRequest,
    TTSEngine,
    Voice,
    acting_instruction,
    auto_tagged_text,
    safe_performance_text,
    strip_tags,
)

PNG = b"\x89PNG fake"
GOOD = {
    "has_dialogue": True, "speaker": "Yuki", "kind": "dialogue", "text": "I-I'm fine! Really!",
    "emotion": "embarrassed", "intensity": 2, "volume": "normal", "pace": "fast",
    "delivery": "flustered, insisting too hard", "performance_text": "[nervous] I-I'm fine! REALLY!",
    "speaker_gender": "female", "speaker_age": "teen", "speaker_voice": "bright, flustered teenage girl",
}


def response(payload: dict | None = None, stop_reason: str = "end_turn", model: str = "claude-opus-5-5"):
    usage = SimpleNamespace(input_tokens=900, output_tokens=150, cache_read_input_tokens=1100,
                            cache_creation_input_tokens=0)
    content = [SimpleNamespace(type="thinking", thinking=""),
               SimpleNamespace(type="text", text=json.dumps(payload or GOOD))]
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage, model=model)


class FakeMessages:
    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


class FakeClient:
    def __init__(self, reply):
        self.messages = FakeMessages(reply)
        self.beta = SimpleNamespace(messages=FakeMessages(reply))


# ------------------------------------------------------------------ request shape

def test_request_has_cached_system_prompt_schema_and_images_first():
    ctx = ReadContext(game="Test Game", roster=[("Yuki", "female", "teen")], recent=["Yuki [happy]: Hi!"],
                      want_performance_text=True)
    req = build_request("claude-opus-5-5", "low", PNG, b"jpeg", ctx)
    assert req["system"][0]["text"] == SYSTEM_PROMPT
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
    assert "thinking" not in req and "temperature" not in req
    content = req["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["media_type"] == "image/jpeg"
    text = content[2]["text"]
    assert "Game: Test Game" in text and "Yuki (female, teen)" in text and "Include performance_text." in text


def test_schema_is_strict():
    assert OUTPUT_SCHEMA["additionalProperties"] is False
    assert set(OUTPUT_SCHEMA["required"]) == set(OUTPUT_SCHEMA["properties"])


def test_opus_uses_server_side_fallback():
    client = FakeClient(response())
    result = ClaudeReader(None, "claude-opus-5-5", "low", client=client).read(PNG, None, ReadContext())
    call = client.beta.messages.calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert not client.messages.calls
    assert result.line.speaker == "Yuki" and result.line.intensity == 2
    expected = (900 * 4 + 150 * 20 + 1100 * 0.2) / 1_000_000
    assert result.cost_usd == pytest.approx(expected)


def test_haiku_uses_plain_endpoint():
    client = FakeClient(response(model="claude-haiku-5-5"))
    ClaudeReader(None, "claude-haiku-5-5", "low", client=client).read(PNG, None, ReadContext())
    assert client.messages.calls and not client.beta.messages.calls
    assert "fallbacks" not in client.messages.calls[0]


def test_refusal_and_truncation_raise():
    with pytest.raises(ReaderRefusal):
        ClaudeReader(None, "claude-haiku-5-5", client=FakeClient(response(stop_reason="refusal"))).read(
            PNG, None, ReadContext())
    with pytest.raises(ReaderError):
        ClaudeReader(None, "claude-haiku-5-5", client=FakeClient(response(stop_reason="max_tokens"))).read(
            PNG, None, ReadContext())


def test_line_read_is_forgiving():
    line = LineRead.model_validate({"has_dialogue": True, "speaker": "  Mei  ", "text": "Hi\n  there",
                                    "intensity": 9, "emotion": "Very Happy"})
    assert line.speaker == "Mei" and line.text == "Hi there" and line.intensity == 3
    assert line.emotion == "very_happy"
    assert LineRead(has_dialogue=True, speaker="", text="x").is_narration


def test_cost_estimate_unknown_model_is_zero():
    assert estimate_cost("some-other-model", SimpleNamespace(input_tokens=10, output_tokens=10)) == 0.0


# ------------------------------------------------------------------ tone helpers

VOICE = Voice("v", "V", "female", "teen")


def test_performance_text_kept_only_if_words_unchanged():
    ok = SpeechRequest(text="I-I'm fine! Really!", voice=VOICE, performance_text="[nervous] I-I'm fine! REALLY!")
    assert safe_performance_text(ok) == "[nervous] I-I'm fine! REALLY!"
    changed = SpeechRequest(text="I'm fine.", voice=VOICE, performance_text="[sighs] Honestly, I'm fine.",
                            emotion="sad", intensity=2)
    assert safe_performance_text(changed) == "[sad] I'm fine."


def test_auto_tags_and_strip():
    req = SpeechRequest(text="Get out!", voice=VOICE, emotion="angry", intensity=3, volume="shout")
    assert auto_tagged_text(req) == "[shouts] [angry] Get out!"
    assert strip_tags("[whispers] Hey... [sighs] fine.") == "Hey... fine."


def test_acting_instruction():
    req = SpeechRequest(text="Wait...", voice=VOICE, emotion="scared", intensity=3, volume="whisper",
                        delivery="voice trembling")
    assert acting_instruction(req) == "Very scared, voice trembling, whispering."
    assert acting_instruction(SpeechRequest(text="x", voice=VOICE)) == "Natural, conversational."


def test_pcm_joiner_handles_split_samples():
    joiner = Pcm16Joiner()
    data = np.array([1000, -1000, 32767], dtype="<i2").tobytes()
    first = joiner.feed(data[:3])
    rest = joiner.feed(data[3:])
    samples = np.concatenate([first, rest])
    assert samples.shape == (3,) and samples[0] == pytest.approx(1000 / 32768)


# ------------------------------------------------------------------ audio player (fake sound device)

class FakeStream:
    written: list[np.ndarray] = []

    def __init__(self, **kwargs):
        self.aborted = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, block):
        FakeStream.written.append(block.copy())
        time.sleep(0.002)

    def stop(self):
        pass

    def abort(self):
        self.aborted = True


class ToneEngine(TTSEngine):
    id = "tone"
    display_name = "Tone"

    def __init__(self, seconds=0.5):
        self.seconds = seconds
        self.calls = 0

    def voices(self):
        return [Voice("a", "A", "female", "teen"), Voice("b", "B", "male", "adult")]

    def synthesize(self, request):
        self.calls += 1
        n = int(24000 * self.seconds)
        return AudioStream(24000, iter([np.full(n // 2, 0.1, np.float32), np.full(n // 2, 0.1, np.float32)]))


@pytest.fixture
def fake_sound(monkeypatch):
    import sounddevice

    FakeStream.written = []
    monkeypatch.setattr(sounddevice, "OutputStream", FakeStream)
    return FakeStream


def test_player_records_full_lines_and_replays_without_resynthesis(fake_sound):
    from vn_voice.audio import AudioPlayer, Recording

    player = AudioPlayer()
    engine = ToneEngine(seconds=0.2)
    recording = Recording()
    done = threading.Event()
    player.speak(engine, SpeechRequest(text="hi", voice=engine.voices()[0]), recording)
    player.call(done.set)
    assert done.wait(5)
    assert recording.complete and sum(len(c) for c in recording.chunks) == int(24000 * 0.2) // 2 * 2
    done.clear()
    player.play_recording(recording)
    player.call(done.set)
    assert done.wait(5)
    assert engine.calls == 1
    player.shutdown()


def test_player_interrupts_current_line(fake_sound):
    from vn_voice.audio import AudioPlayer, Recording

    player = AudioPlayer()
    engine = ToneEngine(seconds=5)
    first = Recording()
    player.speak(engine, SpeechRequest(text="long", voice=engine.voices()[0]), first)
    time.sleep(0.2)
    second = Recording()
    done = threading.Event()
    player.speak(ToneEngine(seconds=0.1), SpeechRequest(text="next", voice=engine.voices()[0]), second)
    player.call(done.set)
    assert done.wait(5)
    assert not first.complete and second.complete
    player.shutdown()


def test_stop_drops_queued_speech_but_keeps_setup_jobs(fake_sound):
    from vn_voice.audio import AudioPlayer, Recording

    player = AudioPlayer()
    gate = threading.Event()
    player.call(gate.wait)  # hold the audio thread busy
    engine = ToneEngine(seconds=0.1)
    queued = Recording()
    player.speak(engine, SpeechRequest(text="queued", voice=engine.voices()[0]), queued, interrupt=False)
    setup_ran = threading.Event()
    player.call(setup_ran.set)
    player.stop()
    gate.set()
    assert setup_ran.wait(5)
    assert engine.calls == 0 and not queued.complete
    player.shutdown()


# ------------------------------------------------------------------ pipeline (fake reader, engine and player)

class ScriptedReader:
    def __init__(self, lines):
        self.lines = list(lines)
        self.contexts = []

    def read(self, png, jpeg, ctx):
        from vn_voice.reader import ReadResult

        self.contexts.append(ctx)
        return ReadResult(line=LineRead.model_validate(self.lines.pop(0)), model="fake", seconds=0.1, cost_usd=0.01)


class SpyPlayer:
    def __init__(self):
        self.spoken: list[SpeechRequest] = []
        self.volume = 1.0

    def speak(self, engine, request, recording=None, interrupt=True, still_valid=lambda: True):
        if still_valid():
            self.spoken.append(request)

    def stop(self):
        pass


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    from vn_voice import pipeline as pipeline_module
    from vn_voice.pipeline import Frame, Pipeline

    monkeypatch.setattr(pipeline_module, "get_secret", lambda name: "test-key")
    pipe = Pipeline(config.Settings(), config.Profile(name="Test Game"))
    pipe.player.shutdown()
    pipe.player = SpyPlayer()
    pipe.engine = ToneEngine()
    pipe.engine_ready.set()

    def run(lines, manual=False):
        reader = ScriptedReader(lines)
        pipe._reader, pipe._reader_sig = reader, ("test-key", pipe.settings.claude_model, pipe.settings.claude_effort)
        for _ in lines:
            pipe._process(Frame(crop=np.zeros((20, 20, 3), np.uint8), scene=None, manual=manual, at=0.0))
        return reader

    pipe.run = run
    return pipe


def _line(speaker, text, kind="dialogue", gender="female"):
    return {**GOOD, "speaker": speaker, "text": text, "kind": kind, "speaker_gender": gender,
            "performance_text": ""}


def test_pipeline_casts_speaks_and_skips_repeats(pipeline):
    reader = pipeline.run([
        _line("Yuki", "Good morning!"),
        _line("Yuki", "Good morning!"),  # same line seen again (e.g. a background flicker)
        _line("Haru", "Morning.", gender="male"),
    ])
    spoken = pipeline.player.spoken
    assert [r.text for r in spoken] == ["Good morning!", "Morning."]
    assert spoken[0].voice.id == "a" and spoken[1].voice.id == "b"  # female / male voices
    assert set(pipeline.profile.characters) == {"yuki", "haru"}
    assert reader.contexts[-1].roster[0][0] == "Yuki"
    assert reader.contexts[-1].recent == ["Yuki [embarrassed]: Good morning!"]
    assert pipeline.session_cost == pytest.approx(0.03)


def test_pipeline_respects_muted_narration(pipeline):
    pipeline.settings = pipeline.settings.model_copy(update={"read_narration": False})
    pipeline.run([_line("", "The rain kept falling.", kind="narration")])
    assert pipeline.player.spoken == []
    assert pipeline.history[-1].note.startswith("Narration")


def test_manual_read_repeats_the_same_line(pipeline):
    pipeline.run([_line("Yuki", "Again?")])
    pipeline.run([_line("Yuki", "Again?")], manual=True)
    assert len(pipeline.player.spoken) == 2


def test_engine_loads_on_the_audio_thread_and_shuts_down(tmp_path, monkeypatch, fake_sound):
    from PySide6.QtWidgets import QApplication

    from vn_voice import pipeline as pipeline_module

    # Cross-thread signals need an event loop. A full QApplication, so later widget tests can share it.
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(pipeline_module, "create_engine", lambda engine_id, settings: ToneEngine())
    pipe = pipeline_module.Pipeline(config.Settings(), config.Profile(name="T"))
    states: list[tuple[str, str]] = []
    pipe.engine_state.connect(lambda state, message: states.append((state, message)))

    def pump_until(count: int) -> None:
        deadline = time.monotonic() + 5
        while len(states) < count and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

    pipe.load_engine()
    pump_until(2)
    pipe.load_engine()  # switching engines closes the old one
    pump_until(4)
    pipe.shutdown()
    assert [s for s, _ in states] == ["loading", "ready", "loading", "ready"]
    assert states[0][1] == "Connecting to Tone…"


def test_lines_without_ascii_letters_are_not_mistaken_for_repeats(pipeline):
    pipeline.run([
        _line("Yuki", "..."),
        _line("Yuki", "?!"),
        _line("Haru", "...", gender="male"),  # same text, different speaker
        _line("Mei", "こんにちは"),
        _line("Mei", "さようなら"),
        _line("Mei", "さようなら"),  # a real repeat
    ])
    assert [r.text for r in pipeline.player.spoken] == ["...", "?!", "...", "こんにちは", "さようなら"]


def test_stale_skip_does_not_swallow_the_next_frame(pipeline):
    from vn_voice.pipeline import Frame

    reader = ScriptedReader([_line("Yuki", "Wait for me!"), _line("Yuki", "Wait for me!")])
    pipeline._reader = reader
    pipeline._reader_sig = ("test-key", pipeline.settings.claude_model, pipeline.settings.claude_effort)
    first = Frame(crop=np.zeros((20, 20, 3), np.uint8), scene=None, manual=False, at=0.0, seq=1)
    pipeline._frames._seq = 2  # a newer frame (same line) arrived while the first was being read
    pipeline._process(first)
    assert pipeline.player.spoken == []
    pipeline._process(Frame(crop=first.crop, scene=None, manual=False, at=0.1, seq=2))
    assert [r.text for r in pipeline.player.spoken] == ["Wait for me!"]


def test_temporary_reader_failure_schedules_a_retry(pipeline, monkeypatch):
    import queue as queue_module

    from vn_voice.pipeline import Frame

    class FailingReader:
        def read(self, *args):
            raise ReaderError("Can't reach the Claude API.", retryable=True)

    pipeline._reader = FailingReader()
    pipeline._reader_sig = ("test-key", pipeline.settings.claude_model, pipeline.settings.claude_effort)
    monkeypatch.setattr(pipeline._stop, "wait", lambda timeout=None: False)  # skip the 2 s pause
    pipeline._process(Frame(crop=np.zeros((20, 20, 3), np.uint8), scene=None, manual=False, at=0.0))
    commands = []
    while True:
        try:
            commands.append(pipeline._capture_cmds.get_nowait())
        except queue_module.Empty:
            break

    class Detector:
        reset = False

        def reset_reference(self):
            self.reset = True

    detector = Detector()
    for command in commands:
        command(detector)
    assert detector.reset and pipeline._retries == 1


def test_line_for_a_replaced_engine_is_not_spoken(pipeline):
    reader = pipeline.run([_line("Yuki", "Hello?")])
    assert len(pipeline.player.spoken) == 1
    old_engine = pipeline.engine
    pipeline.engine = ToneEngine()  # switched engines
    spoken_before = len(pipeline.player.spoken)
    pipeline.player.speak(old_engine, pipeline.history[-1].request, still_valid=lambda: old_engine is pipeline.engine)
    assert len(pipeline.player.spoken) == spoken_before
    assert reader.contexts


def test_no_dialogue_is_not_spoken(pipeline):
    pipeline.run([{**GOOD, "has_dialogue": False, "text": ""}])
    assert pipeline.player.spoken == [] and not pipeline.history
