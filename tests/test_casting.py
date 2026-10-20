"""Voice casting and the per-game cast list."""
from __future__ import annotations

from collections import Counter

import pytest

from vn_voice import config
from vn_voice.casting import NARRATOR_KEY, Cast, pick_voice
from vn_voice.config import Character, Profile
from vn_voice.reader import LineRead
from vn_voice.tts.base import AudioStream, TTSEngine, Voice

VOICES = [
    Voice("f_teen", "Mia", "female", "teen", description="bright, bubbly", tags=("characters_animation",)),
    Voice("f_young", "Sara", "female", "young_adult", description="warm, gentle"),
    Voice("f_old", "Edna", "female", "elderly", description="kind, wavering"),
    Voice("m_young", "Leo", "male", "young_adult", description="earnest, clear"),
    Voice("m_mid", "Bram", "male", "middle_aged", description="gruff, gravelly, deep"),
    Voice("n_calm", "Nora", "female", "adult", description="calm, warm narrator", tags=("narrative_story",)),
]


class FakeEngine(TTSEngine):
    id = "fake"
    display_name = "Fake"

    def __init__(self, voices=VOICES):
        self._voices = list(voices)

    def voices(self):
        return list(self._voices)

    def synthesize(self, request):
        return AudioStream(24000, iter(()))


@pytest.fixture(autouse=True)
def temp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROFILES_DIR", tmp_path / "profiles")
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")


def line(speaker="", gender="unknown", age="unknown", voice="", kind="dialogue", text="Hello."):
    return LineRead(has_dialogue=True, speaker=speaker, kind=kind, text=text, speaker_gender=gender,
                    speaker_age=age, speaker_voice=voice)


def test_matches_gender_and_age():
    teen_girl = Character(name="Yuki", gender="female", age="teen")
    assert pick_voice(teen_girl, VOICES, Counter()).id == "f_teen"
    gruff = Character(name="Captain", gender="male", age="middle_aged", description="gruff old sailor")
    assert pick_voice(gruff, VOICES, Counter()).id == "m_mid"


def test_prefers_native_english_voices():
    voices = [Voice("dylan", "Dylan", "male", "young_adult", "Chinese"),
              Voice("aiden", "Aiden", "male", "young_adult", "American")]
    for name in ("Kaito", "Ren", "Sora", "Haru"):
        assert pick_voice(Character(name=name, gender="male", age="young_adult"), voices, Counter()).id == "aiden"


def test_avoids_reusing_voices():
    cast = Cast(Profile(name="Test"))
    engine = FakeEngine()
    picked = []
    for name in ("Ami", "Beth", "Cleo"):
        key, _, _ = cast.character_for(line(name, "female", "teen"))
        picked.append(cast.voice_for(key, engine).id)
    assert len(set(picked)) == 3, picked


def test_narrator_gets_narration_voice_and_characters_avoid_it():
    cast = Cast(Profile(name="Test"))
    engine = FakeEngine()
    key, character, _ = cast.character_for(line(kind="narration", text="The rain fell."))
    assert key == NARRATOR_KEY and character.name == "Narrator"
    assert cast.voice_for(NARRATOR_KEY, engine).id == "n_calm"
    key, _, _ = cast.character_for(line("Kana", "female", "adult"))
    assert cast.voice_for(key, engine).id != "n_calm"


def test_speaker_names_match_case_insensitively_and_fill_unknowns():
    cast = Cast(Profile(name="Test"))
    key1, ch, new1 = cast.character_for(line("Yuki"))
    key2, ch2, new2 = cast.character_for(line("  YUKI ", "female", "teen", "shy girl"))
    assert key1 == key2 and new1 and not new2
    assert ch2.gender == "female" and ch2.age == "teen" and ch2.description == "shy girl"
    assert ch2.lines == 2


def test_user_choices_are_not_overwritten():
    cast = Cast(Profile(name="Test"))
    key, _, _ = cast.character_for(line("Riku", "male", "teen"))
    cast.update(key, gender="female")
    cast.character_for(line("Riku", "male", "teen"))
    assert cast.get(key).gender == "female"


def test_voice_is_remembered_and_saved(tmp_path):
    profile = Profile(name="Saved Game")
    cast = Cast(profile)
    engine = FakeEngine()
    key, _, _ = cast.character_for(line("Mei", "female", "young_adult"))
    first = cast.voice_for(key, engine)
    assert cast.voice_for(key, engine) == first
    reloaded = config.load_profile("Saved Game")
    assert reloaded.characters[key].voices["fake"] == first.id


def test_recast_picks_a_different_voice():
    cast = Cast(Profile(name="Test"))
    engine = FakeEngine()
    key, _, _ = cast.character_for(line("Mei", "female", "teen"))
    first = cast.voice_for(key, engine)
    second = cast.recast(key, engine)
    assert second is not None and second.id != first.id


def test_recast_cycles_through_every_voice():
    cast = Cast(Profile(name="Test"))
    engine = FakeEngine()
    key, _, _ = cast.character_for(line("Mei", "female", "teen"))
    seen = [cast.voice_for(key, engine).id]
    for _ in range(len(VOICES) - 1):
        seen.append(cast.recast(key, engine).id)
    assert sorted(seen) == sorted(v.id for v in VOICES), seen
    assert cast.recast(key, engine).id != seen[-1]  # then it starts over without repeating the current one


def test_save_survives_a_locked_file(monkeypatch):
    from vn_voice import casting

    def locked(_profile):
        raise PermissionError("file in use")

    monkeypatch.setattr(casting, "save_profile", locked)
    Cast(Profile(name="Test")).save()  # must not raise


def test_stale_voice_id_is_recast():
    cast = Cast(Profile(name="Test"))
    engine = FakeEngine()
    key, character, _ = cast.character_for(line("Mei", "female", "teen"))
    character.voices["fake"] = "deleted-voice"
    assert cast.voice_for(key, engine).id in {v.id for v in VOICES}


def test_no_voices_means_no_casting():
    cast = Cast(Profile(name="Test"))
    key, _, _ = cast.character_for(line("Mei"))
    assert cast.voice_for(key, FakeEngine(voices=[])) is None


def test_profile_round_trip_and_slugs():
    profile = Profile(name="Doki: Club / Remix", monitor="\\\\.\\DISPLAY2", region=[1, 2, 300, 400])
    profile.characters["mei"] = Character(name="Mei", voices={"fake": "f_teen"})
    config.save_profile(profile)
    assert config.profile_path(profile.name).name == "Doki_Club_Remix.json"
    assert config.list_profiles() == ["Doki: Club / Remix"]
    loaded = config.load_profile(profile.name)
    assert loaded.region == [1, 2, 300, 400] and loaded.characters["mei"].voices == {"fake": "f_teen"}


def test_settings_ignore_unknown_keys(tmp_path):
    config.SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SETTINGS_PATH.write_text('{"engine": "local", "from_the_future": 1}', encoding="utf-8")
    assert config.load_settings().engine == "local"
