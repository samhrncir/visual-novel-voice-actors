"""Free local voices on the GPU with Qwen3-TTS 1.7B (via the faster-qwen3-tts runtime).

Two modes:
- "custom": 9 built-in speakers; each line gets a plain-English acting instruction.
- "design": voices described in words (unlimited); the description and the acting
  instruction are combined per line. The timbre can drift slightly between lines,
  so each voice uses a fixed random seed to keep it as steady as possible.
"""
from __future__ import annotations

import gc
import hashlib
import logging

import numpy as np

from .base import AudioStream, EngineError, SpeechRequest, TTSEngine, Voice, acting_instruction, strip_tags

log = logging.getLogger(__name__)

MODEL_IDS = {
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
SAMPLE_RATE = 24000
LANGUAGE = "English"
CHUNK_SIZE = 8  # ~0.67 s of audio per streamed chunk

BUILTIN_VOICES: list[Voice] = [
    Voice("aiden", "Aiden", "male", "young_adult", "American", "sunny, clear midrange"),
    Voice("ryan", "Ryan", "male", "young_adult", "English", "dynamic, rhythmic, energetic"),
    Voice("dylan", "Dylan", "male", "young_adult", "Chinese", "youthful, clear, natural"),
    Voice("eric", "Eric", "male", "young_adult", "Chinese", "lively, slightly husky"),
    Voice("uncle_fu", "Uncle Fu", "male", "middle_aged", "Chinese", "seasoned, low, mellow"),
    Voice("serena", "Serena", "female", "young_adult", "Chinese", "warm, gentle"),
    Voice("vivian", "Vivian", "female", "young_adult", "Chinese", "bright, slightly edgy"),
    Voice("ono_anna", "Ono Anna", "female", "teen", "Japanese", "playful, light, nimble"),
    Voice("sohee", "Sohee", "female", "young_adult", "Korean", "warm, emotional"),
]


def _design(slug: str, name: str, gender: str, age: str, description: str, *tags: str) -> Voice:
    return Voice(f"design:{slug}", name, gender, age, "", description, tags)


DESIGN_VOICES: list[Voice] = [
    _design("auto", "From character description", "unknown", "unknown",
            "Uses the voice Claude described for this character", "auto"),
    _design("bright_teen_girl", "Bright teen girl", "female", "teen",
            "Bright, bubbly teenage girl with a high, energetic voice"),
    _design("shy_teen_girl", "Shy teen girl", "female", "teen",
            "Soft-spoken, shy teenage girl with a gentle, slightly breathy voice"),
    _design("cool_teen_girl", "Cool teen girl", "female", "teen",
            "Cool, composed teenage girl with a low, even, slightly aloof voice"),
    _design("cheerful_woman", "Cheerful young woman", "female", "young_adult",
            "Cheerful, energetic young woman with a clear, lively voice"),
    _design("elegant_woman", "Elegant young woman", "female", "young_adult",
            "Elegant, refined young woman with a smooth, poised voice"),
    _design("tomboy", "Tomboyish young woman", "female", "young_adult",
            "Confident, tomboyish young woman with a slightly husky, playful voice"),
    _design("warm_woman", "Warm woman", "female", "adult",
            "Warm, caring woman in her thirties with a mellow, gentle voice"),
    _design("sultry_woman", "Mature woman", "female", "adult",
            "Mature, confident woman with a low, velvety voice"),
    _design("stern_woman", "Stern woman", "female", "middle_aged",
            "Stern, authoritative middle-aged woman with a crisp, precise voice"),
    _design("little_girl", "Little girl", "female", "child",
            "Cheerful little girl around ten years old with a high, sweet voice"),
    _design("old_woman", "Old woman", "female", "elderly",
            "Kind elderly woman with a soft, slightly wavering voice"),
    _design("energetic_teen_boy", "Energetic teen boy", "male", "teen",
            "Energetic teenage boy with a bright, slightly raspy voice"),
    _design("quiet_teen_boy", "Quiet teen boy", "male", "teen",
            "Calm, quiet teenage boy with a soft, even voice"),
    _design("earnest_man", "Earnest young man", "male", "young_adult",
            "Friendly, earnest young man with a clear, warm voice"),
    _design("cool_man", "Cool young man", "male", "young_adult",
            "Cool, aloof young man with a low, smooth voice"),
    _design("cheeky_man", "Cheeky young man", "male", "young_adult",
            "Cheeky, playful young man with a quick, bright voice"),
    _design("deep_man", "Confident man", "male", "adult",
            "Confident man in his thirties with a deep, resonant voice"),
    _design("gentle_man", "Gentle man", "male", "adult",
            "Gentle, soft-spoken man with a warm, calm voice"),
    _design("gruff_man", "Gruff man", "male", "middle_aged",
            "Gruff, weary middle-aged man with a rough, gravelly voice"),
    _design("little_boy", "Little boy", "male", "child",
            "Lively little boy around ten years old with a high, clear voice"),
    _design("old_man", "Old man", "male", "elderly",
            "Wise old man with a slow, raspy, kindly voice"),
    _design("narrator_female", "Female narrator", "female", "adult",
            "Calm female narrator with a warm, clear, even voice", "narration"),
    _design("narrator_male", "Male narrator", "male", "adult",
            "Calm male narrator with a rich, measured voice", "narration"),
]


def _seed_for(voice_id: str) -> int:
    return int.from_bytes(hashlib.sha1(voice_id.encode()).digest()[:4], "little")


class QwenLocalEngine(TTSEngine):
    id = "local"

    def __init__(self, mode: str = "custom", device: str = "cuda"):
        self.mode = mode if mode in MODEL_IDS else "custom"
        # Each mode has its own voices, so characters remember a voice per mode.
        self.id = "local" if self.mode == "custom" else "local_design"
        self.device = device or "cuda"
        self.display_name = "Local GPU (built-in voices)" if self.mode == "custom" else "Local GPU (designed voices)"
        self._model = None
        self._closed = False
        self._voices = list(BUILTIN_VOICES if self.mode == "custom" else DESIGN_VOICES)
        self.lines_spoken = 0

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def is_downloaded(self) -> bool:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(MODEL_IDS[self.mode], local_files_only=True)
            return True
        except Exception:
            return False

    def prepare(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from faster_qwen3_tts import FasterQwen3TTS
        except ImportError as e:
            raise EngineError(
                "Local voices aren't installed. Run setup.bat again (it installs PyTorch and Qwen3-TTS)."
            ) from e
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise EngineError("Local voices need an NVIDIA GPU, and PyTorch can't see one.")
        try:
            model = FasterQwen3TTS.from_pretrained(
                MODEL_IDS[self.mode], device=self.device, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            model.warmup(prefill_len=100)  # captures CUDA graphs; makes later lines fast
            if self.mode == "custom":
                supported = {s.lower() for s in (model.model.get_supported_speakers() or [])}
                self._voices = [v for v in BUILTIN_VOICES if v.id in supported] or self._voices
        except Exception as e:
            raise EngineError(f"Couldn't load the local voice model: {e}") from e
        self._model = model

    def usage_text(self) -> str:
        return f"Local GPU: {self.lines_spoken} lines this session (free)"

    def _instruction(self, request: SpeechRequest) -> str:
        instruction = acting_instruction(request)
        if request.speed >= 1.1:
            instruction += " Speaking quickly."
        elif request.speed <= 0.9:
            instruction += " Speaking slowly."
        return instruction

    def synthesize(self, request: SpeechRequest) -> AudioStream:
        if self._closed:
            # A line queued just before switching engines; don't reload the model for it.
            return AudioStream(SAMPLE_RATE, iter(()))
        if self._model is None:
            self.prepare()
        import torch

        text = strip_tags(request.text)
        instruction = self._instruction(request)
        if self.mode == "custom":
            log.info("Qwen custom voice=%s instruct=%r text=%r", request.voice.id, instruction, text)
            generator = self._model.generate_custom_voice_streaming(
                text=text, speaker=request.voice.id, language=LANGUAGE, instruct=instruction, chunk_size=CHUNK_SIZE
            )
        else:
            description = request.voice.description
            if request.voice.id == "design:auto":
                description = request.character_description or "Natural, expressive voice"
            torch.manual_seed(_seed_for(request.voice.id + description))
            prompt = f"{description.rstrip('.')}. {instruction}"
            log.info("Qwen design instruct=%r text=%r", prompt, text)
            generator = self._model.generate_voice_design_streaming(
                text=text, instruct=prompt, language=LANGUAGE, chunk_size=CHUNK_SIZE
            )

        def chunks():
            try:
                for chunk, _sr, _timing in generator:
                    yield np.asarray(chunk, dtype=np.float32).reshape(-1)
            except Exception as e:
                raise EngineError(f"Local voice generation failed: {e}") from e
            finally:
                generator.close()  # stop generating if playback was interrupted
            self.lines_spoken += 1

        return AudioStream(SAMPLE_RATE, chunks())

    def close(self) -> None:
        self._closed = True
        if self._model is None:
            return
        self._model = None
        gc.collect()  # drop the model's tensors before returning GPU memory
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
