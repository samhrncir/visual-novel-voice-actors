"""Voice engines. Engines are imported lazily so the app starts fast without the GPU stack."""
from __future__ import annotations

from .base import AudioStream, EngineError, SpeechRequest, TTSEngine, Voice

ENGINE_NAMES: dict[str, str] = {
    "elevenlabs": "ElevenLabs (cloud, paid)",
    "local": "Local GPU: Qwen3-TTS (free)",
}


def create_engine(engine_id: str, settings) -> TTSEngine:
    if engine_id == "elevenlabs":
        from ..config import get_secret
        from .elevenlabs_engine import ElevenLabsEngine

        return ElevenLabsEngine(get_secret("elevenlabs"), settings.elevenlabs_model, settings.elevenlabs_stability)
    if engine_id == "local":
        from .local_engine import QwenLocalEngine

        return QwenLocalEngine(settings.local_mode, settings.local_device)
    raise EngineError(f"Unknown voice engine {engine_id!r}")


__all__ = ["AudioStream", "ENGINE_NAMES", "EngineError", "SpeechRequest", "TTSEngine", "Voice", "create_engine"]
