"""Generation request, state, and result models."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legacy_music.paths import validate_artist_id


class GenerationModel(BaseModel):
    """Base model for immutable generation contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PipelineStage(StrEnum):
    """Durable stages in the resumable MVP generation pipeline."""

    CREATED = "created"
    MUSIC_GENERATED = "music_generated"
    VOCALS_SEPARATED = "vocals_separated"
    VOICE_CONVERTED = "voice_converted"
    MIXED = "mixed"
    COMPLETE = "complete"
    FAILED = "failed"


class MusicGenerationRequest(GenerationModel):
    """Music-engine portion of a generation request."""

    prompt: str = Field(min_length=1, max_length=10_000)
    adapter: str = "selected"
    duration_seconds: int = Field(default=180, ge=10, le=600)
    bpm: int | None = Field(default=None, ge=20, le=300)
    key: str | None = Field(default=None, max_length=50)
    seed: int = Field(ge=0, le=2**63 - 1)


class LyricsRequest(GenerationModel):
    """Reference to user-supplied lyrics kept outside the request manifest."""

    file: Path
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class VoiceGenerationRequest(GenerationModel):
    """Voice-conversion portion of a generation request."""

    enabled: bool = False
    engine: str = "soulx"
    reference: str = "auto"


class OutputRequest(GenerationModel):
    """Requested output and intermediate-retention behavior."""

    format: Literal["wav", "flac"] = "wav"
    keep_intermediates: bool = True


class GenerationRequest(GenerationModel):
    """Complete reproducible request for one generated song variant."""

    id: str = Field(pattern=r"^generation-[a-zA-Z0-9-]+$")
    artist_id: str
    music: MusicGenerationRequest
    lyrics: LyricsRequest
    voice: VoiceGenerationRequest = VoiceGenerationRequest()
    output: OutputRequest = OutputRequest()
    parent_run_id: str | None = None

    @field_validator("artist_id")
    @classmethod
    def validate_artist(cls, value: str) -> str:
        """Require an explicit canonical artist identifier."""
        return validate_artist_id(value)


class GenerationResult(GenerationModel):
    """Durable result returned by the generation pipeline."""

    run_id: str
    stage: PipelineStage
    final_audio: Path | None = None
    provenance: Path | None = None
