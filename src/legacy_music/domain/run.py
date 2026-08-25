"""Durable generation-run and provenance contracts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legacy_music.domain.generation import GenerationRequest, PipelineStage
from legacy_music.paths import validate_artist_id


class RunModel(BaseModel):
    """Base model for immutable run records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunEvent(RunModel):
    """One timestamped durable pipeline transition."""

    sequence: int = Field(ge=0)
    stage: PipelineStage
    occurred_at: datetime
    message: str = Field(min_length=1)
    artifacts: dict[str, str] = Field(default_factory=dict)
    hashes: dict[str, str] = Field(default_factory=dict)


class RunManifest(RunModel):
    """Current materialized state plus append-only transition history."""

    schema_version: int = Field(default=1, ge=1)
    run_id: str = Field(pattern=r"^run-[0-9]{8}t[0-9]{6}z-[a-f0-9]{8}$")
    artist_id: str
    request: GenerationRequest
    stage: PipelineStage = PipelineStage.CREATED
    created_at: datetime
    updated_at: datetime
    events: tuple[RunEvent, ...]
    error: str | None = None

    @field_validator("artist_id")
    @classmethod
    def validate_artist(cls, value: str) -> str:
        """Require a canonical artist identity."""
        return validate_artist_id(value)


class ProvenanceManifest(RunModel):
    """Disclosure and lineage record for one synthetic output."""

    schema_version: int = Field(default=1, ge=1)
    run_id: str
    artist_id: str
    ai_assisted: bool = True
    disclosure: str = "AI-assisted music generated from an authorized artist profile."
    created_at: datetime
    prompt: str
    lyrics_sha256: str
    seed: int
    music_engine: str
    music_model: str | None = None
    music_adapter: str
    voice_engine: str | None = None
    voice_reference_id: str | None = None
    voice_reference_sha256: str | None = None
    voice_source_sha256: str | None = None
    voice_reference_f0_sha256: str | None = None
    parent_run_id: str | None = None
    output: Path
    output_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    release_approved: bool = False
