"""Voice-reference and conversion request models."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class VoiceConversionRequest(BaseModel):
    """Validated audio-to-audio singing voice conversion request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_audio: Path
    reference_audio: Path
    reference_id: str = Field(min_length=1, max_length=100)
    engine: str = "soulx"


class VoiceReference(BaseModel):
    """Curated artist-specific vocal reference metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    file: Path
    tags: tuple[str, ...] = ()
