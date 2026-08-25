"""Voice-reference and conversion request models."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class VoiceConversionRequest(BaseModel):
    """Validated audio-to-audio singing voice conversion request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_audio: Path
    target_f0: Path
    reference_audio: Path
    reference_f0: Path
    reference_id: str = Field(min_length=1, max_length=100)
    engine: str = "soulx"


class VoiceReference(BaseModel):
    """Curated artist-specific vocal reference and immutable lineage metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    created_at: datetime
    source_file: Path
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    audio_file: Path
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    f0_file: Path
    f0_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tags: tuple[str, ...] = ()


class ResolvedVoiceReference(BaseModel):
    """Integrity-checked voice reference with artist-confined absolute paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: VoiceReference
    source: Path
    audio: Path
    f0: Path
