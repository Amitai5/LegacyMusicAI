"""Song and immutable-source metadata models."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SongSource(BaseModel):
    """Immutable source-audio identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_type: str


class SongMetadata(BaseModel):
    """Catalog metadata for one artist-owned recording."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^song-[a-z0-9-]+$")
    title: str = Field(min_length=1, max_length=500)
    source: SongSource
    original: Path
    normalized: Path | None = None
    vocals: Path | None = None
    accompaniment: Path | None = None
    lyrics: Path | None = None
