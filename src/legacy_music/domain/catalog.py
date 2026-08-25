"""Versioned artist catalog contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legacy_music.paths import validate_artist_id


class CatalogModel(BaseModel):
    """Base model for immutable catalog records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioProperties(CatalogModel):
    """Validated media properties reported by FFprobe."""

    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    codec: str = Field(min_length=1)
    format_name: str = Field(min_length=1)


class CatalogSong(CatalogModel):
    """One immutable imported recording and its derived artifacts."""

    id: str = Field(pattern=r"^song-[a-z0-9-]+$")
    title: str = Field(min_length=1, max_length=500)
    source_filename: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_path: str = Field(min_length=1)
    original_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    audio: AudioProperties
    imported_at: datetime
    normalized_path: str | None = None
    normalized_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ArtistCatalog(CatalogModel):
    """Versioned catalog for exactly one artist profile."""

    schema_version: int = Field(default=1, ge=1)
    artist_id: str
    songs: tuple[CatalogSong, ...] = ()

    @field_validator("artist_id")
    @classmethod
    def validate_artist(cls, value: str) -> str:
        """Require a canonical artist identifier."""
        return validate_artist_id(value)
