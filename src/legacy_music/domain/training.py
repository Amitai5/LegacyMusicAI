"""Artist-specific dataset, training, and selected-adapter models."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from legacy_music.paths import validate_artist_id


class TrainingModel(BaseModel):
    """Base model for immutable training contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MusicTrainingRequest(TrainingModel):
    """Validated request for one isolated artist music-adapter run."""

    artist_id: str
    engine: str = "ace-step"
    rank: int = Field(default=16, ge=1, le=256)
    epochs: int = Field(default=10, ge=1, le=10_000)
    save_every: int = Field(default=5, ge=1)
    gradient_accumulation: int = Field(default=4, ge=1)
    gradient_checkpointing: bool = True
    seed: int = Field(default=42, ge=0)

    @field_validator("artist_id")
    @classmethod
    def validate_artist(cls, value: str) -> str:
        """Require an explicit canonical artist identifier."""
        return validate_artist_id(value)


class TrainingDatasetSong(TrainingModel):
    """One catalog recording copied into an ACE-Step training dataset."""

    song_id: str
    audio_file: Path
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_kind: Literal["normalized", "accompaniment"] = "normalized"
    source_artifact: Path | None = None
    source_manifest: Path | None = None
    source_manifest_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    caption_file: Path
    lyrics_file: Path | None = None


class TrainingDatasetManifest(TrainingModel):
    """Immutable artist-scoped dataset prepared for the official ACE-Step API."""

    schema_version: int = 1
    dataset_id: str = Field(pattern=r"^dataset-[0-9]{8}t[0-9]{6}z-[a-f0-9]{8}$")
    artist_id: str
    created_at: datetime
    dataset_policy_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    custom_tag: str = Field(min_length=1, max_length=200)
    caption: str = Field(min_length=1, max_length=2_000)
    is_instrumental: bool
    audio_dir: Path
    songs: tuple[TrainingDatasetSong, ...]

    @field_validator("artist_id")
    @classmethod
    def validate_dataset_artist(cls, value: str) -> str:
        """Require an explicit canonical artist identifier."""
        return validate_artist_id(value)


class SelectedMusicAdapter(TrainingModel):
    """Human-selected artist adapter with an integrity digest."""

    schema_version: int = 1
    adapter_id: str = Field(min_length=1, max_length=200)
    artist_id: str
    selected_at: datetime
    path: Path
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_training_id: str | None = None

    @field_validator("artist_id")
    @classmethod
    def validate_adapter_artist(cls, value: str) -> str:
        """Require an explicit canonical artist identifier."""
        return validate_artist_id(value)
