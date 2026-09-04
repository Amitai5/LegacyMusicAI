"""Fail-closed artist dataset exclusions and music/lead-vocal review policy."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class DatasetPolicyError(RuntimeError):
    """Raised when an artist dataset policy is invalid or denies an operation."""


class DatasetPolicyModel(BaseModel):
    """Base model for immutable dataset-curation contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceExclusion(DatasetPolicyModel):
    """One source recording that must never re-enter an active dataset."""

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    excluded_at: datetime

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Require a leaf filename so exclusions cannot escape the source directory."""
        if Path(value).name != value:
            raise ValueError("Excluded source filename must not contain a directory.")
        return value


class ExcludedInterval(DatasetPolicyModel):
    """One reviewed time range that must not contribute lead-vocal samples."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self) -> ExcludedInterval:
        """Require each interval to have positive duration."""
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Excluded interval end must be after its start.")
        return self

    def overlaps(self, start_seconds: float, end_seconds: float) -> bool:
        """Return whether a candidate range intersects this excluded interval."""
        return start_seconds < self.end_seconds and end_seconds > self.start_seconds


class SongVoicePolicy(DatasetPolicyModel):
    """Per-song lead-vocal eligibility and reviewed exclusion windows."""

    song_id: str = Field(pattern=r"^song-[a-z0-9-]+$")
    exclude_entire_song: bool = False
    reason: str | None = None
    excluded_intervals: tuple[ExcludedInterval, ...] = ()

    @model_validator(mode="after")
    def validate_exclusion_reason(self) -> SongVoicePolicy:
        """Require a reason when an entire song is excluded from voice training."""
        if self.exclude_entire_song and not self.reason:
            raise ValueError("Entire-song voice exclusions require a reason.")
        return self


class SongMusicPolicy(DatasetPolicyModel):
    """Per-song eligibility for artist music-style training."""

    song_id: str = Field(pattern=r"^song-[a-z0-9-]+$")
    exclude_from_training: bool = False
    reason: str | None = None
    excluded_at: datetime | None = None

    @model_validator(mode="after")
    def validate_exclusion_audit(self) -> SongMusicPolicy:
        """Require auditable context for every music-training exclusion."""
        if self.exclude_from_training and (not self.reason or self.excluded_at is None):
            raise ValueError("Music-training exclusions require a reason and timestamp.")
        return self


class ArtistDatasetPolicy(DatasetPolicyModel):
    """Versioned source, music, and lead-vocal policy for one artist profile."""

    schema_version: int = Field(default=1, ge=1)
    artist_id: str
    reject_live_voice_recordings: bool = True
    source_exclusions: tuple[SourceExclusion, ...] = ()
    music_songs: tuple[SongMusicPolicy, ...] = ()
    voice_songs: tuple[SongVoicePolicy, ...] = ()

    def source_exclusion(self, sha256: str, filename: str | None = None) -> SourceExclusion | None:
        """Resolve a matching source exclusion by immutable hash or exact filename."""
        normalized_filename = filename.casefold() if filename is not None else None
        return next(
            (
                item
                for item in self.source_exclusions
                if item.sha256 == sha256
                or (
                    normalized_filename is not None
                    and item.filename.casefold() == normalized_filename
                )
            ),
            None,
        )

    def require_source_allowed(self, sha256: str, filename: str | None = None) -> None:
        """Reject a source that was permanently removed from the usable dataset."""
        excluded = self.source_exclusion(sha256, filename)
        if excluded is not None:
            raise DatasetPolicyError(
                f"Source is permanently excluded from this artist dataset: {excluded.filename}."
            )

    def voice_policy(self, song_id: str) -> SongVoicePolicy | None:
        """Return the optional lead-vocal policy for one catalog song."""
        return next((item for item in self.voice_songs if item.song_id == song_id), None)

    def music_policy(self, song_id: str) -> SongMusicPolicy | None:
        """Return the optional music-style training policy for one catalog song."""
        return next((item for item in self.music_songs if item.song_id == song_id), None)

    def is_music_song_excluded(self, song_id: str) -> bool:
        """Return whether a song is ineligible for music-style training."""
        policy = self.music_policy(song_id)
        return policy is not None and policy.exclude_from_training

    def is_voice_song_excluded(self, song_id: str, recording_label: str = "") -> bool:
        """Return whether a song is ineligible for lead-vocal training and references."""
        policy = self.voice_policy(song_id)
        if policy is not None and policy.exclude_entire_song:
            return True
        searchable = f"{song_id} {recording_label}".casefold()
        live_markers = ("live", "concert", "shabahangi", "shabahang")
        return self.reject_live_voice_recordings and any(
            marker in searchable for marker in live_markers
        )

    def interval_is_excluded(self, song_id: str, start_seconds: float, end_seconds: float) -> bool:
        """Return whether a candidate lead excerpt intersects a reviewed bad range."""
        policy = self.voice_policy(song_id)
        return policy is not None and any(
            item.overlaps(start_seconds, end_seconds) for item in policy.excluded_intervals
        )


def load_dataset_policy(artist_root: Path, artist_id: str) -> ArtistDatasetPolicy:
    """Load an artist policy or return a restrictive, empty default for test profiles."""
    path = artist_root.resolve() / "data/dataset-policy.yaml"
    if not path.is_file():
        return ArtistDatasetPolicy(artist_id=artist_id)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        policy = ArtistDatasetPolicy.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise DatasetPolicyError(f"Invalid artist dataset policy: {error}") from error
    if policy.artist_id != artist_id:
        raise DatasetPolicyError("Dataset policy identity does not match the artist profile.")
    return policy
