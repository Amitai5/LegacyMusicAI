"""Voice-reference, quality, and conversion contracts."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VoiceModel(BaseModel):
    """Base model for immutable voice-domain contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VoiceReviewStatus(StrEnum):
    """Human or machine review state for one curated prompt."""

    APPROVED = "approved"
    REJECTED = "rejected"


class VoiceReferenceMetrics(VoiceModel):
    """Stable acoustic measurements used for prompt matching."""

    duration_seconds: float = Field(gt=0, le=60)
    voiced_fraction: float = Field(ge=0, le=1)
    median_f0_hz: float = Field(gt=0)
    f0_p05_hz: float = Field(gt=0)
    f0_p95_hz: float = Field(gt=0)
    active_rms_dbfs: float = Field(ge=-120, le=0)
    one_second_level_range_db: float = Field(ge=0, le=120)
    peak_dbfs: float = Field(ge=-120, le=1)
    clipping_fraction: float = Field(ge=0, le=1)
    high_band_ratio_db: float = Field(ge=-120, le=60)


class VoiceReference(VoiceModel):
    """Curated artist-specific vocal reference and immutable lineage metadata."""

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    created_at: datetime
    source_file: Path
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    audio_file: Path
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    f0_file: Path
    f0_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    tags: tuple[str, ...] = ()
    parent_source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_song_id: str | None = None
    source_stem_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    source_start_seconds: float | None = Field(default=None, ge=0)
    source_end_seconds: float | None = Field(default=None, gt=0)
    metrics: VoiceReferenceMetrics | None = None
    review_status: VoiceReviewStatus = VoiceReviewStatus.APPROVED


class ResolvedVoiceReference(VoiceModel):
    """Integrity-checked voice reference with artist-confined absolute paths."""

    reference: VoiceReference
    source: Path
    audio: Path
    f0: Path


class VoiceConversionRequest(VoiceModel):
    """Validated multi-reference singing-voice conversion request."""

    target_audio: Path
    target_f0: Path
    references: tuple[ResolvedVoiceReference, ...] = Field(min_length=1)
    engine: str = "soulx"
    seed: int = Field(default=0, ge=0, le=2**63 - 1)


class VoiceCandidateMetrics(VoiceModel):
    """Objective measurements for one generated phrase candidate."""

    candidate_index: int = Field(ge=0)
    reference_id: str | None = None
    seed: int = Field(ge=0)
    passed: bool
    raw_peak_dbfs: float
    peak_reduction_db: float = Field(ge=0)
    sustained_dropout_count: int = Field(ge=0)
    repaired_dropout_count: int = Field(default=0, ge=0)
    dropout_fraction: float = Field(ge=0, le=1)
    spectral_distance_db: float = Field(ge=0)
    identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    reference_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    identity_threshold: float | None = Field(default=None, ge=-1, le=1)
    minimum_window_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    mean_window_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    identity_window_threshold: float | None = Field(default=None, ge=-1, le=1)
    identity_window_count: int | None = Field(default=None, ge=1)
    f0_median_error_cents: float | None = Field(default=None, ge=0)
    f0_gross_error_fraction: float | None = Field(default=None, ge=0, le=1)


class VoicePhraseDecision(VoiceModel):
    """Selected reference and candidate for one overlapping target phrase."""

    index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    reference_id: str
    selected_candidate_index: int = Field(ge=0)
    join_gain_db: float = Field(default=0, ge=-6, le=6)
    candidates: tuple[VoiceCandidateMetrics, ...] = Field(min_length=1)


class VoiceConversionMetrics(VoiceModel):
    """Aggregate objective quality measurements for one converted stem."""

    phrase_count: int = Field(ge=1)
    selected_reference_ids: tuple[str, ...] = Field(min_length=1)
    sustained_dropout_count: int = Field(ge=0)
    peak_dbfs: float
    clipping_fraction: float = Field(ge=0, le=1)
    max_voiced_join_jump_db: float = Field(default=0, ge=0)
    minimum_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    mean_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    identity_threshold: float | None = Field(default=None, ge=-1, le=1)
    minimum_window_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    mean_window_identity_similarity: float | None = Field(default=None, ge=-1, le=1)
    identity_window_threshold: float | None = Field(default=None, ge=-1, le=1)
    f0_median_error_cents: float | None = Field(default=None, ge=0)
    f0_gross_error_fraction: float | None = Field(default=None, ge=0, le=1)


class VoiceConversionManifest(VoiceModel):
    """Sanitized reproducibility and QC record for one converted vocal stem."""

    schema_version: int = Field(default=4, ge=2)
    engine: str
    mode: str
    created_at: datetime
    target_audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    target_f0_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cleaned_f0_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_audio_hashes: dict[str, str]
    reference_f0_hashes: dict[str, str] = Field(default_factory=dict)
    reference_source_hashes: dict[str, str] = Field(default_factory=dict)
    reference_parent_source_hashes: dict[str, str] = Field(default_factory=dict)
    reference_stem_hashes: dict[str, str] = Field(default_factory=dict)
    settings: dict[str, Any]
    phrases: tuple[VoicePhraseDecision, ...]
    metrics: VoiceConversionMetrics
    output_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class VoiceConversionResult(VoiceModel):
    """Typed artifact returned by a singing-voice engine."""

    audio: Path
    manifest: Path
    cleaned_f0: Path
    reference_ids: tuple[str, ...] = Field(min_length=1)
    phrases: tuple[VoicePhraseDecision, ...] = Field(min_length=1)
    metrics: VoiceConversionMetrics
