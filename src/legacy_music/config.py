"""Validated application configuration loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ConfigError(RuntimeError):
    """Raised when an application configuration cannot be loaded safely."""


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictConfigModel):
    """Project identity configuration."""

    name: str = "legacy-music-ai"


class PathConfig(StrictConfigModel):
    """Repository-relative runtime paths."""

    artists: Path = Path("artists")
    shared_models: Path = Path("models/shared")
    model_cache: Path = Path("models/cache")
    runs: Path = Path("runs")
    vendor: Path = Path("vendor")


class GenerationConfig(StrictConfigModel):
    """Default generation behavior."""

    default_music_engine: str = "ace-step"
    default_voice_engine: str = "soulx"
    default_duration_seconds: int = Field(default=180, ge=10, le=600)
    default_variants: int = Field(default=1, ge=1, le=16)
    keep_intermediates: bool = True


class AudioOutputConfig(StrictConfigModel):
    """Default output formats."""

    preview_format: Literal["mp3", "ogg", "opus"] = "mp3"
    final_format: Literal["wav", "flac"] = "wav"
    ffmpeg_executable: Path | None = None


class VoiceQualityConfig(StrictConfigModel):
    """Quality-first singing-voice conversion and mastering defaults."""

    phrase_min_seconds: float = Field(default=12.0, ge=5.0, le=30.0)
    phrase_max_seconds: float = Field(default=20.0, ge=5.0, le=30.0)
    phrase_overlap_seconds: float = Field(default=1.0, ge=0.1, le=5.0)
    candidates_per_phrase: int = Field(default=2, ge=1, le=4)
    identity_candidates_per_phrase: int = Field(default=5, ge=2, le=5)
    identity_max_candidates_per_phrase: int = Field(default=15, ge=5, le=20)
    identity_phrase_min_seconds: float = Field(default=6.0, ge=5.0, le=30.0)
    identity_phrase_max_seconds: float = Field(default=8.0, ge=5.0, le=30.0)
    identity_phrase_overlap_seconds: float = Field(default=1.0, ge=0.1, le=5.0)
    identity_similarity_margin: float = Field(default=0.08, ge=0.0, le=0.5)
    identity_minimum_similarity: float = Field(default=0.2, ge=-1.0, le=1.0)
    identity_window_seconds: float = Field(default=4.0, ge=2.0, le=8.0)
    identity_window_hop_seconds: float = Field(default=2.0, ge=0.5, le=8.0)
    identity_window_min_voiced_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    identity_window_similarity_margin: float = Field(default=0.04, ge=0.0, le=0.5)
    identity_inference_cfg_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    f0_gap_fill_ms: int = Field(default=120, ge=0, le=500)
    f0_median_error_limit_cents: float = Field(default=50.0, ge=0.0, le=1200.0)
    f0_gross_error_limit_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    audio_dropout_repair_ms: int = Field(default=120, ge=0, le=200)
    audio_dropout_max_gain_db: float = Field(default=12.0, ge=0.0, le=24.0)
    local_gain_limit_db: float = Field(default=6.0, ge=0.0, le=12.0)
    compressor_threshold_dbfs: float = Field(default=-18.0, ge=-60.0, le=0.0)
    compressor_ratio: float = Field(default=2.0, ge=1.0, le=20.0)
    compressor_attack_ms: float = Field(default=15.0, ge=0.1, le=200.0)
    compressor_release_ms: float = Field(default=120.0, ge=10.0, le=2000.0)
    target_lufs: float = Field(default=-14.0, ge=-30.0, le=-5.0)
    target_lra: float = Field(default=9.0, ge=1.0, le=20.0)
    true_peak_dbfs: float = Field(default=-1.0, ge=-6.0, le=-0.1)

    @field_validator("phrase_max_seconds")
    @classmethod
    def validate_phrase_maximum(cls, value: float, info: Any) -> float:
        """Require the maximum phrase length to exceed the configured minimum."""
        minimum = info.data.get("phrase_min_seconds")
        if minimum is not None and value < minimum:
            raise ValueError("phrase_max_seconds must be at least phrase_min_seconds.")
        return value

    @field_validator("phrase_overlap_seconds")
    @classmethod
    def validate_phrase_overlap(cls, value: float, info: Any) -> float:
        """Keep overlap shorter than the configured minimum phrase length."""
        minimum = info.data.get("phrase_min_seconds")
        if minimum is not None and value >= minimum:
            raise ValueError("phrase_overlap_seconds must be shorter than a phrase.")
        return value

    @field_validator("identity_phrase_max_seconds")
    @classmethod
    def validate_identity_phrase_maximum(cls, value: float, info: Any) -> float:
        """Require identity phrase maximum length to exceed its minimum."""
        minimum = info.data.get("identity_phrase_min_seconds")
        if minimum is not None and value < minimum:
            raise ValueError(
                "identity_phrase_max_seconds must be at least identity_phrase_min_seconds."
            )
        return value

    @field_validator("identity_phrase_overlap_seconds")
    @classmethod
    def validate_identity_phrase_overlap(cls, value: float, info: Any) -> float:
        """Keep identity overlap shorter than its minimum phrase length."""
        minimum = info.data.get("identity_phrase_min_seconds")
        if minimum is not None and value >= minimum:
            raise ValueError(
                "identity_phrase_overlap_seconds must be shorter than an identity phrase."
            )
        return value

    @model_validator(mode="after")
    def validate_identity_windows(self) -> VoiceQualityConfig:
        """Keep local identity windows ordered and inside the shortest phrase."""
        if self.identity_max_candidates_per_phrase < self.identity_candidates_per_phrase:
            raise ValueError(
                "identity_max_candidates_per_phrase must be at least the initial candidate count."
            )
        if self.identity_window_hop_seconds > self.identity_window_seconds:
            raise ValueError("identity_window_hop_seconds must not exceed identity_window_seconds.")
        if self.identity_window_seconds > self.identity_phrase_min_seconds:
            raise ValueError("identity_window_seconds must not exceed identity_phrase_min_seconds.")
        return self


class TrainingVocalConfig(StrictConfigModel):
    """Independent cleanup and lead-only selection settings for voice training."""

    dereverb_strength: float = Field(default=0.9, ge=0.0, le=1.0)
    lead_min_voiced_fraction: float = Field(default=0.72, ge=0.0, le=1.0)
    lead_max_voiced_fraction: float = Field(default=0.98, ge=0.0, le=1.0)
    lead_max_level_range_db: float = Field(default=8.0, ge=0.0, le=30.0)
    lead_max_relative_level_deficit_db: float = Field(default=6.0, ge=0.0, le=30.0)
    lead_min_continuous_voicing_seconds: float = Field(default=1.2, ge=0.1, le=10.0)
    lead_min_high_band_ratio_db: float = Field(default=-35.0, ge=-120.0, le=0.0)
    lead_min_center_dominance_db: float = Field(default=3.0, ge=-20.0, le=60.0)

    @model_validator(mode="after")
    def validate_voiced_range(self) -> TrainingVocalConfig:
        """Require the maximum voiced fraction to exceed the minimum."""
        if self.lead_max_voiced_fraction < self.lead_min_voiced_fraction:
            raise ValueError("lead_max_voiced_fraction must be at least lead_min_voiced_fraction.")
        return self


class FinalVocalMixConfig(StrictConfigModel):
    """Independent vocal-bus treatment used only for final generated mixes."""

    vocal_presence_gain_db: float = Field(default=1.5, ge=-6.0, le=6.0)
    instrumental_gain_db: float = Field(default=-0.75, ge=-6.0, le=3.0)
    presence_eq_frequency_hz: float = Field(default=2800.0, ge=500.0, le=8000.0)
    presence_eq_gain_db: float = Field(default=1.5, ge=-6.0, le=6.0)
    reverb_wet: float = Field(default=0.08, ge=0.0, le=0.3)
    reverb_pre_delay_ms: float = Field(default=24.0, ge=0.0, le=100.0)
    reverb_decay: float = Field(default=0.22, ge=0.0, le=0.8)
    automatic_vocal_balance: bool = True
    target_vocal_to_instrumental_db: float = Field(default=1.5, ge=-10.0, le=20.0)
    maximum_automatic_balance_correction_db: float = Field(default=12.0, ge=0.0, le=24.0)
    minimum_vocal_to_instrumental_db: float = Field(default=-4.0, ge=-20.0, le=10.0)
    maximum_vocal_to_instrumental_db: float = Field(default=6.0, ge=-10.0, le=20.0)

    @model_validator(mode="after")
    def validate_vocal_balance_range(self) -> FinalVocalMixConfig:
        """Require the maximum accepted vocal balance to exceed the minimum."""
        if self.maximum_vocal_to_instrumental_db < self.minimum_vocal_to_instrumental_db:
            raise ValueError(
                "maximum_vocal_to_instrumental_db must be at least the configured minimum."
            )
        if not (
            self.minimum_vocal_to_instrumental_db
            <= self.target_vocal_to_instrumental_db
            <= self.maximum_vocal_to_instrumental_db
        ):
            raise ValueError(
                "target_vocal_to_instrumental_db must be inside the accepted balance range."
            )
        return self


class ProvenanceConfig(StrictConfigModel):
    """Synthetic-output provenance requirements."""

    enabled: bool = True
    require_disclosure: bool = True


class SafetyConfig(StrictConfigModel):
    """Application-wide authorization gates."""

    require_approved_rights: bool = True
    require_voice_authorization: bool = True
    allow_automatic_distribution: bool = False


class AppConfig(StrictConfigModel):
    """Complete application configuration."""

    project: ProjectConfig = ProjectConfig()
    paths: PathConfig = PathConfig()
    generation: GenerationConfig = GenerationConfig()
    audio: AudioOutputConfig = AudioOutputConfig()
    voice_quality: VoiceQualityConfig = VoiceQualityConfig()
    training_vocals: TrainingVocalConfig = TrainingVocalConfig()
    final_vocal_mix: FinalVocalMixConfig = FinalVocalMixConfig()
    provenance: ProvenanceConfig = ProvenanceConfig()
    safety: SafetyConfig = SafetyConfig()


def load_app_config(config_path: Path, local_override_path: Path | None = None) -> AppConfig:
    """Load the base YAML configuration and an optional local override."""
    base = _read_yaml_mapping(config_path)
    merged = base

    if local_override_path is not None and local_override_path.exists():
        merged = _deep_merge(base, _read_yaml_mapping(local_override_path))

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as error:
        raise ConfigError(f"Invalid application configuration: {error}") from error


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"Unable to read configuration '{path}': {error}") from error

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration '{path}': {error}") from error

    if not isinstance(value, dict):
        raise ConfigError(f"Configuration '{path}' must contain a YAML mapping.")

    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged
