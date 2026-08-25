"""Validated application configuration loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


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
