from pathlib import Path

import pytest

from legacy_music.config import ConfigError, load_app_config


def test_load_app_config_with_local_override_merges_nested_values(tmp_path: Path) -> None:
    base_path = tmp_path / "app.yaml"
    base_path.write_text(
        """
project:
  name: legacy-music-ai
paths:
  artists: artists
  shared_models: models/shared
  model_cache: models/cache
  runs: runs
  vendor: vendor
generation:
  default_music_engine: ace-step
  default_voice_engine: soulx
  default_duration_seconds: 180
  default_variants: 1
  keep_intermediates: true
audio:
  preview_format: mp3
  final_format: wav
provenance:
  enabled: true
  require_disclosure: true
safety:
  require_approved_rights: true
  require_voice_authorization: true
  allow_automatic_distribution: false
""".strip(),
        encoding="utf-8",
    )
    override_path = tmp_path / "local.yaml"
    override_path.write_text(
        "paths:\n  runs: /mnt/legacy-music/runs\ngeneration:\n  default_variants: 4\n",
        encoding="utf-8",
    )

    config = load_app_config(base_path, override_path)

    assert config.paths.artists == Path("artists")
    assert config.paths.runs == Path("/mnt/legacy-music/runs")
    assert config.generation.default_variants == 4
    assert config.safety.allow_automatic_distribution is False


def test_load_app_config_with_unknown_key_rejects_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid application configuration"):
        load_app_config(config_path)
