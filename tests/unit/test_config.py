from pathlib import Path

import pytest
from pydantic import ValidationError

from legacy_music.config import ConfigError, FinalVocalMixConfig, load_app_config


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
    assert config.voice_quality.phrase_max_seconds == 20
    assert config.voice_quality.candidates_per_phrase == 2
    assert config.voice_quality.identity_candidates_per_phrase == 5
    assert config.voice_quality.identity_max_candidates_per_phrase == 15
    assert config.voice_quality.identity_phrase_min_seconds == 6
    assert config.voice_quality.identity_phrase_max_seconds == 8
    assert config.voice_quality.identity_phrase_overlap_seconds == 1
    assert config.voice_quality.identity_similarity_margin == 0.08
    assert config.voice_quality.identity_minimum_similarity == 0.2
    assert config.voice_quality.identity_window_seconds == 4
    assert config.voice_quality.identity_window_hop_seconds == 2
    assert config.voice_quality.identity_window_min_voiced_fraction == 0.25
    assert config.voice_quality.identity_window_similarity_margin == 0.04
    assert config.voice_quality.identity_inference_cfg_rate == 1
    assert config.voice_quality.f0_gap_fill_ms == 120
    assert config.voice_quality.f0_median_error_limit_cents == 50
    assert config.voice_quality.f0_gross_error_limit_fraction == 0.05
    assert config.voice_quality.audio_dropout_repair_ms == 120
    assert config.voice_quality.audio_dropout_max_gain_db == 12
    assert config.voice_quality.target_lufs == -14
    assert config.voice_quality.true_peak_dbfs == -1
    assert config.training_vocals.dereverb_strength == 0.9
    assert config.training_vocals.lead_min_voiced_fraction == 0.72
    assert config.final_vocal_mix.vocal_presence_gain_db == 1.5
    assert config.final_vocal_mix.automatic_vocal_balance is True
    assert config.final_vocal_mix.target_vocal_to_instrumental_db == 1.5
    assert config.final_vocal_mix.maximum_automatic_balance_correction_db == 12
    assert config.final_vocal_mix.reverb_wet == 0.08
    assert config.safety.allow_automatic_distribution is False


def test_load_app_config_with_unknown_key_rejects_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid application configuration"):
        load_app_config(config_path)


def test_final_vocal_mix_rejects_target_outside_acceptance_range() -> None:
    with pytest.raises(ValidationError, match="target_vocal_to_instrumental_db"):
        FinalVocalMixConfig(
            minimum_vocal_to_instrumental_db=-4,
            maximum_vocal_to_instrumental_db=6,
            target_vocal_to_instrumental_db=7,
        )
