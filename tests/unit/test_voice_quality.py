from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from legacy_music.domain.voice import (
    ResolvedVoiceReference,
    VoiceReference,
    VoiceReferenceMetrics,
)
from legacy_music.voice_quality import (
    VoiceQualityError,
    analyze_reference_samples,
    clean_f0,
    plan_phrases,
    rank_references,
    select_reference,
)

RuntimePath = Path(__file__).parents[2] / "scripts/soulx_runtime.py"
RuntimeSpec = spec_from_file_location("soulx_runtime", RuntimePath)
assert RuntimeSpec is not None and RuntimeSpec.loader is not None
RuntimeModule = module_from_spec(RuntimeSpec)
RuntimeSpec.loader.exec_module(RuntimeModule)
candidate_seed = RuntimeModule._candidate_seed
join_phrases = RuntimeModule._join_phrases
max_join_jump_db = RuntimeModule._max_join_jump_db

SeedRuntimePath = Path(__file__).parents[2] / "scripts/seed_vc_runtime.py"
SeedRuntimeSpec = spec_from_file_location("seed_vc_runtime", SeedRuntimePath)
assert SeedRuntimeSpec is not None and SeedRuntimeSpec.loader is not None
SeedRuntimeModule = module_from_spec(SeedRuntimeSpec)
with patch.dict("sys.modules", {"torch": ModuleType("torch")}):
    SeedRuntimeSpec.loader.exec_module(SeedRuntimeModule)
retry_reference_ids = SeedRuntimeModule._retry_reference_ids


def reference(
    tmp_path: Path,
    reference_id: str,
    median_f0: float,
    active_level: float,
) -> ResolvedVoiceReference:
    digest = "a" * 64
    record = VoiceReference(
        id=reference_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_file=Path(f"voice/{reference_id}/source.wav"),
        source_sha256=digest,
        audio_file=Path(f"voice/{reference_id}/audio.wav"),
        audio_sha256=digest,
        f0_file=Path(f"voice/{reference_id}/f0.npy"),
        f0_sha256=digest,
        metrics=VoiceReferenceMetrics(
            duration_seconds=15,
            voiced_fraction=0.8,
            median_f0_hz=median_f0,
            f0_p05_hz=median_f0 * 0.8,
            f0_p95_hz=median_f0 * 1.2,
            active_rms_dbfs=active_level,
            one_second_level_range_db=2,
            peak_dbfs=-3,
            clipping_fraction=0,
            high_band_ratio_db=-20,
        ),
    )
    return ResolvedVoiceReference(
        reference=record,
        source=tmp_path / f"{reference_id}-source.wav",
        audio=tmp_path / f"{reference_id}-audio.wav",
        f0=tmp_path / f"{reference_id}-f0.npy",
    )


def test_clean_f0_repairs_short_gap_and_isolated_octave_error() -> None:
    values = np.array([200, 200, 0, 0, 202, 404, 202], dtype=np.float32)

    cleaned = clean_f0(values, maximum_gap_frames=2)

    assert np.all(cleaned[2:4] > 0)
    assert abs(float(cleaned[5]) - 202) < 1


def test_clean_f0_preserves_long_rest() -> None:
    values = np.array([200, 200, 0, 0, 0, 200], dtype=np.float32)

    cleaned = clean_f0(values, maximum_gap_frames=2)

    assert np.array_equal(cleaned[2:5], np.zeros(3, dtype=np.float32))


def test_reference_analysis_rejects_scattered_f0_without_sustained_voicing() -> None:
    audio = np.full(48000 * 2, 0.1, dtype=np.float32)
    f0 = np.zeros(100, dtype=np.float32)
    f0[::10] = 200

    with pytest.raises(VoiceQualityError, match="sustained"):
        analyze_reference_samples(audio, 48000, f0)


def test_plan_phrases_has_overlap_and_complete_coverage() -> None:
    phrases = plan_phrases(np.full(50 * 48, 220, dtype=np.float32), 12, 20, 1)

    assert phrases[0].start_seconds == 0
    assert phrases[-1].end_seconds == 48
    assert all(12 <= phrase.end_seconds - phrase.start_seconds <= 20 for phrase in phrases)
    assert all(
        current.start_seconds < previous.end_seconds
        for previous, current in zip(phrases, phrases[1:], strict=False)
    )


def test_select_reference_uses_register_and_energy(tmp_path: Path) -> None:
    target_audio = tmp_path / "target.wav"
    samples = np.arange(24000 * 12) / 24000
    sf.write(
        target_audio,
        0.1 * np.sin(2 * np.pi * 330 * samples),
        24000,
        subtype="PCM_16",
    )
    phrase = plan_phrases(np.full(50 * 12, 330, dtype=np.float32), 12, 20, 1)[0]
    references = (
        reference(tmp_path, "low", 140, -30),
        reference(tmp_path, "high", 325, -23),
    )

    selected = select_reference(
        references,
        target_audio,
        np.full(50 * 12, 330, dtype=np.float32),
        phrase,
    )

    assert selected.reference.id == "high"


def test_rank_references_preserves_register_order(tmp_path: Path) -> None:
    target_audio = tmp_path / "target.wav"
    samples = np.arange(24000 * 12) / 24000
    sf.write(target_audio, 0.1 * np.sin(2 * np.pi * 330 * samples), 24000)
    phrase = plan_phrases(np.full(50 * 12, 330, dtype=np.float32), 12, 20, 1)[0]
    references = (
        reference(tmp_path, "low", 140, -30),
        reference(tmp_path, "mid", 220, -24),
        reference(tmp_path, "high", 325, -23),
    )

    ranked = rank_references(
        references,
        target_audio,
        np.full(50 * 12, 330, dtype=np.float32),
        phrase,
    )

    assert [item.reference.id for item in ranked] == ["high", "mid", "low"]


def test_candidate_seeds_are_deterministic_and_distinct() -> None:
    first = [candidate_seed(42, phrase, candidate) for phrase in range(3) for candidate in range(2)]
    second = [
        candidate_seed(42, phrase, candidate) for phrase in range(3) for candidate in range(2)
    ]

    assert first == second
    assert len(first) == len(set(first))


def test_retry_references_preserve_continuity_and_identity_diversity() -> None:
    records = [
        {
            "candidate_index": 0,
            "reference_id": "soft",
            "minimum_window_identity_similarity": 0.40,
            "mean_window_identity_similarity": 0.60,
            "sustained_dropout_count": 0,
            "f0_gross_error_fraction": 0.02,
        },
        {
            "candidate_index": 1,
            "reference_id": "low",
            "minimum_window_identity_similarity": 0.61,
            "mean_window_identity_similarity": 0.69,
            "sustained_dropout_count": 1,
            "f0_gross_error_fraction": 0.02,
        },
        {
            "candidate_index": 2,
            "reference_id": "neutral",
            "minimum_window_identity_similarity": 0.55,
            "mean_window_identity_similarity": 0.66,
            "sustained_dropout_count": 1,
            "f0_gross_error_fraction": 0.03,
        },
    ]

    ranked = retry_reference_ids(records, ["soft", "low", "neutral", "high"])

    assert ranked[:3] == ["soft", "low", "neutral"]
    assert ranked == ["soft", "low", "neutral", "high"]


def test_join_phrases_uses_equal_power_crossfade_without_dropped_samples() -> None:
    first = np.ones(8, dtype=np.float32)
    second = np.full(8, 0.5, dtype=np.float32)

    joined, join_gains = join_phrases([(0, 8, first), (6, 14, second)], 14)

    expected_phase = np.linspace(0, np.pi / 2, 2, endpoint=True)
    expected_overlap = np.cos(expected_phase) + 0.5 * (10 ** (6 / 20)) * np.sin(expected_phase)
    assert np.allclose(joined[6:8], expected_overlap)
    assert join_gains == [0, pytest.approx(6)]
    assert len(joined) == 14
    assert np.all(joined != 0)


def test_join_jump_ignores_real_rest_but_checks_continuous_voicing() -> None:
    audio = np.concatenate(
        (np.full(480, 0.5, dtype=np.float32), np.full(480, 0.01, dtype=np.float32))
    )
    rest_f0 = np.array([220, 0, 0], dtype=np.float32)
    voiced_f0 = np.array([220, 220, 220], dtype=np.float32)

    ignored = max_join_jump_db(audio, [480], rest_f0, 24000, 50)
    measured = max_join_jump_db(audio, [480], voiced_f0, 24000, 50)

    assert ignored == 0
    assert measured > 6
