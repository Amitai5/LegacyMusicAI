from pathlib import Path

import numpy as np
import pytest

from legacy_music.domain.voice import VoiceCandidateMetrics
from legacy_music.engines.seed_vc import (
    SeedVCError,
    _confined_runtime_path,
    _count_sustained_dropouts,
    _fit_length,
    _join_phrases,
    _repair_short_dropouts,
    _select_identity_candidate,
)


def candidate(
    index: int,
    similarity: float,
    passed: bool,
    gross_pitch_error: float = 0.01,
    minimum_window_similarity: float | None = None,
) -> VoiceCandidateMetrics:
    window_similarity = (
        similarity if minimum_window_similarity is None else minimum_window_similarity
    )
    return VoiceCandidateMetrics(
        candidate_index=index,
        reference_id="neutral",
        seed=100 + index,
        passed=passed,
        raw_peak_dbfs=-3,
        peak_reduction_db=0,
        sustained_dropout_count=0,
        dropout_fraction=0,
        spectral_distance_db=1,
        identity_similarity=similarity,
        reference_identity_similarity=similarity,
        identity_threshold=0.4,
        minimum_window_identity_similarity=window_similarity,
        mean_window_identity_similarity=window_similarity,
        identity_window_threshold=0.4,
        identity_window_count=3,
        f0_median_error_cents=10,
        f0_gross_error_fraction=gross_pitch_error,
    )


def test_select_identity_candidate_prefers_passing_highest_similarity() -> None:
    candidates = (
        candidate(0, 0.8, False),
        candidate(1, 0.6, True),
        candidate(2, 0.7, True),
    )

    selected = _select_identity_candidate(candidates)

    assert selected == 2


def test_select_identity_candidate_rejects_higher_identity_with_failed_pitch() -> None:
    candidates = (
        candidate(0, 0.9, False, gross_pitch_error=0.08),
        candidate(1, 0.7, True, gross_pitch_error=0.02),
    )

    selected = _select_identity_candidate(candidates)

    assert selected == 1


def test_select_identity_candidate_prefers_stronger_weakest_window() -> None:
    candidates = (
        candidate(0, 0.9, True, minimum_window_similarity=0.45),
        candidate(1, 0.75, True, minimum_window_similarity=0.7),
    )

    selected = _select_identity_candidate(candidates)

    assert selected == 1


def test_join_identity_phrases_crossfades_complete_coverage() -> None:
    first = np.ones(8, dtype=np.float32)
    second = np.full(8, 0.5, dtype=np.float32)

    joined, gains = _join_phrases([(0, 8, first), (6, 14, second)], 14)

    assert len(joined) == 14
    assert gains == [0, pytest.approx(6)]
    assert np.all(joined != 0)


def test_join_identity_phrases_rejects_uncovered_gap() -> None:
    with pytest.raises(SeedVCError, match="gap"):
        _join_phrases(
            [
                (0, 4, np.ones(4, dtype=np.float32)),
                (5, 8, np.ones(3, dtype=np.float32)),
            ],
            8,
        )


def test_fit_length_preserves_exact_requested_duration() -> None:
    assert np.array_equal(
        _fit_length(np.array([1, 2], dtype=np.float32), 4),
        np.array([1, 2, 0, 0], dtype=np.float32),
    )
    assert np.array_equal(
        _fit_length(np.array([1, 2, 3], dtype=np.float32), 2),
        np.array([1, 2], dtype=np.float32),
    )


def test_runtime_candidate_path_cannot_escape_identity_directory(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")

    with pytest.raises(SeedVCError, match="escaped"):
        _confined_runtime_path(runtime, "../outside.wav")


def test_short_audio_dropout_is_repaired_without_changing_duration() -> None:
    sample_rate = 1000
    audio = np.full(sample_rate, 0.1, dtype=np.float32)
    audio[400:520] = 0.001
    f0 = np.full(50, 220, dtype=np.float32)

    before, _ = _count_sustained_dropouts(audio, sample_rate, f0)
    repaired, repaired_count = _repair_short_dropouts(audio, sample_rate, f0, 120, 12)
    after, _ = _count_sustained_dropouts(repaired, sample_rate, f0)

    assert before == 1
    assert repaired_count == 1
    assert after == 0
    assert len(repaired) == len(audio)


def test_long_audio_dropout_remains_a_qc_failure() -> None:
    sample_rate = 1000
    audio = np.full(sample_rate, 0.1, dtype=np.float32)
    audio[400:540] = 0.001
    f0 = np.full(50, 220, dtype=np.float32)

    repaired, repaired_count = _repair_short_dropouts(audio, sample_rate, f0, 120, 12)
    after, _ = _count_sustained_dropouts(repaired, sample_rate, f0)

    assert repaired_count == 0
    assert after == 1
