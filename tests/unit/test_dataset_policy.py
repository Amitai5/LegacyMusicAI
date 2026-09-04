from datetime import UTC, datetime
from pathlib import Path

import pytest

from legacy_music.dataset_policy import (
    ArtistDatasetPolicy,
    DatasetPolicyError,
    ExcludedInterval,
    SongMusicPolicy,
    SongVoicePolicy,
    SourceExclusion,
)


def test_source_exclusion_denies_hash_and_exact_filename() -> None:
    policy = ArtistDatasetPolicy(
        artist_id="test-artist",
        source_exclusions=(
            SourceExclusion(
                sha256="a" * 64,
                filename="removed.mp3",
                reason="reviewed removal",
                excluded_at=datetime(2026, 8, 26, tzinfo=UTC),
            ),
        ),
    )

    with pytest.raises(DatasetPolicyError, match="permanently excluded"):
        policy.require_source_allowed("a" * 64, "renamed.mp3")
    with pytest.raises(DatasetPolicyError, match="permanently excluded"):
        policy.require_source_allowed("b" * 64, "REMOVED.MP3")


def test_voice_policy_rejects_live_songs_and_reviewed_intersections() -> None:
    song_policy = SongVoicePolicy(
        song_id="song-enferadi",
        excluded_intervals=(
            ExcludedInterval(
                start_seconds=20,
                end_seconds=32,
                reason="background vocal",
            ),
        ),
    )
    policy = ArtistDatasetPolicy(artist_id="test-artist", voice_songs=(song_policy,))

    assert policy.is_voice_song_excluded("song-live-in-concert") is True
    assert policy.interval_is_excluded("song-enferadi", 15, 25) is True
    assert policy.interval_is_excluded("song-enferadi", 32, 47) is False


def test_source_exclusion_rejects_path_bearing_filename() -> None:
    with pytest.raises(ValueError, match="filename"):
        SourceExclusion(
            sha256="a" * 64,
            filename=str(Path("nested") / "removed.mp3"),
            reason="reviewed removal",
            excluded_at=datetime(2026, 8, 26, tzinfo=UTC),
        )


def test_music_exclusion_does_not_exclude_voice_training() -> None:
    policy = ArtistDatasetPolicy(
        artist_id="test-artist",
        music_songs=(
            SongMusicPolicy(
                song_id="song-upbeat-a1b2c3d4",
                exclude_from_training=True,
                reason="Reviewed as upbeat music-style material.",
                excluded_at=datetime(2026, 8, 26, tzinfo=UTC),
            ),
        ),
    )

    assert policy.is_music_song_excluded("song-upbeat-a1b2c3d4") is True
    assert policy.is_voice_song_excluded("song-upbeat-a1b2c3d4") is False


def test_music_exclusion_requires_audit_metadata() -> None:
    with pytest.raises(ValueError, match="reason and timestamp"):
        SongMusicPolicy(
            song_id="song-upbeat-a1b2c3d4",
            exclude_from_training=True,
        )
