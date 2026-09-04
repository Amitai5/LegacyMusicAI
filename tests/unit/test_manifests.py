from datetime import UTC, datetime
from pathlib import Path

import pytest

from legacy_music.domain.rights import RightsManifest, RightsPermissions, RightsStatus
from legacy_music.domain.voice import VoiceReference
from legacy_music.manifests import load_artist_profile, load_rights_manifest

ProjectRoot = Path(__file__).resolve().parents[2]


def test_committed_artist_template_is_valid_and_not_voice_enabled() -> None:
    profile = load_artist_profile(ProjectRoot / "artists/_template/artist.yaml")

    assert profile.id == "example-artist"
    assert profile.capabilities.music_style is True
    assert profile.capabilities.singing_voice is False


def test_committed_rights_template_is_pending_and_denies_training() -> None:
    rights = load_rights_manifest(ProjectRoot / "artists/_template/rights.yaml")

    assert rights.status is RightsStatus.PENDING
    assert rights.permits("training", datetime(2026, 8, 24, tzinfo=UTC)) is False


def test_approved_unexpired_rights_permit_only_explicit_capabilities() -> None:
    rights = RightsManifest(
        artist_id="example-artist",
        status=RightsStatus.APPROVED,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
        permissions=RightsPermissions(training=True, music_style=True),
    )
    instant = datetime(2026, 8, 24, tzinfo=UTC)

    assert rights.permits("training", instant) is True
    assert rights.permits("singing_voice", instant) is False


def test_rights_manifest_with_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RightsManifest(
            artist_id="example-artist",
            status=RightsStatus.APPROVED,
            effective_at=datetime(2026, 1, 1),
        )


def test_rights_manifest_with_inverted_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="expiration"):
        RightsManifest(
            artist_id="example-artist",
            status=RightsStatus.APPROVED,
            effective_at=datetime(2027, 1, 1, tzinfo=UTC),
            expires_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_rights_manifest_with_unknown_capability_is_rejected() -> None:
    rights = RightsManifest(artist_id="example-artist")

    with pytest.raises(ValueError, match="Unknown rights capability"):
        rights.permits("impersonation", datetime(2026, 8, 24, tzinfo=UTC))


def test_legacy_voice_reference_manifest_remains_loadable() -> None:
    digest = "a" * 64
    reference = VoiceReference.model_validate(
        {
            "id": "neutral",
            "created_at": "2026-01-01T00:00:00Z",
            "source_file": "voice/references/neutral/source.wav",
            "source_sha256": digest,
            "audio_file": "voice/references/neutral/audio.wav",
            "audio_sha256": digest,
            "f0_file": "voice/references/neutral/f0.npy",
            "f0_sha256": digest,
            "tags": ["neutral"],
        }
    )

    assert reference.metrics is None
    assert reference.parent_source_sha256 is None
    assert reference.review_status.value == "approved"
