from datetime import UTC, datetime
from pathlib import Path

import pytest

from legacy_music.domain.catalog import ArtistCatalog, AudioProperties, CatalogSong
from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
    SourceAssetAuthorization,
)
from legacy_music.repositories.training import TrainingRepository, TrainingRepositoryError
from legacy_music.utils.hashing import sha256_file


def training_rights(source_hash: str) -> RightsManifest:
    approved_at = datetime(2026, 1, 1, tzinfo=UTC)
    return RightsManifest(
        artist_id="test-artist",
        status=RightsStatus.APPROVED,
        permissions=RightsPermissions(training=True),
        approvals=(
            ApprovalRecord(
                holder="Synthetic fixture owner",
                scope="Synthetic training fixture",
                evidence_reference="test://approval",
                approved_at=approved_at,
            ),
        ),
        source_assets=(
            SourceAssetAuthorization(
                sha256=source_hash,
                evidence_reference="test://source",
            ),
        ),
    )


def test_prepare_dataset_and_select_adapter_preserve_artist_integrity(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/test-artist"
    normalized = artist_root / "data/derived/normalized/song-test.wav"
    normalized.parent.mkdir(parents=True)
    normalized.write_bytes(b"synthetic-normalized-audio")
    source_hash = "a" * 64
    song = CatalogSong(
        id="song-test-a1b2c3d4",
        title="Test",
        source_filename="test.wav",
        source_sha256=source_hash,
        original_path="data/raw/originals/test.wav",
        original_sha256=source_hash,
        audio=AudioProperties(
            duration_seconds=10,
            sample_rate=48000,
            channels=2,
            codec="pcm_s24le",
            format_name="wav",
        ),
        imported_at=datetime(2026, 1, 1, tzinfo=UTC),
        normalized_path="data/derived/normalized/song-test.wav",
        normalized_sha256=sha256_file(normalized),
    )
    repository = TrainingRepository(artist_root, "test-artist")

    dataset = repository.prepare_dataset(
        ArtistCatalog(artist_id="test-artist", songs=(song,)),
        training_rights(source_hash),
        "teststyle",
        "original synthetic instrumental",
        True,
    )

    assert len(dataset.songs) == 1
    caption = artist_root / dataset.songs[0].caption_file
    assert caption.read_text(encoding="utf-8").strip() == (
        "teststyle, original synthetic instrumental"
    )

    export = artist_root / "models/music/training/training-test/export"
    adapter = export / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"synthetic-adapter")
    selected = repository.select_adapter("training-test", export, "training-test")
    resolved, resolved_path = repository.resolve_selected_adapter()
    assert resolved.sha256 == selected.sha256
    assert resolved_path == adapter.resolve()

    weights.write_bytes(b"tampered")
    with pytest.raises(TrainingRepositoryError, match="integrity"):
        repository.resolve_selected_adapter()
