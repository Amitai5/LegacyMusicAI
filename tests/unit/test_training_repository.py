import json
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


def write_accompaniment_manifest(
    artist_root: Path,
    song: CatalogSong,
    accompaniment_bytes: bytes = b"synthetic-accompaniment-audio",
) -> Path:
    stem_root = artist_root / "data/derived/stems" / song.id
    stem_root.mkdir(parents=True)
    accompaniment = stem_root / "accompaniment.wav"
    accompaniment.write_bytes(accompaniment_bytes)
    manifest = stem_root / "stems.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artist_id": "test-artist",
                "song_id": song.id,
                "source_sha256": song.source_sha256,
                "normalized_sha256": song.normalized_sha256,
                "accompaniment": {
                    "path": accompaniment.relative_to(artist_root).as_posix(),
                    "sha256": sha256_file(accompaniment),
                },
            }
        ),
        encoding="utf-8",
    )
    return accompaniment


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
    accompaniment = write_accompaniment_manifest(artist_root, song)

    dataset = repository.prepare_dataset(
        ArtistCatalog(artist_id="test-artist", songs=(song,)),
        training_rights(source_hash),
        "teststyle",
        "original synthetic instrumental",
        True,
    )

    assert len(dataset.songs) == 1
    assert dataset.dataset_policy_sha256 is None
    assert dataset.songs[0].source_kind == "accompaniment"
    assert dataset.songs[0].audio_sha256 == sha256_file(accompaniment)
    assert dataset.songs[0].source_manifest is not None
    assert dataset.songs[0].source_manifest_sha256 is not None
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
    (export.parent / "training.json").write_text(
        json.dumps({"dataset_id": dataset.dataset_id}),
        encoding="utf-8",
    )
    selected = repository.select_adapter("training-test", export, "training-test")
    resolved, resolved_path = repository.resolve_selected_adapter()
    assert resolved.sha256 == selected.sha256
    assert resolved_path == adapter.resolve()

    weights.write_bytes(b"tampered")
    with pytest.raises(TrainingRepositoryError, match="integrity"):
        repository.resolve_selected_adapter()


def test_load_dataset_rejects_a_stale_dataset_policy(tmp_path: Path) -> None:
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
    write_accompaniment_manifest(artist_root, song)
    dataset = repository.prepare_dataset(
        ArtistCatalog(artist_id="test-artist", songs=(song,)),
        training_rights(source_hash),
        "teststyle",
        "original synthetic instrumental",
        True,
    )
    policy = artist_root / "data/dataset-policy.yaml"
    policy.write_text(
        "schema_version: 1\nartist_id: test-artist\nsource_exclusions: []\n",
        encoding="utf-8",
    )

    with pytest.raises(TrainingRepositoryError, match="stale dataset policy"):
        repository.load_dataset(dataset.dataset_id)


def test_prepare_instrumental_dataset_rejects_tampered_accompaniment(tmp_path: Path) -> None:
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
    accompaniment = write_accompaniment_manifest(artist_root, song)
    accompaniment.write_bytes(b"tampered")

    with pytest.raises(TrainingRepositoryError, match="integrity"):
        TrainingRepository(artist_root, "test-artist").prepare_dataset(
            ArtistCatalog(artist_id="test-artist", songs=(song,)),
            training_rights(source_hash),
            "teststyle",
            "original synthetic instrumental",
            True,
        )


def test_prepare_dataset_honors_music_only_exclusion(tmp_path: Path) -> None:
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
    write_accompaniment_manifest(artist_root, song)
    policy = artist_root / "data/dataset-policy.yaml"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        "schema_version: 2\n"
        "artist_id: test-artist\n"
        "music_songs:\n"
        "  - song_id: song-test-a1b2c3d4\n"
        "    exclude_from_training: true\n"
        "    reason: Reviewed as upbeat music-style material.\n"
        "    excluded_at: '2026-08-26T09:20:00Z'\n",
        encoding="utf-8",
    )

    with pytest.raises(TrainingRepositoryError, match="from music training"):
        TrainingRepository(artist_root, "test-artist").prepare_dataset(
            ArtistCatalog(artist_id="test-artist", songs=(song,)),
            training_rights(source_hash),
            "teststyle",
            "original synthetic instrumental",
            True,
        )
