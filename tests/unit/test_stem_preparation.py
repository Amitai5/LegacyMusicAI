from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from legacy_music.domain.catalog import AudioProperties, CatalogSong
from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
    SourceAssetAuthorization,
)
from legacy_music.engines.base import StemResult
from legacy_music.pipeline.stem_preparation import CatalogStemPipeline
from legacy_music.repositories.catalog import CatalogRepository
from legacy_music.utils.hashing import sha256_file


class FakeStemSeparator:
    def __init__(self) -> None:
        self.calls = 0

    def separate(self, audio: Path, output_dir: Path) -> StemResult:
        self.calls += 1
        samples, sample_rate = sf.read(audio, dtype="float32")
        output_dir.mkdir(parents=True)
        vocals = output_dir / "vocals.wav"
        accompaniment = output_dir / "accompaniment.wav"
        f0 = output_dir / "vocals-f0.npy"
        sf.write(vocals, samples * 0.6, sample_rate, subtype="PCM_24")
        sf.write(accompaniment, samples * 0.4, sample_rate, subtype="PCM_24")
        np.save(f0, np.full(round(len(samples) / sample_rate * 50), 220, dtype=np.float32))
        return StemResult(vocals=vocals, accompaniment=accompaniment, f0=f0)


def test_catalog_stem_pipeline_creates_and_reuses_source_linked_bundle(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/test-artist"
    normalized = artist_root / "data/derived/normalized/song-one.wav"
    normalized.parent.mkdir(parents=True)
    time = np.arange(48000, dtype=np.float64) / 48000
    sf.write(normalized, 0.1 * np.sin(2 * np.pi * 220 * time), 48000, subtype="PCM_24")
    source_sha256 = "a" * 64
    CatalogRepository(artist_root, "test-artist").add(
        CatalogSong(
            id="song-one",
            title="Studio fixture",
            source_filename="fixture.mp3",
            source_sha256=source_sha256,
            original_path="data/raw/originals/song-one/fixture.mp3",
            original_sha256=source_sha256,
            audio=AudioProperties(
                duration_seconds=1,
                sample_rate=48000,
                channels=1,
                codec="mp3",
                format_name="mp3",
            ),
            imported_at=datetime(2026, 8, 26, tzinfo=UTC),
            normalized_path=normalized.relative_to(artist_root).as_posix(),
            normalized_sha256=sha256_file(normalized),
        )
    )
    separator = FakeStemSeparator()
    pipeline = CatalogStemPipeline(artist_root, "test-artist", separator)

    first = pipeline.execute(_rights(source_sha256))
    second = pipeline.execute(_rights(source_sha256))

    assert first.records[0].status == "created"
    assert second.records[0].status == "reused"
    assert separator.calls == 1
    assert first.records[0].manifest.is_file()


def _rights(source_sha256: str) -> RightsManifest:
    return RightsManifest(
        artist_id="test-artist",
        status=RightsStatus.APPROVED,
        permissions=RightsPermissions(training=True, singing_voice=True),
        approvals=(
            ApprovalRecord(
                holder="test",
                scope="test",
                evidence_reference="test://approval",
                approved_at=datetime(2026, 8, 26, tzinfo=UTC),
            ),
        ),
        source_assets=(
            SourceAssetAuthorization(
                sha256=source_sha256,
                evidence_reference="test://source",
            ),
        ),
    )
