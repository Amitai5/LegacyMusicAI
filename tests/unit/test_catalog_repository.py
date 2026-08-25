from datetime import UTC, datetime
from pathlib import Path

from legacy_music.domain.catalog import AudioProperties, CatalogSong
from legacy_music.repositories.catalog import CatalogRepository


def song(source_hash: str = "a" * 64) -> CatalogSong:
    return CatalogSong(
        id=f"song-fixture-{source_hash[:8]}",
        title="Fixture",
        source_filename="fixture.wav",
        source_sha256=source_hash,
        original_path="data/raw/originals/song-fixture/fixture.wav",
        original_sha256=source_hash,
        audio=AudioProperties(
            duration_seconds=10,
            sample_rate=48000,
            channels=2,
            codec="pcm_s24le",
            format_name="wav",
        ),
        imported_at=datetime(2026, 8, 24, tzinfo=UTC),
    )


def test_add_persists_artist_scoped_song_and_deduplicates_hash(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/test-artist"
    (artist_root / "data").mkdir(parents=True)
    repository = CatalogRepository(artist_root, "test-artist")

    first = repository.add(song())
    second = repository.add(song())

    assert len(first.songs) == 1
    assert second == first
    assert repository.load() == first
