"""Crash-safe, artist-scoped catalog persistence."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from legacy_music.dataset_policy import load_dataset_policy
from legacy_music.domain.catalog import ArtistCatalog, CatalogSong
from legacy_music.persistence import PersistenceError, dump_yaml_atomic, file_lock
from legacy_music.repositories.artist import confined_path


class CatalogRepositoryError(RuntimeError):
    """Raised when an artist catalog is invalid or cannot be updated."""


class CatalogRepository:
    """Persist a versioned catalog beneath one pre-resolved artist root."""

    def __init__(self, artist_root: Path, artist_id: str) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id
        self.path = confined_path(self.artist_root, "data/catalog.yaml")
        self.lock_path = confined_path(self.artist_root, "data/.catalog.lock")

    def load(self) -> ArtistCatalog:
        """Load and validate the current catalog."""
        if not self.path.exists():
            return ArtistCatalog(artist_id=self.artist_id)
        try:
            value = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            catalog = ArtistCatalog.model_validate(value)
        except (OSError, yaml.YAMLError, ValidationError) as error:
            raise CatalogRepositoryError(f"Invalid artist catalog: {error}") from error
        if catalog.artist_id != self.artist_id:
            raise CatalogRepositoryError("Catalog identity does not match the artist profile.")
        return catalog

    def add(self, song: CatalogSong) -> ArtistCatalog:
        """Append a song unless its immutable hash is already cataloged."""
        load_dataset_policy(self.artist_root, self.artist_id).require_source_allowed(
            song.source_sha256,
            song.source_filename,
        )
        try:
            with file_lock(self.lock_path):
                catalog = self.load()
                existing = next(
                    (item for item in catalog.songs if item.source_sha256 == song.source_sha256),
                    None,
                )
                if existing is not None:
                    return catalog
                if any(item.id == song.id for item in catalog.songs):
                    raise CatalogRepositoryError(f"Catalog song ID collision: {song.id}")
                updated = catalog.model_copy(update={"songs": (*catalog.songs, song)})
                dump_yaml_atomic(self.path, updated.model_dump(mode="json"))
                return updated
        except PersistenceError as error:
            raise CatalogRepositoryError(str(error)) from error

    def remove_source_hashes(self, source_hashes: set[str]) -> ArtistCatalog:
        """Remove explicitly denied source records while preserving catalog ordering."""
        try:
            with file_lock(self.lock_path):
                catalog = self.load()
                retained = tuple(
                    song for song in catalog.songs if song.source_sha256 not in source_hashes
                )
                updated = catalog.model_copy(update={"songs": retained})
                dump_yaml_atomic(self.path, updated.model_dump(mode="json"))
                return updated
        except PersistenceError as error:
            raise CatalogRepositoryError(str(error)) from error
