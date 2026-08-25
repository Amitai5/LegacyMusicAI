from pathlib import Path

import pytest

from legacy_music.domain.rights import RightsStatus
from legacy_music.paths import ProjectPaths
from legacy_music.repositories.artist import ArtistRepository, ArtistRepositoryError, confined_path


def test_create_builds_complete_pending_isolated_profile(tmp_path: Path) -> None:
    repository = ArtistRepository(ProjectPaths.from_root(tmp_path))

    profile = repository.create("test-artist", "Test Artist", singing_voice=True)

    root = tmp_path / "artists/test-artist"
    assert profile.id == "test-artist"
    assert (root / "data/raw/originals").is_dir()
    assert (root / "data/derived/normalized").is_dir()
    assert (root / "datasets/ace-step").is_dir()
    assert (root / "models/music").is_dir()
    assert (root / "voice/references").is_dir()
    assert repository.get_rights("test-artist").status is RightsStatus.PENDING


def test_create_duplicate_does_not_replace_profile(tmp_path: Path) -> None:
    repository = ArtistRepository(ProjectPaths.from_root(tmp_path))
    repository.create("test-artist", "Original")

    with pytest.raises(ArtistRepositoryError, match="already exists"):
        repository.create("test-artist", "Replacement")

    assert repository.get("test-artist").display_name == "Original"


def test_confined_path_rejects_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "artist"
    root.mkdir()

    with pytest.raises(ArtistRepositoryError, match="escaped"):
        confined_path(root, "../sibling/private.wav")
