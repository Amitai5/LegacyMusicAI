from pathlib import Path

import pytest

from legacy_music.paths import ArtistPathError, ProjectPaths, validate_artist_id


@pytest.mark.parametrize(
    "artist_id",
    ["artist-a", "artist-2", "a"],
)
def test_validate_artist_id_with_canonical_value_returns_value(artist_id: str) -> None:
    assert validate_artist_id(artist_id) == artist_id


@pytest.mark.parametrize(
    "artist_id",
    ["../artist-b", "Artist-A", "artist_a", "artist--a", "artist/a", ""],
)
def test_validate_artist_id_with_unsafe_value_raises(artist_id: str) -> None:
    with pytest.raises(ArtistPathError):
        validate_artist_id(artist_id)


def test_artist_root_with_valid_id_resolves_one_isolated_profile(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)

    result = paths.artist_root("artist-a")

    assert result == tmp_path.resolve() / "artists" / "artist-a"
