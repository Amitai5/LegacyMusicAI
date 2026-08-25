"""Safe repository and artist path resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ArtistIdPattern = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ArtistPathError(ValueError):
    """Raised when an artist identifier could escape its isolated directory."""


def validate_artist_id(value: str) -> str:
    """Validate and return a canonical filesystem-safe artist identifier."""
    if not 1 <= len(value) <= 64 or ArtistIdPattern.fullmatch(value) is None:
        raise ArtistPathError(
            "Artist IDs must be 1-64 lowercase letters, digits, or single hyphen separators."
        )
    return value


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolved roots for repository-local application data."""

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> ProjectPaths:
        """Create project paths from an explicit repository root."""
        return cls(root=root.resolve())

    @property
    def config(self) -> Path:
        """Return the configuration directory."""
        return self.root / "config"

    @property
    def artists(self) -> Path:
        """Return the artist-profile root."""
        return self.root / "artists"

    @property
    def models(self) -> Path:
        """Return the shared-model root."""
        return self.root / "models"

    @property
    def runs(self) -> Path:
        """Return the immutable run root."""
        return self.root / "runs"

    @property
    def vendor(self) -> Path:
        """Return the isolated upstream checkout root."""
        return self.root / "vendor"

    def artist_root(self, artist_id: str) -> Path:
        """Resolve one artist root without permitting traversal or cross-profile access."""
        safe_id = validate_artist_id(artist_id)
        artists_root = self.artists.resolve()
        candidate = (artists_root / safe_id).resolve()

        if candidate.parent != artists_root:
            raise ArtistPathError(f"Artist path escaped the profile root: {artist_id}")

        return candidate
