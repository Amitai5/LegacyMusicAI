"""Artist-profile manifest loading."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from legacy_music.domain.artist import ArtistProfile


class ArtistManifestError(RuntimeError):
    """Raised when an artist profile cannot be parsed or validated."""


def load_artist_profile(path: Path) -> ArtistProfile:
    """Load and strictly validate one artist YAML profile."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ArtistProfile.model_validate(value)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ArtistManifestError(f"Invalid artist profile '{path}': {error}") from error
