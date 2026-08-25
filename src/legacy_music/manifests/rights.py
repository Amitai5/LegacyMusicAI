"""Rights-manifest loading."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from legacy_music.domain.rights import RightsManifest


class RightsManifestError(RuntimeError):
    """Raised when a rights manifest cannot be parsed or validated."""


def load_rights_manifest(path: Path) -> RightsManifest:
    """Load and strictly validate one authorization manifest."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return RightsManifest.model_validate(value)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise RightsManifestError(f"Invalid rights manifest '{path}': {error}") from error
