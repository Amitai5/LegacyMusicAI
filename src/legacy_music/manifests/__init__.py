"""Loaders and writers for persisted application contracts."""

from legacy_music.manifests.artist import load_artist_profile
from legacy_music.manifests.rights import load_rights_manifest

__all__ = ["load_artist_profile", "load_rights_manifest"]
