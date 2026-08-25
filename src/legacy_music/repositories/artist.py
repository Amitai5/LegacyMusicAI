"""Atomic artist-profile repository."""

from __future__ import annotations

import shutil
from pathlib import Path

from legacy_music.domain.artist import (
    ArtistCapabilities,
    ArtistProfile,
    VoiceProfile,
)
from legacy_music.domain.rights import RightsManifest
from legacy_music.manifests.artist import load_artist_profile
from legacy_music.manifests.rights import load_rights_manifest
from legacy_music.paths import ProjectPaths, validate_artist_id
from legacy_music.persistence import dump_yaml_atomic, file_lock


class ArtistRepositoryError(RuntimeError):
    """Raised when artist state cannot be created or loaded safely."""


ArtistDirectories = (
    "data/raw/originals",
    "data/derived/normalized",
    "data/derived/stems",
    "data/metadata",
    "datasets/ace-step",
    "models/music",
    "models/voice",
    "voice/references",
    "presets",
    "evaluations",
    "logs",
)


class ArtistRepository:
    """Create and load artist profiles without crossing profile roots."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def create(
        self,
        artist_id: str,
        display_name: str,
        singing_voice: bool = False,
        voice_engine: str = "soulx",
    ) -> ArtistProfile:
        """Create a pending artist profile and complete private directory layout."""
        safe_id = validate_artist_id(artist_id)
        target = self.paths.artist_root(safe_id)
        if target.exists():
            raise ArtistRepositoryError(f"Artist '{safe_id}' already exists.")

        self.paths.artists.mkdir(parents=True, exist_ok=True)
        profile = ArtistProfile(
            id=safe_id,
            display_name=display_name,
            capabilities=ArtistCapabilities(music_style=True, singing_voice=singing_voice),
            voice=VoiceProfile(engine=voice_engine),
        )
        rights = RightsManifest(artist_id=safe_id)

        try:
            with file_lock(self.paths.artists / ".artists.lock"):
                if target.exists():
                    raise ArtistRepositoryError(f"Artist '{safe_id}' already exists.")
                target.mkdir()
                marker = target / ".creating"
                marker.touch()
                for relative in ArtistDirectories:
                    (target / relative).mkdir(parents=True)
                dump_yaml_atomic(target / "artist.yaml", profile.model_dump(mode="json"))
                dump_yaml_atomic(target / "rights.yaml", rights.model_dump(mode="json"))
                dump_yaml_atomic(
                    target / "data/catalog.yaml",
                    {"schema_version": 1, "artist_id": safe_id, "songs": []},
                )
                marker.unlink()
        except Exception as error:
            shutil.rmtree(target, ignore_errors=True)
            if isinstance(error, ArtistRepositoryError):
                raise
            raise ArtistRepositoryError(f"Unable to create artist '{safe_id}': {error}") from error

        return profile

    def list(self) -> tuple[ArtistProfile, ...]:
        """Return all valid local artist profiles in stable identifier order."""
        if not self.paths.artists.exists():
            return ()
        profiles = []
        for child in sorted(self.paths.artists.iterdir(), key=lambda item: item.name):
            if child.name.startswith("_") or child.name.startswith(".") or not child.is_dir():
                continue
            profiles.append(self.get(child.name))
        return tuple(profiles)

    def get(self, artist_id: str) -> ArtistProfile:
        """Load one artist profile and verify its on-disk identity."""
        root = self.paths.artist_root(artist_id)
        if (root / ".creating").exists():
            raise ArtistRepositoryError(f"Artist '{artist_id}' has an incomplete profile.")
        profile = load_artist_profile(root / "artist.yaml")
        if profile.id != artist_id:
            raise ArtistRepositoryError("Artist profile identity does not match its directory.")
        return profile

    def get_rights(self, artist_id: str) -> RightsManifest:
        """Load one artist's rights manifest and verify its identity."""
        profile = self.get(artist_id)
        root = self.paths.artist_root(artist_id)
        rights_path = confined_path(root, profile.rights.manifest)
        rights = load_rights_manifest(rights_path)
        if rights.artist_id != artist_id:
            raise ArtistRepositoryError(
                "Rights manifest identity does not match the artist profile."
            )
        return rights


def confined_path(root: Path, relative: Path | str) -> Path:
    """Resolve a relative path while rejecting absolute paths and symlink escape."""
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ArtistRepositoryError("Artist manifest paths must be relative.")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ArtistRepositoryError("Artist path escaped its isolated profile root.")
    return candidate
