"""Prepare reusable catalog vocal stems with immutable source lineage."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from legacy_music.audio import probe_audio
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.dataset_policy import load_dataset_policy
from legacy_music.domain.rights import RightsManifest
from legacy_music.engines.base import StemSeparator
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.repositories.artist import confined_path
from legacy_music.repositories.catalog import CatalogRepository
from legacy_music.utils.hashing import sha256_file


class StemPreparationError(RuntimeError):
    """Raised when reusable catalog stems cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class PreparedStemRecord:
    """One verified catalog stem bundle."""

    song_id: str
    source_sha256: str
    status: str
    manifest: Path


@dataclass(frozen=True, slots=True)
class StemPreparationResult:
    """Complete reusable stem-preparation batch result."""

    records: tuple[PreparedStemRecord, ...]


class CatalogStemPipeline:
    """Separate every eligible normalized catalog song into validated reusable stems."""

    def __init__(self, artist_root: Path, artist_id: str, separator: StemSeparator) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id
        self.separator = separator

    def execute(self, rights: RightsManifest) -> StemPreparationResult:
        """Create or integrity-check stem bundles for every allowed catalog recording."""
        require_capability(rights, self.artist_id, "training")
        require_capability(rights, self.artist_id, "singing_voice")
        policy = load_dataset_policy(self.artist_root, self.artist_id)
        catalog = CatalogRepository(self.artist_root, self.artist_id).load()
        stems_root = self.artist_root / "data/derived/stems"
        stems_root.mkdir(parents=True, exist_ok=True)
        records = []

        for song in catalog.songs:
            if policy.source_exclusion(song.source_sha256, song.source_filename) is not None:
                continue
            require_source_asset(rights, song.source_sha256)
            if song.normalized_path is None or song.normalized_sha256 is None:
                raise StemPreparationError(f"Song '{song.id}' has no normalized artifact.")
            normalized = confined_path(self.artist_root, song.normalized_path)
            if not normalized.is_file() or sha256_file(normalized) != song.normalized_sha256:
                raise StemPreparationError(f"Normalized source failed integrity for '{song.id}'.")

            output_root = stems_root / song.id
            manifest_path = output_root / "stems.json"
            if manifest_path.is_file():
                self._verify_existing(
                    manifest_path,
                    song.id,
                    song.source_sha256,
                    song.normalized_sha256,
                )
                records.append(
                    PreparedStemRecord(song.id, song.source_sha256, "reused", manifest_path)
                )
                continue
            if output_root.exists():
                raise StemPreparationError(
                    f"Untracked stem directory exists for '{song.id}'; refusing overwrite."
                )

            staging = stems_root / f".{song.id}.{uuid4().hex}.staging"
            try:
                result = self.separator.separate(normalized, staging)
                if result.f0 is None:
                    raise StemPreparationError(
                        f"Stem separator did not create an F0 contour for '{song.id}'."
                    )
                vocal_properties = probe_audio(result.vocals)
                accompaniment_properties = probe_audio(result.accompaniment)
                source_properties = probe_audio(normalized)
                tolerance = 0.05
                if (
                    abs(vocal_properties.duration_seconds - source_properties.duration_seconds)
                    > tolerance
                    or abs(
                        accompaniment_properties.duration_seconds
                        - source_properties.duration_seconds
                    )
                    > tolerance
                ):
                    raise StemPreparationError(f"Stem timing changed for '{song.id}'.")
                f0 = np.load(result.f0, allow_pickle=False).astype(np.float32).reshape(-1)
                if f0.size == 0 or not np.any(f0 > 0):
                    raise StemPreparationError(f"Stem F0 is empty for '{song.id}'.")

                os.replace(staging, output_root)
                vocals = output_root / result.vocals.name
                accompaniment = output_root / result.accompaniment.name
                f0_path = output_root / result.f0.name
                searchable = f"{song.title} {song.source_filename}".casefold()
                recording_kind = (
                    "live"
                    if any(marker in searchable for marker in ("live", "concert", "shabahangi"))
                    else "studio"
                )
                dump_json_atomic(
                    manifest_path,
                    {
                        "schema_version": 1,
                        "artist_id": self.artist_id,
                        "song_id": song.id,
                        "source_sha256": song.source_sha256,
                        "normalized_sha256": song.normalized_sha256,
                        "recording_kind": recording_kind,
                        "created_at": datetime.now(UTC).isoformat(),
                        "vocals": {
                            "path": vocals.relative_to(self.artist_root).as_posix(),
                            "sha256": sha256_file(vocals),
                            "duration_seconds": vocal_properties.duration_seconds,
                        },
                        "accompaniment": {
                            "path": accompaniment.relative_to(self.artist_root).as_posix(),
                            "sha256": sha256_file(accompaniment),
                            "duration_seconds": accompaniment_properties.duration_seconds,
                        },
                        "f0": {
                            "path": f0_path.relative_to(self.artist_root).as_posix(),
                            "sha256": sha256_file(f0_path),
                        },
                    },
                )
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                if output_root.exists() and not manifest_path.exists():
                    shutil.rmtree(output_root)
                raise
            records.append(
                PreparedStemRecord(song.id, song.source_sha256, "created", manifest_path)
            )

        if not records:
            raise StemPreparationError("No eligible catalog songs were available for stems.")
        return StemPreparationResult(tuple(records))

    def _verify_existing(
        self,
        manifest_path: Path,
        song_id: str,
        source_sha256: str,
        normalized_sha256: str,
    ) -> None:
        payload = load_json(manifest_path)
        if not isinstance(payload, dict) or (
            payload.get("artist_id") != self.artist_id
            or payload.get("song_id") != song_id
            or payload.get("source_sha256") != source_sha256
            or payload.get("normalized_sha256") != normalized_sha256
        ):
            raise StemPreparationError(f"Existing stem lineage mismatch for '{song_id}'.")
        for key in ("vocals", "accompaniment", "f0"):
            artifact = payload.get(key)
            if not isinstance(artifact, dict):
                raise StemPreparationError(f"Existing '{key}' stem metadata is invalid.")
            path = confined_path(self.artist_root, str(artifact.get("path", "")))
            if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
                raise StemPreparationError(f"Existing '{key}' stem failed integrity.")
