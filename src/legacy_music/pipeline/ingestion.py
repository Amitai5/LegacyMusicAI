"""Rights-gated immutable audio ingestion workflow."""

from __future__ import annotations

import os
import re
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from legacy_music.audio import FfmpegAudioService, probe_audio
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.domain.catalog import CatalogSong
from legacy_music.domain.rights import RightsManifest
from legacy_music.repositories.catalog import CatalogRepository
from legacy_music.utils.hashing import sha256_file

SupportedAudioExtensions = {
    ".aac",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}


class IngestModel(BaseModel):
    """Base model for immutable ingest reporting."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InventoryItem(IngestModel):
    """Hash and validation result for a discovered audio source."""

    path: Path
    sha256: str
    is_authorized: bool
    duration_seconds: float


class IngestResult(IngestModel):
    """Outcome for one source in a resumable import."""

    source: Path
    sha256: str
    song_id: str
    status: str
    original: Path
    normalized: Path


def discover_audio(source: Path) -> tuple[Path, ...]:
    """Discover supported regular audio files recursively in stable order."""
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = [path for path in source.rglob("*") if path.is_file()]
    else:
        raise FileNotFoundError(f"Ingest source does not exist: {source}")
    return tuple(
        sorted(
            (path for path in candidates if path.suffix.lower() in SupportedAudioExtensions),
            key=lambda path: str(path).casefold(),
        )
    )


def inventory_audio(source: Path, rights: RightsManifest) -> tuple[InventoryItem, ...]:
    """Hash and validate candidate recordings without importing them."""
    authorized_hashes = {asset.sha256 for asset in rights.source_assets}
    results = []
    for path in discover_audio(source):
        sha256 = sha256_file(path)
        audio = probe_audio(path)
        results.append(
            InventoryItem(
                path=path,
                sha256=sha256,
                is_authorized=sha256 in authorized_hashes,
                duration_seconds=audio.duration_seconds,
            )
        )
    return tuple(results)


def ingest_audio(
    artist_id: str,
    artist_root: Path,
    rights: RightsManifest,
    source: Path,
    audio_service: FfmpegAudioService | None = None,
) -> tuple[IngestResult, ...]:
    """Import authorized sources byte-for-byte and create normalized derivatives."""
    require_capability(rights, artist_id, "training")
    service = audio_service or FfmpegAudioService()
    catalog_repository = CatalogRepository(artist_root, artist_id)
    catalog = catalog_repository.load()
    results = []

    for source_path in discover_audio(source):
        source_hash = sha256_file(source_path)
        require_source_asset(rights, source_hash)
        existing = next(
            (song for song in catalog.songs if song.source_sha256 == source_hash),
            None,
        )
        if existing is not None:
            results.append(
                IngestResult(
                    source=source_path,
                    sha256=source_hash,
                    song_id=existing.id,
                    status="duplicate",
                    original=artist_root / existing.original_path,
                    normalized=artist_root / (existing.normalized_path or ""),
                )
            )
            continue

        audio = probe_audio(source_path)
        song_id = _song_id(source_path.stem, source_hash)
        original_relative = Path("data/raw/originals") / song_id / source_path.name
        original = artist_root / original_relative
        normalized_relative = Path("data/derived/normalized") / f"{song_id}.wav"
        normalized = artist_root / normalized_relative

        original.parent.mkdir(parents=True, exist_ok=True)
        if original.exists() and sha256_file(original) != source_hash:
            raise RuntimeError(f"Immutable original collision for {song_id}.")
        if not original.exists():
            temporary = original.with_suffix(f"{original.suffix}.copying")
            shutil.copyfile(source_path, temporary)
            if sha256_file(temporary) != source_hash:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Copied source hash mismatch for {source_path.name}.")
            os.replace(temporary, original)
            original.chmod(stat.S_IREAD)

        if not normalized.exists():
            service.normalize(original, normalized)
        normalized_hash = sha256_file(normalized)
        song = CatalogSong(
            id=song_id,
            title=source_path.stem,
            source_filename=source_path.name,
            source_sha256=source_hash,
            original_path=original_relative.as_posix(),
            original_sha256=sha256_file(original),
            audio=audio,
            imported_at=datetime.now(UTC),
            normalized_path=normalized_relative.as_posix(),
            normalized_sha256=normalized_hash,
        )
        catalog = catalog_repository.add(song)
        results.append(
            IngestResult(
                source=source_path,
                sha256=source_hash,
                song_id=song_id,
                status="imported",
                original=original,
                normalized=normalized,
            )
        )

    return tuple(results)


def _song_id(stem: str, sha256: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "recording"
    return f"song-{slug[:40].rstrip('-')}-{sha256[:8]}"
