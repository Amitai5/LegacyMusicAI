"""Artist-confined immutable singing-voice reference repository."""

from __future__ import annotations

import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from legacy_music.audio import FfmpegAudioService, probe_audio
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.voice import (
    ResolvedVoiceReference,
    VoiceReference,
    VoiceReferenceMetrics,
    VoiceReviewStatus,
)
from legacy_music.paths import validate_artist_id
from legacy_music.persistence import PersistenceError, dump_json_atomic, file_lock, load_json
from legacy_music.repositories.artist import confined_path
from legacy_music.utils.hashing import sha256_file


class VoiceReferenceRepositoryError(RuntimeError):
    """Raised when a voice reference cannot be created or validated safely."""


class F0Extractor(Protocol):
    """Boundary implemented by an isolated pitch-extraction runtime."""

    def extract_f0(self, audio: Path, output: Path) -> Path:
        """Create an F0 contour for one vocal reference."""
        ...


class VoiceReferenceRepository:
    """Create and resolve integrity-checked references beneath one artist root."""

    def __init__(self, artist_root: Path, artist_id: str) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = validate_artist_id(artist_id)
        self.references_root = self.artist_root / "voice/references"

    def create(
        self,
        reference_id: str,
        source: Path,
        rights: RightsManifest,
        f0_extractor: F0Extractor,
        audio_service: FfmpegAudioService | None = None,
        tags: tuple[str, ...] = (),
    ) -> ResolvedVoiceReference:
        """Copy an authorized source, normalize it, and create its immutable F0 artifact."""
        return self._create(
            reference_id,
            source,
            rights,
            f0_extractor,
            audio_service=audio_service,
            tags=tags,
        )

    def create_derived(
        self,
        reference_id: str,
        source: Path,
        rights: RightsManifest,
        f0_extractor: F0Extractor,
        *,
        parent_source_sha256: str,
        source_song_id: str,
        source_stem_sha256: str,
        source_start_seconds: float,
        source_end_seconds: float,
        metrics: VoiceReferenceMetrics,
        audio_service: FfmpegAudioService | None = None,
        tags: tuple[str, ...] = (),
    ) -> ResolvedVoiceReference:
        """Create a reference derived reproducibly from an authorized catalog source."""
        if source_end_seconds <= source_start_seconds:
            raise VoiceReferenceRepositoryError(
                "Derived reference end time must be after its start time."
            )
        return self._create(
            reference_id,
            source,
            rights,
            f0_extractor,
            authorization_sha256=parent_source_sha256,
            parent_source_sha256=parent_source_sha256,
            source_song_id=source_song_id,
            source_stem_sha256=source_stem_sha256,
            source_start_seconds=source_start_seconds,
            source_end_seconds=source_end_seconds,
            metrics=metrics,
            audio_service=audio_service,
            tags=tags,
        )

    def _create(
        self,
        reference_id: str,
        source: Path,
        rights: RightsManifest,
        f0_extractor: F0Extractor,
        *,
        authorization_sha256: str | None = None,
        parent_source_sha256: str | None = None,
        source_song_id: str | None = None,
        source_stem_sha256: str | None = None,
        source_start_seconds: float | None = None,
        source_end_seconds: float | None = None,
        metrics: VoiceReferenceMetrics | None = None,
        audio_service: FfmpegAudioService | None = None,
        tags: tuple[str, ...] = (),
    ) -> ResolvedVoiceReference:
        """Persist one exact or reproducibly derived authorized voice prompt."""
        safe_id = validate_artist_id(reference_id)
        require_capability(rights, self.artist_id, "singing_voice")
        if not source.is_file():
            raise VoiceReferenceRepositoryError(f"Voice reference does not exist: {source}")
        probe_audio(source)
        source_hash = sha256_file(source)
        require_source_asset(rights, authorization_sha256 or source_hash)

        target = self.references_root / safe_id
        service = audio_service or FfmpegAudioService()
        self.references_root.mkdir(parents=True, exist_ok=True)
        try:
            with file_lock(self.references_root / ".references.lock"):
                if target.exists():
                    raise VoiceReferenceRepositoryError(
                        f"Voice reference '{safe_id}' already exists."
                    )
                target.mkdir()
                marker = target / ".creating"
                marker.touch()

                source_copy = target / f"source{source.suffix.lower()}"
                temporary = target / f".{source_copy.name}.copying"
                shutil.copyfile(source, temporary)
                if sha256_file(temporary) != source_hash:
                    raise VoiceReferenceRepositoryError("Copied voice reference hash mismatch.")
                os.replace(temporary, source_copy)

                audio = target / "audio.wav"
                f0 = target / "f0.npy"
                service.prepare_voice_reference(source_copy, audio)
                f0_extractor.extract_f0(audio, f0)
                reference = VoiceReference(
                    id=safe_id,
                    created_at=datetime.now(UTC),
                    source_file=source_copy.relative_to(self.artist_root),
                    source_sha256=source_hash,
                    audio_file=audio.relative_to(self.artist_root),
                    audio_sha256=sha256_file(audio),
                    f0_file=f0.relative_to(self.artist_root),
                    f0_sha256=sha256_file(f0),
                    tags=tags,
                    parent_source_sha256=parent_source_sha256,
                    source_song_id=source_song_id,
                    source_stem_sha256=source_stem_sha256,
                    source_start_seconds=source_start_seconds,
                    source_end_seconds=source_end_seconds,
                    metrics=metrics,
                )
                dump_json_atomic(
                    target / "reference.json",
                    reference.model_dump(mode="json"),
                )
                source_copy.chmod(stat.S_IREAD)
                marker.unlink()
        except Exception as error:
            _remove_incomplete_reference(target)
            if isinstance(error, VoiceReferenceRepositoryError):
                raise
            raise VoiceReferenceRepositoryError(
                f"Unable to create voice reference '{safe_id}': {error}"
            ) from error
        return self.resolve(safe_id)

    def update_metrics(
        self,
        reference_id: str,
        metrics: VoiceReferenceMetrics,
    ) -> ResolvedVoiceReference:
        """Add or refresh derived acoustic metadata without changing reference audio."""
        safe_id = validate_artist_id(reference_id)
        with file_lock(self.references_root / ".references.lock"):
            reference = self.get(safe_id)
            updated = reference.model_copy(update={"metrics": metrics})
            dump_json_atomic(
                self.references_root / safe_id / "reference.json",
                updated.model_dump(mode="json"),
            )
        return self.resolve(safe_id)

    def update_review_status(
        self,
        reference_id: str,
        review_status: VoiceReviewStatus,
    ) -> ResolvedVoiceReference:
        """Update only the review decision while preserving immutable audio lineage."""
        safe_id = validate_artist_id(reference_id)
        with file_lock(self.references_root / ".references.lock"):
            reference = self.get(safe_id)
            updated = reference.model_copy(update={"review_status": review_status})
            dump_json_atomic(
                self.references_root / safe_id / "reference.json",
                updated.model_dump(mode="json"),
            )
        return self.resolve(safe_id)

    def get(self, reference_id: str) -> VoiceReference:
        """Load validated reference metadata without exposing another artist's paths."""
        safe_id = validate_artist_id(reference_id)
        root = self.references_root / safe_id
        if (root / ".creating").exists():
            raise VoiceReferenceRepositoryError(
                f"Voice reference '{safe_id}' is incomplete."
            )
        try:
            reference = VoiceReference.model_validate(load_json(root / "reference.json"))
        except (PersistenceError, ValidationError) as error:
            raise VoiceReferenceRepositoryError(
                f"Invalid voice reference '{safe_id}': {error}"
            ) from error
        if reference.id != safe_id:
            raise VoiceReferenceRepositoryError(
                "Voice reference identity does not match its directory."
            )
        return reference

    def resolve(self, reference_id: str) -> ResolvedVoiceReference:
        """Resolve and hash-check every artifact before it can reach a model."""
        reference = self.get(reference_id)
        source = confined_path(self.artist_root, reference.source_file)
        audio = confined_path(self.artist_root, reference.audio_file)
        f0 = confined_path(self.artist_root, reference.f0_file)
        expected = {
            source: reference.source_sha256,
            audio: reference.audio_sha256,
            f0: reference.f0_sha256,
        }
        for path, digest in expected.items():
            if not path.is_file() or sha256_file(path) != digest:
                raise VoiceReferenceRepositoryError(
                    f"Voice reference artifact failed integrity validation: {path.name}"
                )
        return ResolvedVoiceReference(
            reference=reference,
            source=source,
            audio=audio,
            f0=f0,
        )

    def list(self) -> tuple[VoiceReference, ...]:
        """Return complete reference records in stable identifier order."""
        if not self.references_root.exists():
            return ()
        records = []
        for child in sorted(self.references_root.iterdir(), key=lambda item: item.name):
            if child.is_dir() and not child.name.startswith("."):
                records.append(self.get(child.name))
        return tuple(records)


def _remove_incomplete_reference(target: Path) -> None:
    if not target.exists():
        return
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(target, ignore_errors=True)
