"""Build a quality-screened multi-register prompt bank from authorized vocal stems."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from legacy_music.audio import FfmpegAudioService
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.dataset_policy import load_dataset_policy
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.voice import (
    ResolvedVoiceReference,
    VoiceReferenceMetrics,
    VoiceReviewStatus,
)
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.repositories.catalog import CatalogRepository
from legacy_music.repositories.voice import VoiceReferenceRepository
from legacy_music.utils.hashing import sha256_file
from legacy_music.voice_quality import F0RateHz, analyze_reference, analyze_reference_samples

ReferenceLabels = ("neutral", "soft", "strong", "low", "high")
ExcludedRecordingMarkers = ("live", "concert", "shabahangi")


class VoiceBankError(RuntimeError):
    """Raised when an authorized voice bank cannot be built safely."""


@dataclass(frozen=True, slots=True)
class VoiceBankCandidate:
    """One quality-screened excerpt candidate backed by a verified stem manifest."""

    song_id: str
    parent_source_sha256: str
    stem_sha256: str
    vocals_path: Path
    f0_path: Path
    start_seconds: float
    end_seconds: float
    metrics: VoiceReferenceMetrics


@dataclass(frozen=True, slots=True)
class VoiceBankResult:
    """Resolved references and sanitized bank manifest created by one build."""

    references: tuple[ResolvedVoiceReference, ...]
    manifest: Path


class ArrayF0Extractor:
    """Write an already validated excerpt from a parent stem F0 contour."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=np.float32).reshape(-1).copy()

    def extract_f0(self, _audio: Path, output: Path) -> Path:
        """Persist the precomputed F0 excerpt at the repository-requested path."""
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, self.values)
        return output


class VoiceBankBuilder:
    """Curate deterministic low, mid, high, soft, and strong artist prompts."""

    def __init__(self, artist_root: Path, artist_id: str) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id
        self.references = VoiceReferenceRepository(self.artist_root, artist_id)
        self.catalog = CatalogRepository(self.artist_root, artist_id)
        self.policy = load_dataset_policy(self.artist_root, artist_id)

    def build(
        self,
        rights: RightsManifest,
        audio_service: FfmpegAudioService | None = None,
    ) -> VoiceBankResult:
        """Build missing bank members, enrich legacy prompts, and persist bank metadata."""
        require_capability(rights, self.artist_id, "singing_voice")
        service = audio_service or FfmpegAudioService()
        existing = {reference.id: reference for reference in self.references.list()}
        current_clean_hashes = {
            str(clean.get("sha256"))
            for path in (self.artist_root / "data/derived/stems").glob("*/stems.json")
            if isinstance((payload := load_json(path)), dict)
            and isinstance((clean := payload.get("clean_vocals")), dict)
        }
        for reference_id, reference in tuple(existing.items()):
            if reference.metrics is None:
                resolved = self.references.resolve(reference_id)
                existing[reference_id] = self.references.update_metrics(
                    reference_id,
                    analyze_reference(resolved.audio, resolved.f0),
                ).reference

        approved_by_role = {}
        for reference_id, reference in tuple(existing.items()):
            role = next(
                (
                    label
                    for label in ReferenceLabels
                    if reference_id == label or label in reference.tags
                ),
                None,
            )
            if role is None:
                continue
            source_excluded = (
                reference.parent_source_sha256 is not None
                and self.policy.source_exclusion(reference.parent_source_sha256) is not None
            )
            voice_song_excluded = (
                reference.source_song_id is not None
                and self.policy.is_voice_song_excluded(reference.source_song_id)
            )
            stale_clean_stem = (
                "clean-vocal-stem" in reference.tags
                and reference.source_stem_sha256 not in current_clean_hashes
            )
            if source_excluded or voice_song_excluded or stale_clean_stem:
                if reference.review_status is VoiceReviewStatus.APPROVED:
                    existing[reference_id] = self.references.update_review_status(
                        reference_id,
                        VoiceReviewStatus.REJECTED,
                    ).reference
                continue
            if (
                reference.review_status is VoiceReviewStatus.APPROVED
                and reference.metrics is not None
                and _passes_bank_quality(reference.metrics)
                and "clean-vocal-stem" in reference.tags
            ):
                approved_by_role.setdefault(role, reference)
            elif (
                reference.review_status is VoiceReviewStatus.APPROVED
                and reference.metrics is not None
                and not _passes_bank_quality(reference.metrics)
            ):
                existing[reference_id] = self.references.update_review_status(
                    reference_id,
                    VoiceReviewStatus.REJECTED,
                ).reference

        missing = [label for label in ReferenceLabels if label not in approved_by_role]
        if missing:
            candidates = self._scan_candidates(rights)
            selected = self._select_candidates(candidates, missing)
            for label in missing:
                reference_id = _next_reference_id(label, existing)
                resolved = self._materialize(
                    reference_id,
                    label,
                    selected[label],
                    rights,
                    service,
                )
                existing[reference_id] = resolved.reference
                approved_by_role[label] = resolved.reference

        resolved_references = tuple(
            self.references.resolve(approved_by_role[label].id) for label in ReferenceLabels
        )
        manifest_path = self.artist_root / "voice/bank.json"
        dump_json_atomic(
            manifest_path,
            {
                "schema_version": 2,
                "artist_id": self.artist_id,
                "created_at": datetime.now(UTC).isoformat(),
                "source_kind": "clean_vocals",
                "references": [
                    {
                        "id": item.reference.id,
                        "role": next(
                            label for label in ReferenceLabels if label in item.reference.tags
                        ),
                        "audio_sha256": item.reference.audio_sha256,
                        "f0_sha256": item.reference.f0_sha256,
                        "source_sha256": item.reference.source_sha256,
                        "parent_source_sha256": item.reference.parent_source_sha256,
                        "review_status": item.reference.review_status.value,
                        "metrics": (
                            item.reference.metrics.model_dump(mode="json")
                            if item.reference.metrics is not None
                            else None
                        ),
                    }
                    for item in resolved_references
                ],
            },
        )
        return VoiceBankResult(references=resolved_references, manifest=manifest_path)

    def _scan_candidates(self, rights: RightsManifest) -> tuple[VoiceBankCandidate, ...]:
        catalog = {song.id: song for song in self.catalog.load().songs}
        candidates = []
        stems_root = self.artist_root / "data/derived/stems"
        for manifest_path in sorted(stems_root.glob("*/stems.json")):
            payload = load_json(manifest_path)
            song_id = payload.get("song_id")
            song = catalog.get(song_id)
            if song is None or payload.get("artist_id") != self.artist_id:
                continue
            searchable = f"{song.title} {song.source_filename}".casefold()
            if any(marker in searchable for marker in ExcludedRecordingMarkers):
                continue
            parent_hash = str(payload.get("source_sha256", ""))
            if self.policy.source_exclusion(parent_hash, song.source_filename) is not None:
                continue
            if self.policy.is_voice_song_excluded(song_id, searchable):
                continue
            require_source_asset(rights, parent_hash)
            vocal_payload = payload.get("clean_vocals", {})
            f0_payload = payload.get("f0", {})
            if not isinstance(vocal_payload, dict) or not vocal_payload:
                raise VoiceBankError(
                    f"Stem '{song_id}' has no clean_vocals entry; run voice clean-vocals first."
                )
            vocals = self._resolve_derived(vocal_payload.get("path"))
            f0_path = self._resolve_derived(f0_payload.get("path"))
            if sha256_file(vocals) != vocal_payload.get("sha256"):
                raise VoiceBankError(f"Clean vocal stem hash mismatch for '{song_id}'.")
            if sha256_file(f0_path) != f0_payload.get("sha256"):
                raise VoiceBankError(f"F0 stem hash mismatch for '{song_id}'.")
            audio, sample_rate = sf.read(vocals, dtype="float32", always_2d=True)
            f0 = np.load(f0_path, allow_pickle=False).astype(np.float32).reshape(-1)
            candidates.extend(
                self._song_candidates(
                    song_id,
                    parent_hash,
                    str(vocal_payload["sha256"]),
                    vocals,
                    f0_path,
                    audio,
                    sample_rate,
                    f0,
                )
            )
        if not candidates:
            raise VoiceBankError("No studio vocal excerpts passed voice-bank quality gates.")
        return tuple(candidates)

    def _song_candidates(
        self,
        song_id: str,
        parent_hash: str,
        stem_hash: str,
        vocals_path: Path,
        f0_path: Path,
        audio: np.ndarray,
        sample_rate: int,
        f0: np.ndarray,
    ) -> list[VoiceBankCandidate]:
        window_frames = 15 * F0RateHz
        hop_frames = 2 * F0RateHz
        candidates = []
        for start_frame in range(0, max(1, len(f0) - window_frames + 1), hop_frames):
            end_frame = start_frame + window_frames
            if end_frame > len(f0):
                continue
            excerpt_f0 = f0[start_frame:end_frame]
            voiced_fraction = float(np.mean(excerpt_f0 > 0))
            if not 0.70 <= voiced_fraction <= 0.95:
                continue
            start_seconds = start_frame / F0RateHz
            end_seconds = end_frame / F0RateHz
            if self.policy.interval_is_excluded(song_id, start_seconds, end_seconds):
                continue
            excerpt_audio = audio[
                round(start_seconds * sample_rate) : round(end_seconds * sample_rate)
            ]
            metrics = analyze_reference_samples(excerpt_audio, sample_rate, excerpt_f0)
            if (
                metrics.one_second_level_range_db > 6
                or metrics.clipping_fraction > 0
                or metrics.active_rms_dbfs < -40
            ):
                continue
            candidates.append(
                VoiceBankCandidate(
                    song_id=song_id,
                    parent_source_sha256=parent_hash,
                    stem_sha256=stem_hash,
                    vocals_path=vocals_path,
                    f0_path=f0_path,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    metrics=metrics,
                )
            )
        return candidates

    @staticmethod
    def _select_candidates(
        candidates: tuple[VoiceBankCandidate, ...],
        labels: list[str],
    ) -> dict[str, VoiceBankCandidate]:
        pitches = np.array([item.metrics.median_f0_hz for item in candidates])
        levels = np.array([item.metrics.active_rms_dbfs for item in candidates])
        pitch_min, pitch_max = float(np.min(pitches)), float(np.max(pitches))
        level_min, level_max = float(np.min(levels)), float(np.max(levels))

        def score(label: str, item: VoiceBankCandidate) -> tuple[float, str, float]:
            pitch = (item.metrics.median_f0_hz - pitch_min) / max(pitch_max - pitch_min, 1e-6)
            level = (item.metrics.active_rms_dbfs - level_min) / max(
                level_max - level_min,
                1e-6,
            )
            quality = (
                abs(item.metrics.voiced_fraction - 0.85)
                + item.metrics.one_second_level_range_db / 12
            )
            category = {
                "neutral": abs(pitch - 0.5) + abs(level - 0.5),
                "soft": level + 0.5 * abs(pitch - 0.5),
                "strong": 1 - level + 0.5 * abs(pitch - 0.5),
                "low": pitch + 0.25 * abs(level - 0.5),
                "high": 1 - pitch + 0.25 * abs(level - 0.5),
            }[label]
            return category + 0.2 * quality, item.song_id, item.start_seconds

        selected = {}
        used_songs: set[str] = set()
        for label in labels:
            available = [item for item in candidates if item.song_id not in used_songs]
            if not available:
                available = list(candidates)
            choice = min(available, key=lambda item: score(label, item))
            selected[label] = choice
            used_songs.add(choice.song_id)
        return selected

    def _materialize(
        self,
        reference_id: str,
        label: str,
        candidate: VoiceBankCandidate,
        rights: RightsManifest,
        service: FfmpegAudioService,
    ) -> ResolvedVoiceReference:
        audio, sample_rate = sf.read(candidate.vocals_path, dtype="float32", always_2d=True)
        f0 = np.load(candidate.f0_path, allow_pickle=False).astype(np.float32).reshape(-1)
        start_sample = round(candidate.start_seconds * sample_rate)
        end_sample = round(candidate.end_seconds * sample_rate)
        start_frame = round(candidate.start_seconds * F0RateHz)
        end_frame = round(candidate.end_seconds * F0RateHz)
        excerpt_audio = audio[start_sample:end_sample]
        excerpt_f0 = f0[start_frame:end_frame]

        candidates_root = self.artist_root / "voice/candidates"
        candidates_root.mkdir(parents=True, exist_ok=True)
        start_milliseconds = round(candidate.start_seconds * 1000)
        source = candidates_root / (
            f"bank-{label}-{candidate.song_id[-8:]}-{start_milliseconds:06d}ms.wav"
        )
        temporary = source.with_name(f".{source.stem}.{uuid4().hex}.wav")
        sf.write(temporary, excerpt_audio, sample_rate, subtype="PCM_24")
        os.replace(temporary, source)

        resolved = self.references.create_derived(
            reference_id,
            source,
            rights,
            ArrayF0Extractor(excerpt_f0),
            parent_source_sha256=candidate.parent_source_sha256,
            source_song_id=candidate.song_id,
            source_stem_sha256=candidate.stem_sha256,
            source_start_seconds=candidate.start_seconds,
            source_end_seconds=candidate.end_seconds,
            metrics=candidate.metrics,
            audio_service=service,
            tags=(label, "studio", "derived-vocal-stem", "clean-vocal-stem"),
        )
        measured = analyze_reference(resolved.audio, resolved.f0)
        return self.references.update_metrics(reference_id, measured)

    def _resolve_derived(self, relative: object) -> Path:
        if not isinstance(relative, str) or not relative:
            raise VoiceBankError("Stem manifest contains an invalid path.")
        candidate = (self.artist_root / relative).resolve()
        if self.artist_root != candidate and self.artist_root not in candidate.parents:
            raise VoiceBankError("Stem path escaped the artist profile.")
        if not candidate.is_file():
            raise VoiceBankError(f"Stem artifact is missing: {candidate.name}")
        return candidate


def _passes_bank_quality(metrics: VoiceReferenceMetrics) -> bool:
    return (
        12 <= metrics.duration_seconds <= 18
        and 0.70 <= metrics.voiced_fraction <= 0.95
        and metrics.one_second_level_range_db <= 6
        and metrics.clipping_fraction == 0
        and metrics.active_rms_dbfs >= -40
    )


def _next_reference_id(role: str, existing: dict[str, object]) -> str:
    if role not in existing:
        return role
    revision = 2
    while f"{role}-v{revision}" in existing:
        revision += 1
    return f"{role}-v{revision}"
