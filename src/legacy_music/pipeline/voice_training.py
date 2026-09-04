"""Prepare quality-screened clean-vocal segments for authorized voice fine-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.config import TrainingVocalConfig
from legacy_music.dataset_policy import ArtistDatasetPolicy, load_dataset_policy
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.voice import VoiceReferenceMetrics
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.repositories.catalog import CatalogRepository
from legacy_music.utils.hashing import sha256_file
from legacy_music.voice_quality import F0RateHz, VoiceQualityError, analyze_reference_samples


class VoiceTrainingDatasetError(RuntimeError):
    """Raised when a clean-only voice-training dataset cannot be prepared."""


@dataclass(frozen=True, slots=True)
class VoiceTrainingSegment:
    """One immutable clean-vocal training or validation excerpt."""

    song_id: str
    split: str
    audio: Path
    audio_sha256: str
    clean_vocals_sha256: str
    start_seconds: float
    end_seconds: float
    metrics: VoiceReferenceMetrics
    lead_metrics: LeadVocalMetrics


@dataclass(frozen=True, slots=True)
class LeadVocalMetrics:
    """Measurements that distinguish a centered, sustained lead from vocal fragments."""

    center_dominance_db: float
    longest_continuous_voicing_seconds: float
    relative_level_deficit_db: float
    confidence: float


@dataclass(frozen=True, slots=True)
class VoiceTrainingDatasetResult:
    """One complete dataset with a sanitized lineage manifest."""

    dataset_id: str
    root: Path
    manifest: Path
    segments: tuple[VoiceTrainingSegment, ...]


@dataclass(frozen=True, slots=True)
class _SegmentCandidate:
    start_seconds: float
    end_seconds: float
    metrics: VoiceReferenceMetrics
    center_dominance_db: float
    longest_continuous_voicing_seconds: float
    relative_level_deficit_db: float = 0.0
    confidence: float = 0.0


class VoiceTrainingDatasetBuilder:
    """Segment every verified clean_vocals.wav while rejecting weak excerpts."""

    def __init__(
        self,
        artist_root: Path,
        artist_id: str,
        config: TrainingVocalConfig | None = None,
    ) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id
        self.config = config or TrainingVocalConfig()
        self.policy: ArtistDatasetPolicy = load_dataset_policy(self.artist_root, artist_id)

    def prepare(
        self,
        rights: RightsManifest,
        *,
        segment_seconds: float = 15.0,
        hop_seconds: float = 5.0,
        training_segments_per_song: int = 5,
    ) -> VoiceTrainingDatasetResult:
        """Build a 1-30-second Seed-VC dataset sourced only from clean vocal stems."""
        require_capability(rights, self.artist_id, "training")
        require_capability(rights, self.artist_id, "singing_voice")
        if not 1 <= segment_seconds <= 30:
            raise VoiceTrainingDatasetError("Training segments must be between 1 and 30 seconds.")
        if not 0 < hop_seconds <= segment_seconds:
            raise VoiceTrainingDatasetError("Training segment hop must be positive and bounded.")
        if training_segments_per_song < 1:
            raise VoiceTrainingDatasetError("At least one training segment per song is required.")

        dataset_id = datetime.now(UTC).strftime("voice-dataset-%Y%m%dt%H%M%Sz-") + uuid4().hex[:8]
        dataset_root = self.artist_root / "voice/training/datasets" / dataset_id
        policy_path = self.artist_root / "data/dataset-policy.yaml"
        if dataset_root.exists():
            raise VoiceTrainingDatasetError(f"Voice dataset already exists: {dataset_id}")
        segments: list[VoiceTrainingSegment] = []
        source_records = []
        excluded_records = []
        catalog = {
            song.id: song
            for song in CatalogRepository(self.artist_root, self.artist_id).load().songs
        }

        try:
            for manifest_path in sorted(
                (self.artist_root / "data/derived/stems").glob("*/stems.json")
            ):
                payload = load_json(manifest_path)
                if not isinstance(payload, dict):
                    raise VoiceTrainingDatasetError("Stem manifest must contain a JSON object.")
                song_id = str(payload.get("song_id", ""))
                source_sha256 = str(payload.get("source_sha256", ""))
                song = catalog.get(song_id)
                recording_label = (
                    f"{song.title} {song.source_filename}"
                    if song is not None
                    else str(payload.get("recording_kind", ""))
                )
                source_exclusion = self.policy.source_exclusion(source_sha256)
                if source_exclusion is not None:
                    excluded_records.append(
                        {"song_id": song_id, "reason": source_exclusion.reason, "scope": "source"}
                    )
                    continue
                if self.policy.is_voice_song_excluded(song_id, recording_label):
                    song_policy = self.policy.voice_policy(song_id)
                    excluded_records.append(
                        {
                            "song_id": song_id,
                            "reason": (
                                song_policy.reason
                                if song_policy is not None and song_policy.reason is not None
                                else (
                                    "live-or-concert recordings are excluded from "
                                    "lead-voice training"
                                )
                            ),
                            "scope": "voice_training",
                        }
                    )
                    continue
                song_segments, source_record = self._prepare_song(
                    payload,
                    rights,
                    dataset_root,
                    segment_seconds,
                    hop_seconds,
                    training_segments_per_song,
                )
                segments.extend(song_segments)
                source_records.append(source_record)
            if not source_records:
                raise VoiceTrainingDatasetError("No clean vocal stem manifests were found.")
            if any(record["training_segment_count"] < 1 for record in source_records):
                raise VoiceTrainingDatasetError(
                    "Every clean vocal source must contribute at least one training segment."
                )

            manifest = dataset_root / "dataset.json"
            dump_json_atomic(
                manifest,
                {
                    "schema_version": 1,
                    "dataset_id": dataset_id,
                    "artist_id": self.artist_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "source_kind": "clean_vocals",
                    "dataset_policy_sha256": (
                        sha256_file(policy_path) if policy_path.is_file() else None
                    ),
                    "settings": {
                        "segment_seconds": segment_seconds,
                        "hop_seconds": hop_seconds,
                        "training_segments_per_song": training_segments_per_song,
                        "sample_rate": 44100,
                        "channels": 1,
                        "format": "wav-pcm24",
                        "lead_min_voiced_fraction": self.config.lead_min_voiced_fraction,
                        "lead_max_voiced_fraction": self.config.lead_max_voiced_fraction,
                        "lead_max_level_range_db": self.config.lead_max_level_range_db,
                        "lead_max_relative_level_deficit_db": (
                            self.config.lead_max_relative_level_deficit_db
                        ),
                        "lead_min_continuous_voicing_seconds": (
                            self.config.lead_min_continuous_voicing_seconds
                        ),
                        "lead_min_high_band_ratio_db": (
                            self.config.lead_min_high_band_ratio_db
                        ),
                        "lead_min_center_dominance_db": (
                            self.config.lead_min_center_dominance_db
                        ),
                    },
                    "song_count": len(source_records),
                    "training_segment_count": sum(
                        segment.split == "train" for segment in segments
                    ),
                    "validation_segment_count": sum(
                        segment.split == "validation" for segment in segments
                    ),
                    "sources": source_records,
                    "excluded_sources": excluded_records,
                    "segments": [
                        {
                            "song_id": segment.song_id,
                            "split": segment.split,
                            "file": segment.audio.relative_to(dataset_root).as_posix(),
                            "sha256": segment.audio_sha256,
                            "clean_vocals_sha256": segment.clean_vocals_sha256,
                            "start_seconds": segment.start_seconds,
                            "end_seconds": segment.end_seconds,
                            "metrics": segment.metrics.model_dump(mode="json"),
                            "lead_metrics": {
                                "center_dominance_db": segment.lead_metrics.center_dominance_db,
                                "longest_continuous_voicing_seconds": (
                                    segment.lead_metrics.longest_continuous_voicing_seconds
                                ),
                                "relative_level_deficit_db": (
                                    segment.lead_metrics.relative_level_deficit_db
                                ),
                                "confidence": segment.lead_metrics.confidence,
                            },
                        }
                        for segment in segments
                    ],
                },
            )
        except Exception:
            self._remove_incomplete_dataset(dataset_root)
            raise

        return VoiceTrainingDatasetResult(dataset_id, dataset_root, manifest, tuple(segments))

    def _prepare_song(
        self,
        payload: object,
        rights: RightsManifest,
        dataset_root: Path,
        segment_seconds: float,
        hop_seconds: float,
        training_segments_per_song: int,
    ) -> tuple[list[VoiceTrainingSegment], dict[str, object]]:
        if not isinstance(payload, dict):
            raise VoiceTrainingDatasetError("Stem manifest must contain a JSON object.")
        song_id = payload.get("song_id")
        if not isinstance(song_id, str) or payload.get("artist_id") != self.artist_id:
            raise VoiceTrainingDatasetError("Stem manifest artist or song identity is invalid.")
        source_sha256 = str(payload.get("source_sha256", ""))
        require_source_asset(rights, source_sha256)
        vocals = payload.get("vocals")
        clean = payload.get("clean_vocals")
        f0_payload = payload.get("f0")
        if not isinstance(vocals, dict) or not isinstance(clean, dict):
            raise VoiceTrainingDatasetError(
                f"Stem '{song_id}' is missing vocals.wav or clean_vocals.wav metadata."
            )
        if not isinstance(f0_payload, dict):
            raise VoiceTrainingDatasetError(f"Stem '{song_id}' is missing F0 metadata.")
        clean_path = self._resolve(clean.get("path"))
        f0_path = self._resolve(f0_payload.get("path"))
        if clean_path.name != "clean_vocals.wav":
            raise VoiceTrainingDatasetError(
                f"Training source for '{song_id}' must be named clean_vocals.wav."
            )
        clean_sha256 = sha256_file(clean_path)
        if clean_sha256 != clean.get("sha256"):
            raise VoiceTrainingDatasetError(f"Clean vocal hash mismatch for '{song_id}'.")
        if clean.get("source_vocals_sha256") != vocals.get("sha256"):
            raise VoiceTrainingDatasetError(f"Clean vocal lineage mismatch for '{song_id}'.")
        if sha256_file(f0_path) != f0_payload.get("sha256"):
            raise VoiceTrainingDatasetError(f"F0 hash mismatch for '{song_id}'.")

        audio, sample_rate = sf.read(clean_path, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1, dtype=np.float32)
        f0 = np.load(f0_path, allow_pickle=False).astype(np.float32).reshape(-1)
        candidates = self._candidates(
            song_id,
            audio,
            sample_rate,
            f0,
            segment_seconds,
            hop_seconds,
        )
        selected = self._select_non_overlapping(candidates, training_segments_per_song + 1)
        if not selected:
            raise VoiceTrainingDatasetError(
                f"No quality-screened clean vocal segment was found for '{song_id}'."
            )
        validation = selected[-1] if len(selected) > 1 else None
        training = selected[:-1] if validation is not None else selected
        song_segments = []
        for index, candidate in enumerate(training):
            song_segments.append(
                self._write_segment(
                    dataset_root,
                    song_id,
                    "train",
                    index,
                    mono,
                    sample_rate,
                    clean_sha256,
                    candidate,
                )
            )
        if validation is not None:
            song_segments.append(
                self._write_segment(
                    dataset_root,
                    song_id,
                    "validation",
                    0,
                    mono,
                    sample_rate,
                    clean_sha256,
                    validation,
                )
            )
        return song_segments, {
            "song_id": song_id,
            "source_sha256": source_sha256,
            "vocals_sha256": vocals.get("sha256"),
            "clean_vocals_sha256": clean_sha256,
            "f0_sha256": f0_payload.get("sha256"),
            "candidate_count": len(candidates),
            "training_segment_count": len(training),
            "validation_segment_count": int(validation is not None),
            "excluded_intervals": [
                {
                    "start_seconds": interval.start_seconds,
                    "end_seconds": interval.end_seconds,
                    "reason": interval.reason,
                }
                for interval in (
                    self.policy.voice_policy(song_id).excluded_intervals
                    if self.policy.voice_policy(song_id) is not None
                    else ()
                )
            ],
        }

    def _candidates(
        self,
        song_id: str,
        audio: np.ndarray,
        sample_rate: int,
        f0: np.ndarray,
        segment_seconds: float,
        hop_seconds: float,
    ) -> list[_SegmentCandidate]:
        mono = np.mean(audio, axis=1, dtype=np.float32)
        window_frames = round(segment_seconds * F0RateHz)
        hop_frames = round(hop_seconds * F0RateHz)
        candidates = []
        for start_frame in range(0, max(1, len(f0) - window_frames + 1), hop_frames):
            end_frame = start_frame + window_frames
            if end_frame > len(f0):
                continue
            start_seconds = start_frame / F0RateHz
            end_seconds = end_frame / F0RateHz
            if self.policy.interval_is_excluded(song_id, start_seconds, end_seconds):
                continue
            samples = mono[
                round(start_seconds * sample_rate) : round(end_seconds * sample_rate)
            ]
            excerpt_f0 = f0[start_frame:end_frame]
            try:
                metrics = analyze_reference_samples(samples, sample_rate, excerpt_f0)
            except VoiceQualityError:
                continue
            stereo_excerpt = audio[
                round(start_seconds * sample_rate) : round(end_seconds * sample_rate)
            ]
            center_dominance_db = _center_dominance_db(stereo_excerpt)
            longest_voicing = _longest_voiced_run_seconds(excerpt_f0)
            if (
                self.config.lead_min_voiced_fraction
                <= metrics.voiced_fraction
                <= self.config.lead_max_voiced_fraction
                and metrics.one_second_level_range_db
                <= self.config.lead_max_level_range_db
                and metrics.clipping_fraction == 0
                and metrics.active_rms_dbfs >= -45
                and metrics.peak_dbfs <= -0.1
                and metrics.high_band_ratio_db >= self.config.lead_min_high_band_ratio_db
                and longest_voicing >= self.config.lead_min_continuous_voicing_seconds
                and center_dominance_db >= self.config.lead_min_center_dominance_db
            ):
                candidates.append(
                    _SegmentCandidate(
                        start_seconds,
                        end_seconds,
                        metrics,
                        center_dominance_db,
                        longest_voicing,
                    )
                )
        if not candidates:
            return []
        reference_level = float(
            np.percentile([item.metrics.active_rms_dbfs for item in candidates], 75)
        )
        filtered = []
        for candidate in candidates:
            deficit = max(0.0, reference_level - candidate.metrics.active_rms_dbfs)
            if deficit > self.config.lead_max_relative_level_deficit_db:
                continue
            confidence = float(
                np.clip(
                    1.0
                    - deficit / max(self.config.lead_max_relative_level_deficit_db, 1e-6) * 0.35
                    - candidate.metrics.one_second_level_range_db
                    / max(self.config.lead_max_level_range_db, 1e-6)
                    * 0.25
                    - abs(candidate.metrics.voiced_fraction - 0.84) * 0.5,
                    0.0,
                    1.0,
                )
            )
            filtered.append(
                _SegmentCandidate(
                    candidate.start_seconds,
                    candidate.end_seconds,
                    candidate.metrics,
                    candidate.center_dominance_db,
                    candidate.longest_continuous_voicing_seconds,
                    deficit,
                    confidence,
                )
            )
        return filtered

    @staticmethod
    def _select_non_overlapping(
        candidates: list[_SegmentCandidate],
        limit: int,
    ) -> list[_SegmentCandidate]:
        def score(candidate: _SegmentCandidate) -> tuple[float, float]:
            metrics = candidate.metrics
            value = (
                abs(metrics.voiced_fraction - 0.82)
                + metrics.one_second_level_range_db / 20
                + max(0, -35 - metrics.active_rms_dbfs) / 20
            )
            return value, candidate.start_seconds

        selected = []
        for candidate in sorted(candidates, key=score):
            if all(
                abs(candidate.start_seconds - existing.start_seconds) >= 10
                for existing in selected
            ):
                selected.append(candidate)
            if len(selected) == limit:
                break
        return sorted(selected, key=lambda item: item.start_seconds)

    @staticmethod
    def _write_segment(
        dataset_root: Path,
        song_id: str,
        split: str,
        index: int,
        mono: np.ndarray,
        sample_rate: int,
        clean_sha256: str,
        candidate: _SegmentCandidate,
    ) -> VoiceTrainingSegment:
        output = dataset_root / split / song_id / f"segment-{index:03d}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        start = round(candidate.start_seconds * sample_rate)
        end = round(candidate.end_seconds * sample_rate)
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
        sf.write(temporary, mono[start:end], sample_rate, subtype="PCM_24")
        os.replace(temporary, output)
        return VoiceTrainingSegment(
            song_id,
            split,
            output,
            sha256_file(output),
            clean_sha256,
            candidate.start_seconds,
            candidate.end_seconds,
            candidate.metrics,
            LeadVocalMetrics(
                candidate.center_dominance_db,
                candidate.longest_continuous_voicing_seconds,
                candidate.relative_level_deficit_db,
                candidate.confidence,
            ),
        )

    def _resolve(self, relative: object) -> Path:
        if not isinstance(relative, str) or not relative:
            raise VoiceTrainingDatasetError("Stem manifest contains an invalid path.")
        path = (self.artist_root / relative).resolve()
        if self.artist_root != path and self.artist_root not in path.parents:
            raise VoiceTrainingDatasetError("Stem path escaped the artist profile.")
        if not path.is_file():
            raise VoiceTrainingDatasetError(f"Stem artifact is missing: {path.name}")
        return path

    @staticmethod
    def _remove_incomplete_dataset(dataset_root: Path) -> None:
        if not dataset_root.exists():
            return
        for path in sorted(dataset_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        dataset_root.rmdir()


def _center_dominance_db(audio: np.ndarray) -> float:
    """Measure centered energy relative to stereo-side energy for one excerpt."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        return 60.0
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(np.square(mid))))
    side_rms = float(np.sqrt(np.mean(np.square(side))))
    return float(np.clip(20 * np.log10(max(mid_rms, 1e-12) / max(side_rms, 1e-12)), -120, 120))


def _longest_voiced_run_seconds(f0: np.ndarray) -> float:
    """Return the longest uninterrupted target-voiced region in seconds."""
    longest = 0
    current = 0
    for is_voiced in np.asarray(f0).reshape(-1) > 0:
        if is_voiced:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / F0RateHz
