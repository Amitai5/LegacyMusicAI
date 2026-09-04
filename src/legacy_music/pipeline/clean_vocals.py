"""Create immutable-lineage dereverberated derivatives for authorized vocal stems."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from legacy_music.audio import probe_audio
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.dataset_policy import load_dataset_policy
from legacy_music.domain.rights import RightsManifest
from legacy_music.persistence import dump_json_atomic, file_lock, load_json
from legacy_music.utils.hashing import sha256_file


class CleanVocalError(RuntimeError):
    """Raised when a clean vocal derivative cannot be created safely."""


class DereverbRuntime(Protocol):
    """Boundary for one-load batch dereverberation runtimes."""

    def dereverb_batch(
        self,
        jobs: tuple[tuple[Path, Path], ...],
        *,
        strength: float,
    ) -> tuple[dict[str, object], ...]:
        """Dereverberate each source into its distinct output path."""
        ...


@dataclass(frozen=True, slots=True)
class CleanVocalRecord:
    """One verified original/clean vocal pair and its source lineage."""

    song_id: str
    vocals: Path
    clean_vocals: Path
    source_sha256: str
    vocals_sha256: str
    clean_vocals_sha256: str
    status: str


@dataclass(frozen=True, slots=True)
class CleanVocalResult:
    """Completed batch manifest and stable per-song outputs."""

    cleaning_id: str
    manifest: Path
    records: tuple[CleanVocalRecord, ...]


@dataclass(frozen=True, slots=True)
class _PendingStem:
    song_id: str
    manifest_path: Path
    payload: dict[str, object]
    source_sha256: str
    vocals: Path
    vocals_sha256: str
    temporary: Path
    output: Path


class CleanVocalPipeline:
    """Dereverberate every authorized vocal stem without modifying its source."""

    def __init__(self, artist_root: Path, artist_id: str, runtime: DereverbRuntime) -> None:
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id
        self.runtime = runtime

    def execute(
        self,
        rights: RightsManifest,
        *,
        model_sha256: str,
        config_sha256: str,
        dereverb_strength: float = 0.9,
        rebuild: bool = False,
    ) -> CleanVocalResult:
        """Create or reuse clean_vocals.wav for every verified stem manifest."""
        require_capability(rights, self.artist_id, "training")
        require_capability(rights, self.artist_id, "singing_voice")
        if len(model_sha256) != 64 or len(config_sha256) != 64:
            raise CleanVocalError("Dereverberation model identity must use SHA-256 hashes.")
        if not 0 <= dereverb_strength <= 1:
            raise CleanVocalError("Dereverberation strength must be between zero and one.")

        cleaning_id = datetime.now(UTC).strftime("cleaning-%Y%m%dt%H%M%Sz-") + uuid4().hex[:8]
        cleaning_root = self.artist_root / "voice/cleaning" / cleaning_id
        manifest_path = cleaning_root / "manifest.json"
        records: list[CleanVocalRecord] = []
        pending: list[_PendingStem] = []
        lock_path = self.artist_root / "voice/.clean-vocals.lock"
        policy = load_dataset_policy(self.artist_root, self.artist_id)

        with file_lock(lock_path):
            for stem_manifest in sorted(
                (self.artist_root / "data/derived/stems").glob("*/stems.json")
            ):
                stem_payload = load_json(stem_manifest)
                if isinstance(stem_payload, dict) and policy.source_exclusion(
                    str(stem_payload.get("source_sha256", ""))
                ) is not None:
                    continue
                prepared = self._prepare_stem(
                    stem_manifest,
                    rights,
                    model_sha256,
                    config_sha256,
                    dereverb_strength,
                    rebuild,
                )
                if isinstance(prepared, CleanVocalRecord):
                    records.append(prepared)
                else:
                    pending.append(prepared)

            if not records and not pending:
                raise CleanVocalError("No stem manifests were available to clean.")

            runtime_results: tuple[dict[str, object], ...] = ()
            try:
                if pending:
                    runtime_results = self.runtime.dereverb_batch(
                        tuple((item.vocals, item.temporary) for item in pending),
                        strength=dereverb_strength,
                    )
                if len(runtime_results) != len(pending):
                    raise CleanVocalError("Dereverberation returned an incomplete result batch.")
                for item, metrics in zip(pending, runtime_results, strict=True):
                    records.append(
                        self._finalize_stem(
                            item,
                            metrics,
                            model_sha256,
                            config_sha256,
                            dereverb_strength,
                        )
                    )
            finally:
                for item in pending:
                    item.temporary.unlink(missing_ok=True)

            records.sort(key=lambda item: item.song_id)
            dump_json_atomic(
                manifest_path,
                {
                    "schema_version": 1,
                    "cleaning_id": cleaning_id,
                    "artist_id": self.artist_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "purpose": "authorized-voice-training-and-reference-bank",
                    "model_sha256": model_sha256,
                    "config_sha256": config_sha256,
                    "dereverb_strength": dereverb_strength,
                    "source_restoration_ratio": 1 - dereverb_strength,
                    "source_count": len(records),
                    "created_count": sum(record.status == "created" for record in records),
                    "reused_count": sum(record.status == "reused" for record in records),
                    "records": [
                        {
                            "song_id": record.song_id,
                            "source_sha256": record.source_sha256,
                            "vocals_sha256": record.vocals_sha256,
                            "clean_vocals_sha256": record.clean_vocals_sha256,
                            "status": record.status,
                        }
                        for record in records
                    ],
                },
            )

        return CleanVocalResult(cleaning_id, manifest_path, tuple(records))

    def _prepare_stem(
        self,
        manifest_path: Path,
        rights: RightsManifest,
        model_sha256: str,
        config_sha256: str,
        dereverb_strength: float,
        rebuild: bool,
    ) -> CleanVocalRecord | _PendingStem:
        payload = load_json(manifest_path)
        if not isinstance(payload, dict):
            raise CleanVocalError(f"Stem manifest is not a JSON object: {manifest_path}")
        song_id = payload.get("song_id")
        if not isinstance(song_id, str) or payload.get("artist_id") != self.artist_id:
            raise CleanVocalError(f"Stem manifest identity is invalid: {manifest_path}")
        source_sha256 = str(payload.get("source_sha256", ""))
        require_source_asset(rights, source_sha256)
        vocal_payload = payload.get("vocals")
        if not isinstance(vocal_payload, dict):
            raise CleanVocalError(f"Stem manifest has no vocals entry for '{song_id}'.")
        vocals = self._resolve(vocal_payload.get("path"))
        if vocals.name != "vocals.wav":
            raise CleanVocalError(f"Source vocal stem must be named vocals.wav for '{song_id}'.")
        vocals_sha256 = sha256_file(vocals)
        if vocals_sha256 != vocal_payload.get("sha256"):
            raise CleanVocalError(f"Source vocal stem hash mismatch for '{song_id}'.")
        output = vocals.with_name("clean_vocals.wav")

        clean_payload = payload.get("clean_vocals")
        cleaning = payload.get("vocal_cleaning")
        if output.exists():
            if not isinstance(clean_payload, dict) or not isinstance(cleaning, dict):
                raise CleanVocalError(
                    f"Untracked clean_vocals.wav already exists for '{song_id}'; "
                    "refusing overwrite."
                )
            if (
                clean_payload.get("path") != self._relative(output)
                or clean_payload.get("source_vocals_sha256") != vocals_sha256
            ):
                raise CleanVocalError(
                    "Existing clean vocal source lineage does not match "
                    f"for '{song_id}'."
                )
            clean_sha256 = sha256_file(output)
            if clean_sha256 != clean_payload.get("sha256"):
                raise CleanVocalError(f"Clean vocal stem hash mismatch for '{song_id}'.")
            processing_matches = (
                cleaning.get("model_sha256") == model_sha256
                and cleaning.get("config_sha256") == config_sha256
                and float(cleaning.get("dereverb_strength", 1.0)) == dereverb_strength
            )
            if processing_matches:
                return CleanVocalRecord(
                    song_id,
                    vocals,
                    output,
                    source_sha256,
                    vocals_sha256,
                    clean_sha256,
                    "reused",
                )
            if not rebuild:
                raise CleanVocalError(
                    "Existing clean vocal processing does not match the requested settings "
                    f"for '{song_id}'; use rebuild after reviewing lineage."
                )

        temporary = output.with_name(f".clean_vocals.{uuid4().hex}.wav")
        return _PendingStem(
            song_id,
            manifest_path,
            payload,
            source_sha256,
            vocals,
            vocals_sha256,
            temporary,
            output,
        )

    def _finalize_stem(
        self,
        pending: _PendingStem,
        metrics: dict[str, object],
        model_sha256: str,
        config_sha256: str,
        dereverb_strength: float,
    ) -> CleanVocalRecord:
        if sha256_file(pending.vocals) != pending.vocals_sha256:
            raise CleanVocalError(f"Source vocal changed while cleaning '{pending.song_id}'.")
        source_audio = probe_audio(pending.vocals)
        clean_audio = probe_audio(pending.temporary)
        if source_audio.sample_rate != clean_audio.sample_rate:
            raise CleanVocalError(f"Clean vocal sample rate changed for '{pending.song_id}'.")
        if source_audio.channels != clean_audio.channels:
            raise CleanVocalError(f"Clean vocal channel count changed for '{pending.song_id}'.")
        tolerance = 1 / source_audio.sample_rate
        if abs(source_audio.duration_seconds - clean_audio.duration_seconds) > tolerance:
            raise CleanVocalError(f"Clean vocal duration changed for '{pending.song_id}'.")
        if int(metrics.get("clipped_sample_count", -1)) != 0:
            raise CleanVocalError(f"Clean vocal contains clipped samples for '{pending.song_id}'.")
        if int(metrics.get("nonfinite_sample_count", -1)) != 0:
            raise CleanVocalError(
                f"Clean vocal contains non-finite samples for '{pending.song_id}'."
            )

        os.replace(pending.temporary, pending.output)
        clean_sha256 = sha256_file(pending.output)
        updated = dict(pending.payload)
        updated["schema_version"] = max(int(updated.get("schema_version", 1)), 2)
        updated["clean_vocals"] = {
            "path": self._relative(pending.output),
            "sha256": clean_sha256,
            "source_vocals_sha256": pending.vocals_sha256,
            "duration_seconds": clean_audio.duration_seconds,
            "sample_rate": clean_audio.sample_rate,
            "channels": clean_audio.channels,
        }
        updated["vocal_cleaning"] = {
            "method": "mel-band-roformer-dereverberation",
            "model_sha256": model_sha256,
            "config_sha256": config_sha256,
            "dereverb_strength": dereverb_strength,
            "source_restoration_ratio": 1 - dereverb_strength,
            "chunk_seconds": 5.0,
            "active_level_match_db": metrics.get("active_level_match_db"),
            "peak_reduction_db": metrics.get("peak_reduction_db"),
            "active_rms_dbfs": metrics.get("active_rms_dbfs"),
            "peak_dbfs": metrics.get("peak_dbfs"),
            "clipped_sample_count": 0,
            "nonfinite_sample_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
        }
        dump_json_atomic(pending.manifest_path, updated)
        return CleanVocalRecord(
            pending.song_id,
            pending.vocals,
            pending.output,
            pending.source_sha256,
            pending.vocals_sha256,
            clean_sha256,
            "created",
        )

    def _resolve(self, relative: object) -> Path:
        if not isinstance(relative, str) or not relative:
            raise CleanVocalError("Stem manifest contains an invalid vocal path.")
        path = (self.artist_root / relative).resolve()
        if self.artist_root != path and self.artist_root not in path.parents:
            raise CleanVocalError("Stem path escaped the artist profile.")
        if not path.is_file():
            raise CleanVocalError(f"Stem artifact is missing: {path.name}")
        return path

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        if self.artist_root != resolved and self.artist_root not in resolved.parents:
            raise CleanVocalError("Clean vocal path escaped the artist profile.")
        return resolved.relative_to(self.artist_root).as_posix()
