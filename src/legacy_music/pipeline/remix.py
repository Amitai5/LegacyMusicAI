"""Immutable quality-first vocal remix orchestration for an existing draft run."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from legacy_music.audio import (
    AudioMasteringResult,
    FfmpegAudioService,
    VocalLevelMatchResult,
    probe_audio,
)
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.config import FinalVocalMixConfig, VoiceQualityConfig
from legacy_music.domain.generation import (
    GenerationResult,
    LyricsRequest,
    PipelineStage,
    VoiceGenerationRequest,
)
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.run import ProvenanceManifest, RunManifest
from legacy_music.domain.voice import (
    ResolvedVoiceReference,
    VoiceConversionRequest,
    VoiceConversionResult,
    VoiceReviewStatus,
)
from legacy_music.engines.base import VoiceEngine
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.repositories.run import RunRepository
from legacy_music.utils.hashing import sha256_file


class VocalRemixError(RuntimeError):
    """Raised after an immutable vocal-remix child run fails validation or QC."""


class VocalRemixPipeline:
    """Reuse a parent draft and stems while replacing only its vocal post-processing."""

    def __init__(
        self,
        run_repository: RunRepository,
        voice_engine: VoiceEngine,
        quality_config: VoiceQualityConfig,
        audio_service: FfmpegAudioService | None = None,
        mix_config: FinalVocalMixConfig | None = None,
    ) -> None:
        self.runs = run_repository
        self.voice_engine = voice_engine
        self.quality = quality_config
        self.mix_config = mix_config or FinalVocalMixConfig()
        self.audio = audio_service or FfmpegAudioService()

    def execute(
        self,
        parent_run_id: str,
        rights: RightsManifest,
        references: tuple[ResolvedVoiceReference, ...],
        *,
        reference_label: str = "auto",
        duration_seconds: int | None = None,
        voice_engine_id: str | None = None,
    ) -> GenerationResult:
        """Create a child run and quality-remix an exact parent draft and stem set."""
        parent = self.runs.load(parent_run_id)
        self._validate_authorization(parent, rights, references)
        parent_root = self.runs.root(parent_run_id)
        parent_artifacts = self._resolve_parent_artifacts(parent, parent_root)
        parent_hashes = {name: sha256_file(path) for name, path in parent_artifacts.items()}
        self._verify_recorded_hashes(parent, parent_hashes)
        self._verify_parent_timing(parent_artifacts)

        duration = duration_seconds or parent.request.music.duration_seconds
        parent_duration = probe_audio(parent_artifacts["draft"]).duration_seconds
        if duration > parent_duration + 0.02:
            raise VocalRemixError(
                f"Requested {duration}s exceeds the {parent_duration:.3f}s parent draft."
            )
        child_request = parent.request.model_copy(
            update={
                "id": f"generation-{uuid4().hex}",
                "lyrics": LyricsRequest(
                    file=parent_artifacts["lyrics"],
                    sha256=parent_hashes["lyrics"],
                ),
                "music": parent.request.music.model_copy(update={"duration_seconds": duration}),
                "voice": VoiceGenerationRequest(
                    enabled=True,
                    engine=voice_engine_id or parent.request.voice.engine,
                    reference=reference_label,
                ),
                "parent_run_id": parent_run_id,
            }
        )
        child_run_id = self.runs.new_run_id()
        manifest = self.runs.create(child_run_id, child_request)
        try:
            child_root = self.runs.root(child_run_id)
            child_artifacts = self._copy_parent_inputs(
                parent_artifacts,
                child_root,
                duration,
                parent.request.music.duration_seconds,
            )
            manifest = self.runs.transition(
                child_run_id,
                PipelineStage.MUSIC_GENERATED,
                "Parent ACE-Step draft reused without music regeneration.",
                artifacts={"draft": "intermediates/ace-step-draft.wav"},
                hashes={"draft": sha256_file(child_artifacts["draft"])},
            )
            manifest = self.runs.transition(
                child_run_id,
                PipelineStage.VOCALS_SEPARATED,
                "Parent stems reused without source separation.",
                artifacts={
                    "vocals": "intermediates/stems/vocals.wav",
                    "accompaniment": "intermediates/stems/accompaniment.wav",
                    "target_f0": "intermediates/stems/vocals-f0.npy",
                },
                hashes={
                    "vocals": sha256_file(child_artifacts["vocals"]),
                    "accompaniment": sha256_file(child_artifacts["accompaniment"]),
                    "target_f0": sha256_file(child_artifacts["target_f0"]),
                },
            )

            converted_path = child_root / "intermediates/converted-vocals.wav"
            conversion = self.voice_engine.convert(
                VoiceConversionRequest(
                    target_audio=child_artifacts["vocals"],
                    target_f0=child_artifacts["target_f0"],
                    references=references,
                    engine=child_request.voice.engine,
                    seed=child_request.music.seed,
                ),
                converted_path,
            )
            manifest = self.runs.transition(
                child_run_id,
                PipelineStage.VOICE_CONVERTED,
                "Quality-first multi-reference singing voice conversion completed.",
                artifacts={
                    "converted_vocals": "intermediates/converted-vocals.wav",
                    "cleaned_f0": str(conversion.cleaned_f0.relative_to(child_root).as_posix()),
                    "voice_conversion_manifest": str(
                        conversion.manifest.relative_to(child_root).as_posix()
                    ),
                },
                hashes={
                    "converted_vocals": sha256_file(conversion.audio),
                    "cleaned_f0": sha256_file(conversion.cleaned_f0),
                    "voice_conversion_manifest": sha256_file(conversion.manifest),
                },
            )

            leveled_path = child_root / "intermediates/mix/leveled-vocals.wav"
            level_match = self.audio.match_vocal_level(
                child_artifacts["vocals"],
                conversion.audio,
                conversion.cleaned_f0,
                leveled_path,
                local_gain_limit_db=self.quality.local_gain_limit_db,
            )
            final_path = child_root / "output/final.wav"
            mastering = self.audio.mix_and_master(
                child_artifacts["accompaniment"],
                level_match.audio,
                final_path,
                compressor_threshold_dbfs=self.quality.compressor_threshold_dbfs,
                compressor_ratio=self.quality.compressor_ratio,
                compressor_attack_ms=self.quality.compressor_attack_ms,
                compressor_release_ms=self.quality.compressor_release_ms,
                vocal_presence_gain_db=self.mix_config.vocal_presence_gain_db,
                instrumental_gain_db=self.mix_config.instrumental_gain_db,
                presence_eq_frequency_hz=self.mix_config.presence_eq_frequency_hz,
                presence_eq_gain_db=self.mix_config.presence_eq_gain_db,
                reverb_wet=self.mix_config.reverb_wet,
                reverb_pre_delay_ms=self.mix_config.reverb_pre_delay_ms,
                reverb_decay=self.mix_config.reverb_decay,
                automatic_vocal_balance=self.mix_config.automatic_vocal_balance,
                target_vocal_to_instrumental_db=(self.mix_config.target_vocal_to_instrumental_db),
                maximum_automatic_balance_correction_db=(
                    self.mix_config.maximum_automatic_balance_correction_db
                ),
                target_lufs=self.quality.target_lufs,
                target_lra=self.quality.target_lra,
                true_peak_dbfs=self.quality.true_peak_dbfs,
            )
            manifest = self.runs.transition(
                child_run_id,
                PipelineStage.MIXED,
                "Guide-matched vocals mixed and two-pass EBU R128 mastering completed.",
                artifacts={
                    "leveled_vocals": "intermediates/mix/leveled-vocals.wav",
                    "vocal_bus": str(mastering.vocal_bus_audio.relative_to(child_root).as_posix()),
                    "accompaniment_bus": str(
                        mastering.accompaniment_bus_audio.relative_to(child_root).as_posix()
                    ),
                    "premaster": str(mastering.premaster_audio.relative_to(child_root).as_posix()),
                    "mastering_manifest": str(
                        mastering.manifest.relative_to(child_root).as_posix()
                    ),
                    "final": "output/final.wav",
                },
                hashes={
                    "leveled_vocals": sha256_file(level_match.audio),
                    "vocal_bus": sha256_file(mastering.vocal_bus_audio),
                    "accompaniment_bus": sha256_file(mastering.accompaniment_bus_audio),
                    "premaster": sha256_file(mastering.premaster_audio),
                    "mastering_manifest": sha256_file(mastering.manifest),
                    "final": sha256_file(mastering.final_audio),
                },
            )

            qc_path = child_root / "output/quality-control.json"
            qc = self._quality_control(
                duration,
                conversion,
                level_match,
                mastering,
                qc_path,
                parent_run_id,
                parent_hashes,
            )
            if not qc["accepted"]:
                failures = "; ".join(qc["failures"])
                raise VocalRemixError(f"Quality acceptance failed: {failures}")

            reference_by_id = {item.reference.id: item for item in references}
            first_reference = reference_by_id[conversion.reference_ids[0]]
            source_hashes = {
                item.reference.id: item.reference.parent_source_sha256
                or item.reference.source_sha256
                for item in references
            }
            parent_provenance = self._load_parent_provenance(parent_root)
            provenance = ProvenanceManifest(
                schema_version=2,
                run_id=child_run_id,
                artist_id=child_request.artist_id,
                created_at=datetime.now(UTC),
                prompt=child_request.music.prompt,
                lyrics_sha256=parent_hashes["lyrics"],
                seed=child_request.music.seed,
                music_engine="ace-step",
                music_model=parent_provenance.get("music_model"),
                music_adapter=child_request.music.adapter,
                voice_engine=child_request.voice.engine,
                voice_reference_id=first_reference.reference.id,
                voice_reference_sha256=first_reference.reference.audio_sha256,
                voice_source_sha256=source_hashes[first_reference.reference.id],
                voice_reference_f0_sha256=first_reference.reference.f0_sha256,
                voice_reference_ids=conversion.reference_ids,
                voice_reference_hashes={
                    item.reference.id: item.reference.audio_sha256 for item in references
                },
                voice_reference_f0_hashes={
                    item.reference.id: item.reference.f0_sha256 for item in references
                },
                voice_reference_stem_hashes={
                    item.reference.id: item.reference.source_stem_sha256
                    for item in references
                    if item.reference.source_stem_sha256 is not None
                },
                voice_source_hashes=source_hashes,
                voice_conversion_manifest=conversion.manifest.relative_to(child_root),
                voice_conversion_manifest_sha256=sha256_file(conversion.manifest),
                mastering_manifest=mastering.manifest.relative_to(child_root),
                mastering_manifest_sha256=sha256_file(mastering.manifest),
                quality_control_manifest=qc_path.relative_to(child_root),
                quality_control_manifest_sha256=sha256_file(qc_path),
                parent_run_id=parent_run_id,
                parent_artifact_hashes=parent_hashes,
                output=Path("output/final.wav"),
                output_sha256=sha256_file(final_path),
            )
            provenance_path = child_root / "output/provenance.json"
            dump_json_atomic(provenance_path, provenance.model_dump(mode="json"))
            self._verify_parent_immutable(parent_artifacts, parent_hashes)
            manifest = self.runs.transition(
                child_run_id,
                PipelineStage.COMPLETE,
                "Vocal reconstruction passed objective QC; human review remains required.",
                artifacts={
                    "final": "output/final.wav",
                    "provenance": "output/provenance.json",
                    "quality_control": "output/quality-control.json",
                },
                hashes={
                    "final": provenance.output_sha256,
                    "provenance": sha256_file(provenance_path),
                    "quality_control": sha256_file(qc_path),
                },
            )
            return GenerationResult(
                run_id=child_run_id,
                stage=manifest.stage,
                final_audio=final_path,
                provenance=provenance_path,
            )
        except Exception as error:
            self._verify_parent_immutable(parent_artifacts, parent_hashes)
            if manifest.stage not in {PipelineStage.COMPLETE, PipelineStage.FAILED}:
                self.runs.transition(
                    child_run_id,
                    PipelineStage.FAILED,
                    "Vocal remix failed; the parent run remains immutable.",
                    error=str(error),
                )
            if isinstance(error, VocalRemixError):
                raise
            raise VocalRemixError(str(error)) from error

    @staticmethod
    def _validate_authorization(
        parent: RunManifest,
        rights: RightsManifest,
        references: tuple[ResolvedVoiceReference, ...],
    ) -> None:
        if parent.stage is not PipelineStage.COMPLETE:
            raise VocalRemixError("Parent run must be complete before it can be remixed.")
        require_capability(rights, parent.artist_id, "music_style")
        require_capability(rights, parent.artist_id, "singing_voice")
        if not references:
            raise VocalRemixError("At least one approved voice reference is required.")
        for resolved in references:
            if resolved.reference.review_status is not VoiceReviewStatus.APPROVED:
                raise VocalRemixError(f"Voice reference '{resolved.reference.id}' is not approved.")
            authorization_hash = (
                resolved.reference.parent_source_sha256 or resolved.reference.source_sha256
            )
            require_source_asset(rights, authorization_hash)

    @staticmethod
    def _resolve_parent_artifacts(parent: RunManifest, root: Path) -> dict[str, Path]:
        artifacts: dict[str, str] = {}
        for event in parent.events:
            artifacts.update(event.artifacts)
        required = {
            "draft": artifacts.get("draft", "intermediates/ace-step-draft.wav"),
            "vocals": artifacts.get("vocals", "intermediates/stems/vocals.wav"),
            "accompaniment": artifacts.get(
                "accompaniment",
                "intermediates/stems/accompaniment.wav",
            ),
            "target_f0": artifacts.get("target_f0", "intermediates/stems/vocals-f0.npy"),
            "lyrics": "inputs/lyrics.txt",
        }
        resolved = {}
        for name, relative in required.items():
            path = (root / relative).resolve()
            if root != path and root not in path.parents:
                raise VocalRemixError(f"Parent {name} path escaped its run directory.")
            if not path.is_file():
                raise VocalRemixError(f"Parent {name} artifact is missing: {path.name}")
            resolved[name] = path
        for name in ("draft", "vocals", "accompaniment"):
            probe_audio(resolved[name])
        try:
            f0 = np.load(resolved["target_f0"], allow_pickle=False)
        except (OSError, ValueError) as error:
            raise VocalRemixError(f"Parent target F0 is invalid: {error}") from error
        if f0.size == 0 or not np.any(np.asarray(f0) > 0):
            raise VocalRemixError("Parent target F0 contains no voiced frames.")
        return resolved

    @staticmethod
    def _verify_recorded_hashes(parent: RunManifest, actual: dict[str, str]) -> None:
        recorded: dict[str, str] = {}
        for event in parent.events:
            recorded.update(event.hashes)
        for name in ("draft", "vocals", "accompaniment", "target_f0"):
            expected = recorded.get(name)
            if expected is not None and expected != actual[name]:
                raise VocalRemixError(f"Parent {name} failed its recorded hash check.")
        lyrics_expected = parent.request.lyrics.sha256
        if lyrics_expected is not None and lyrics_expected != actual["lyrics"]:
            raise VocalRemixError("Parent lyrics failed their recorded hash check.")

    @staticmethod
    def _verify_parent_timing(artifacts: dict[str, Path]) -> None:
        durations = {
            name: probe_audio(artifacts[name]).duration_seconds
            for name in ("draft", "vocals", "accompaniment")
        }
        if max(durations.values()) - min(durations.values()) > 0.05:
            raise VocalRemixError(f"Parent draft and stems are misaligned: {durations}")
        f0_frames = len(np.load(artifacts["target_f0"], allow_pickle=False).reshape(-1))
        if abs(f0_frames / 50 - durations["vocals"]) > 0.05:
            raise VocalRemixError("Parent vocal stem and target F0 have inconsistent timing.")

    @staticmethod
    def _copy_parent_inputs(
        parent: dict[str, Path],
        child_root: Path,
        duration: int,
        parent_request_duration: int,
    ) -> dict[str, Path]:
        destinations = {
            "lyrics": child_root / "inputs/lyrics.txt",
            "draft": child_root / "intermediates/ace-step-draft.wav",
            "vocals": child_root / "intermediates/stems/vocals.wav",
            "accompaniment": child_root / "intermediates/stems/accompaniment.wav",
            "target_f0": child_root / "intermediates/stems/vocals-f0.npy",
        }
        destinations["vocals"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent["lyrics"], destinations["lyrics"])
        is_full_length = duration == parent_request_duration
        for name in ("draft", "vocals", "accompaniment"):
            if is_full_length:
                shutil.copyfile(parent[name], destinations[name])
            else:
                VocalRemixPipeline._write_audio_prefix(
                    parent[name],
                    destinations[name],
                    duration,
                )
        if is_full_length:
            shutil.copyfile(parent["target_f0"], destinations["target_f0"])
        else:
            values = np.load(parent["target_f0"], allow_pickle=False).reshape(-1)
            np.save(destinations["target_f0"], values[: round(duration * 50)])
        return destinations

    @staticmethod
    def _write_audio_prefix(source: Path, output: Path, duration: int) -> None:
        try:
            audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
            info = sf.info(source)
            sf.write(
                output,
                audio[: round(duration * sample_rate)],
                sample_rate,
                subtype=info.subtype,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise VocalRemixError(f"Unable to create smoke-test audio: {error}") from error

    def _quality_control(
        self,
        expected_duration: int,
        conversion: VoiceConversionResult,
        level_match: VocalLevelMatchResult,
        mastering: AudioMasteringResult,
        output: Path,
        parent_run_id: str,
        parent_hashes: dict[str, str],
    ) -> dict[str, object]:
        final_audio = mastering.final_audio
        properties = probe_audio(final_audio)
        samples, _sample_rate = sf.read(final_audio, dtype="float32", always_2d=True)
        clipping_fraction = float(np.mean(np.abs(samples) >= 0.999999))
        coverage_complete = _has_complete_phrase_coverage(
            conversion.phrases,
            expected_duration,
        )
        checks = {
            "duration": abs(properties.duration_seconds - expected_duration) <= 0.02,
            "sample_rate": properties.sample_rate == 48000,
            "channels": properties.channels == 2,
            "pcm_24": properties.codec.lower() in {"pcm_s24le", "pcm_24"},
            "integrated_loudness": (
                abs(mastering.integrated_lufs - self.quality.target_lufs) <= 0.5
            ),
            "true_peak": mastering.true_peak_dbfs <= self.quality.true_peak_dbfs + 0.01,
            "vocal_presence": (
                self.mix_config.minimum_vocal_to_instrumental_db
                <= mastering.vocal_to_instrumental_db
                <= self.mix_config.maximum_vocal_to_instrumental_db
            ),
            "clipping": clipping_fraction == 0,
            "dropouts": conversion.metrics.sustained_dropout_count == 0,
            "join_level": conversion.metrics.max_voiced_join_jump_db <= 6.0,
            "pitch_median": (
                conversion.metrics.f0_median_error_cents is not None
                and conversion.metrics.f0_median_error_cents <= 50
            ),
            "pitch_gross": (
                conversion.metrics.f0_gross_error_fraction is not None
                and conversion.metrics.f0_gross_error_fraction < 0.05
            ),
            "voice_identity": (
                (
                    conversion.metrics.minimum_window_identity_similarity is not None
                    and conversion.metrics.identity_window_threshold is not None
                    and conversion.metrics.minimum_window_identity_similarity
                    >= conversion.metrics.identity_window_threshold
                )
                if conversion.metrics.identity_window_threshold is not None
                else (
                    conversion.metrics.identity_threshold is None
                    or (
                        conversion.metrics.minimum_identity_similarity is not None
                        and conversion.metrics.minimum_identity_similarity
                        >= conversion.metrics.identity_threshold
                    )
                )
            ),
            "phrase_coverage": coverage_complete,
        }
        failures = [name for name, passed in checks.items() if not passed]
        payload: dict[str, object] = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "parent_run_id": parent_run_id,
            "parent_artifact_hashes": parent_hashes,
            "accepted": not failures,
            "human_review_status": "pending",
            "release_approved": False,
            "checks": checks,
            "failures": failures,
            "measurements": {
                "duration_seconds": properties.duration_seconds,
                "sample_rate": properties.sample_rate,
                "channels": properties.channels,
                "codec": properties.codec,
                "integrated_lufs": mastering.integrated_lufs,
                "loudness_range_lu": mastering.loudness_range_lu,
                "true_peak_dbfs": mastering.true_peak_dbfs,
                "vocal_to_instrumental_db": mastering.vocal_to_instrumental_db,
                "pre_balance_vocal_to_instrumental_db": (
                    mastering.pre_balance_vocal_to_instrumental_db
                ),
                "automatic_balance_correction_db": (mastering.automatic_balance_correction_db),
                "automatic_vocal_adjustment_db": (mastering.automatic_vocal_adjustment_db),
                "automatic_instrumental_adjustment_db": (
                    mastering.automatic_instrumental_adjustment_db
                ),
                "vocal_presence_gain_db": self.mix_config.vocal_presence_gain_db,
                "instrumental_gain_db": self.mix_config.instrumental_gain_db,
                "restored_reverb_wet": self.mix_config.reverb_wet,
                "clipping_fraction": clipping_fraction,
                "converted_peak_dbfs": conversion.metrics.peak_dbfs,
                "sustained_dropout_count": conversion.metrics.sustained_dropout_count,
                "max_voiced_join_jump_db": (conversion.metrics.max_voiced_join_jump_db),
                "f0_median_error_cents": conversion.metrics.f0_median_error_cents,
                "f0_gross_error_fraction": (conversion.metrics.f0_gross_error_fraction),
                "minimum_identity_similarity": (conversion.metrics.minimum_identity_similarity),
                "mean_identity_similarity": (conversion.metrics.mean_identity_similarity),
                "identity_threshold": conversion.metrics.identity_threshold,
                "minimum_window_identity_similarity": (
                    conversion.metrics.minimum_window_identity_similarity
                ),
                "mean_window_identity_similarity": (
                    conversion.metrics.mean_window_identity_similarity
                ),
                "identity_window_threshold": (conversion.metrics.identity_window_threshold),
                "global_vocal_gain_db": level_match.global_gain_db,
                "local_vocal_gain_min_db": level_match.local_gain_min_db,
                "local_vocal_gain_max_db": level_match.local_gain_max_db,
            },
            "artifacts": {
                "voice_conversion_manifest_sha256": sha256_file(conversion.manifest),
                "leveled_vocals_sha256": sha256_file(level_match.audio),
                "mastering_manifest_sha256": sha256_file(mastering.manifest),
                "vocal_bus_sha256": sha256_file(mastering.vocal_bus_audio),
                "accompaniment_bus_sha256": sha256_file(mastering.accompaniment_bus_audio),
                "final_sha256": sha256_file(final_audio),
            },
        }
        dump_json_atomic(output, payload)
        return payload

    @staticmethod
    def _load_parent_provenance(parent_root: Path) -> dict[str, object]:
        path = parent_root / "output/provenance.json"
        if not path.is_file():
            return {}
        payload = load_json(path)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _verify_parent_immutable(
        parent_artifacts: dict[str, Path],
        expected_hashes: dict[str, str],
    ) -> None:
        changed = [
            name
            for name, path in parent_artifacts.items()
            if sha256_file(path) != expected_hashes[name]
        ]
        if changed:
            raise VocalRemixError("Parent run changed during remix: " + ", ".join(sorted(changed)))


def _has_complete_phrase_coverage(phrases: tuple[object, ...], duration: float) -> bool:
    if not phrases or abs(phrases[0].start_seconds) > 0.001:
        return False
    covered_until = 0.0
    for phrase in phrases:
        if phrase.start_seconds > covered_until + 0.001:
            return False
        if phrase.end_seconds <= phrase.start_seconds:
            return False
        covered_until = max(covered_until, phrase.end_seconds)
    return covered_until >= duration - 0.02
