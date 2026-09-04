"""Pinned Seed-VC fine-tuning adapter kept outside the main Python environment."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np
import soundfile as sf
import yaml

from legacy_music.audio import probe_audio
from legacy_music.authorization import require_capability
from legacy_music.config import VoiceQualityConfig
from legacy_music.dataset_policy import load_dataset_policy
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.voice import (
    ResolvedVoiceReference,
    VoiceCandidateMetrics,
    VoiceConversionManifest,
    VoiceConversionMetrics,
    VoiceConversionRequest,
    VoiceConversionResult,
    VoicePhraseDecision,
    VoiceReviewStatus,
)
from legacy_music.persistence import dump_json_atomic, dump_yaml_atomic, load_json
from legacy_music.repositories.artist import confined_path
from legacy_music.utils.hashing import sha256_file
from legacy_music.voice_quality import (
    compare_f0,
    plan_phrases,
    rank_references,
    write_clean_f0,
)

SeedVCCommit = "51383efd921027683c89e5348211d93ff12ac2a8"
SeedVCPreset = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"


class SeedVCError(RuntimeError):
    """Raised when an isolated Seed-VC operation fails validation."""


@dataclass(frozen=True, slots=True)
class SeedVCInstallation:
    """Resolved pinned Seed-VC source, interpreter, and writable caches."""

    python: Path
    repository: Path
    preset: Path
    runtime_script: Path
    cache_root: Path

    @classmethod
    def from_project(cls, root: Path, python: Path | None = None) -> SeedVCInstallation:
        """Resolve the standard optional Seed-VC installation."""
        project = root.resolve()
        repository = project / "vendor/seed-vc"
        return cls(
            python=(python or _find_seed_vc_python()).resolve(),
            repository=repository,
            preset=repository / SeedVCPreset,
            runtime_script=project / "scripts/seed_vc_runtime.py",
            cache_root=project / "models/cache/seed-vc",
        )

    def validate(self) -> None:
        """Require a CUDA-capable isolated runtime at the reviewed source commit."""
        required = [
            self.python,
            self.repository / ".git",
            self.repository / "train.py",
            self.preset,
            self.runtime_script,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SeedVCError("Seed-VC is incomplete; missing: " + ", ".join(missing))
        completed = subprocess.run(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or completed.stdout.strip() != SeedVCCommit:
            raise SeedVCError("Seed-VC checkout does not match the reviewed pinned commit.")
        cuda = subprocess.run(
            [
                str(self.python),
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=self.environment(),
        )
        if cuda.returncode != 0:
            raise SeedVCError("Seed-VC requires a working CUDA-enabled PyTorch runtime.")

    def environment(self) -> dict[str, str]:
        """Use project-local model caches and disable optional telemetry."""
        environment = os.environ.copy()
        cache_paths = {
            "HF_HOME": self.cache_root / "huggingface",
            "HUGGINGFACE_HUB_CACHE": self.cache_root / "huggingface/hub",
            "TRANSFORMERS_CACHE": self.cache_root / "transformers",
            "NUMBA_CACHE_DIR": self.cache_root / "numba",
            "MPLCONFIGDIR": self.cache_root / "matplotlib",
        }
        for name, path in cache_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
        environment["WANDB_DISABLED"] = "true"
        environment["PYTHONUTF8"] = "1"
        return environment


@dataclass(frozen=True, slots=True)
class SeedVCTrainingResult:
    """Completed fine-tuning run with immutable checkpoint evidence."""

    training_id: str
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    manifest: Path


class SeedVCTrainingRunner:
    """Run reviewed upstream fine-tuning against a clean-only prepared dataset."""

    def __init__(
        self,
        installation: SeedVCInstallation,
        artist_root: Path,
        artist_id: str,
    ) -> None:
        installation.validate()
        self.installation = installation
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id

    def train(
        self,
        rights: RightsManifest,
        dataset_id: str,
        *,
        max_steps: int,
        save_every: int,
        batch_size: int = 1,
        num_workers: int = 0,
        progress: Callable[[str], None] | None = None,
        select: bool = False,
    ) -> SeedVCTrainingResult:
        """Fine-tune the singing preset and optionally select its verified checkpoint."""
        require_capability(rights, self.artist_id, "training")
        require_capability(rights, self.artist_id, "singing_voice")
        if not 1 <= max_steps <= 100_000:
            raise SeedVCError("Seed-VC max_steps must be between 1 and 100000.")
        if not 1 <= batch_size <= 16:
            raise SeedVCError("Seed-VC batch size must be between 1 and 16.")
        if num_workers != 0:
            raise SeedVCError("Seed-VC uses num_workers=0 on Windows for deterministic loading.")

        dataset_root = confined_path(
            self.artist_root,
            Path("voice/training/datasets") / dataset_id,
        )
        dataset_manifest = dataset_root / "dataset.json"
        payload = self._validate_dataset(dataset_root, dataset_manifest)
        training_id = datetime.now(UTC).strftime("voice-training-%Y%m%dt%H%M%Sz-") + uuid4().hex[:8]
        runs_root = self.artist_root / "voice/training/runs"
        run_root = runs_root / training_id
        run_root.mkdir(parents=True)
        request_root = self.artist_root / "voice/training/.requests"
        request_config = request_root / f"{training_id}.yml"
        manifest_path = run_root / "training.json"
        log_path = run_root / "training.log"

        config = self._training_config(runs_root)
        dump_yaml_atomic(request_config, config)
        request_config_sha256 = sha256_file(request_config)
        base_manifest = {
            "schema_version": 1,
            "training_id": training_id,
            "artist_id": self.artist_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "training",
            "engine": "seed-vc",
            "upstream_commit": SeedVCCommit,
            "dataset_id": dataset_id,
            "dataset_manifest_sha256": sha256_file(dataset_manifest),
            "source_kind": payload["source_kind"],
            "song_count": payload["song_count"],
            "training_segment_count": payload["training_segment_count"],
            "settings": {
                "preset": SeedVCPreset,
                "preset_sha256": sha256_file(self.installation.preset),
                "resolved_config_sha256": request_config_sha256,
                "max_steps": max_steps,
                "save_every": save_every,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "device": "cuda:0",
            },
        }
        dump_json_atomic(manifest_path, base_manifest)
        command = [
            str(self.installation.python),
            "train.py",
            "--config",
            str(request_config),
            "--dataset-dir",
            str(dataset_root / "train"),
            "--run-name",
            training_id,
            "--batch-size",
            str(batch_size),
            "--max-steps",
            str(max_steps),
            "--max-epochs",
            "1000",
            "--save-every",
            str(save_every),
            "--num-workers",
            str(num_workers),
            "--gpu",
            "0",
        ]
        try:
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.installation.repository,
                    env=self.installation.environment(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if process.stdout is None:
                    raise SeedVCError("Seed-VC training output stream was not available.")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if progress is not None:
                        try:
                            progress(line.rstrip())
                        except (OSError, UnicodeError):
                            progress = None
                return_code = process.wait()
            if return_code != 0:
                detail = _tail(log_path, 8000)
                dump_json_atomic(
                    manifest_path,
                    {**base_manifest, "status": "failed", "error": detail},
                )
                raise SeedVCError(f"Seed-VC training failed: {detail}")

            checkpoint = run_root / "ft_model.pth"
            if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                raise SeedVCError("Seed-VC training did not create ft_model.pth.")
            checkpoint_sha256 = sha256_file(checkpoint)
            completed_manifest = {
                **base_manifest,
                "status": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "checkpoint": {
                    "file": checkpoint.name,
                    "sha256": checkpoint_sha256,
                    "bytes": checkpoint.stat().st_size,
                },
                "log_sha256": sha256_file(log_path),
            }
            dump_json_atomic(manifest_path, completed_manifest)
            if select:
                dump_json_atomic(
                    self.artist_root / "voice/training/selected.json",
                    {
                        "schema_version": 1,
                        "artist_id": self.artist_id,
                        "engine": "seed-vc",
                        "training_id": training_id,
                        "checkpoint_sha256": checkpoint_sha256,
                        "selected_at": datetime.now(UTC).isoformat(),
                    },
                )
        finally:
            request_config.unlink(missing_ok=True)

        return SeedVCTrainingResult(
            training_id,
            run_root,
            checkpoint,
            checkpoint_sha256,
            manifest_path,
        )

    def _validate_dataset(self, root: Path, manifest: Path) -> dict[str, object]:
        if not manifest.is_file():
            raise SeedVCError(f"Voice dataset does not exist: {root.name}")
        payload = load_json(manifest)
        if not isinstance(payload, dict) or payload.get("artist_id") != self.artist_id:
            raise SeedVCError("Voice dataset identity is invalid.")
        if payload.get("source_kind") != "clean_vocals":
            raise SeedVCError("Seed-VC training accepts only clean_vocals datasets.")
        policy = load_dataset_policy(self.artist_root, self.artist_id)
        policy_path = self.artist_root / "data/dataset-policy.yaml"
        expected_policy_sha256 = sha256_file(policy_path) if policy_path.is_file() else None
        if payload.get("dataset_policy_sha256") != expected_policy_sha256:
            raise SeedVCError(
                "Voice dataset was prepared under a stale dataset policy; rebuild it."
            )
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            raise SeedVCError("Voice dataset contains no source lineage records.")
        eligible_song_ids: set[str] = set()
        for source in sources:
            if not isinstance(source, dict):
                raise SeedVCError("Voice dataset source lineage is invalid.")
            song_id = source.get("song_id")
            source_sha256 = source.get("source_sha256")
            if not isinstance(song_id, str) or not isinstance(source_sha256, str):
                raise SeedVCError("Voice dataset source lineage is incomplete.")
            if policy.source_exclusion(source_sha256) is not None:
                raise SeedVCError("Voice dataset contains a currently excluded source.")
            if policy.is_voice_song_excluded(song_id):
                raise SeedVCError("Voice dataset contains a currently excluded voice song.")
            eligible_song_ids.add(song_id)
        segments = payload.get("segments")
        if not isinstance(segments, list) or not segments:
            raise SeedVCError("Voice dataset contains no segments.")
        training_count = 0
        for segment in segments:
            if not isinstance(segment, dict):
                raise SeedVCError("Voice dataset segment is invalid.")
            if segment.get("song_id") not in eligible_song_ids:
                raise SeedVCError("Voice dataset segment lacks eligible source lineage.")
            if not isinstance(segment, dict) or segment.get("split") != "train":
                continue
            relative = segment.get("file")
            if not isinstance(relative, str):
                raise SeedVCError("Voice dataset segment path is invalid.")
            audio = (root / relative).resolve()
            if root.resolve() not in audio.parents:
                raise SeedVCError("Voice dataset segment escaped its dataset root.")
            if not audio.is_file() or sha256_file(audio) != segment.get("sha256"):
                raise SeedVCError("Voice dataset segment failed integrity validation.")
            if not isinstance(segment.get("clean_vocals_sha256"), str):
                raise SeedVCError("Voice dataset segment lacks clean-vocal lineage.")
            training_count += 1
        if training_count != payload.get("training_segment_count"):
            raise SeedVCError("Voice dataset training count does not match its manifest.")
        return payload

    def _training_config(self, runs_root: Path) -> dict[str, object]:
        try:
            payload = yaml.safe_load(self.installation.preset.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise SeedVCError(f"Unable to load Seed-VC singing preset: {error}") from error
        if not isinstance(payload, dict):
            raise SeedVCError("Seed-VC singing preset must contain a YAML mapping.")
        payload["log_dir"] = str(runs_root.resolve())
        payload["batch_size"] = 1
        return payload


class ConvertedF0Extractor(Protocol):
    """Boundary for objective post-conversion pitch extraction."""

    def extract_f0(self, audio: Path, output: Path) -> Path:
        """Extract a model-independent F0 contour for one converted stem."""
        ...


class SeedVCVoiceEngine:
    """Use one selected artist fine-tune for deterministic singing conversion."""

    def __init__(
        self,
        installation: SeedVCInstallation,
        artist_root: Path,
        artist_id: str,
        f0_extractor: ConvertedF0Extractor,
        *,
        diffusion_steps: int = 30,
        inference_cfg_rate: float = 0.7,
        fp16: bool = True,
        timeout_seconds: float = 3600,
        quality_config: VoiceQualityConfig | None = None,
        identity_candidates: int | None = None,
        identity_threshold: float | None = None,
    ) -> None:
        installation.validate()
        if not 1 <= diffusion_steps <= 100:
            raise SeedVCError("Seed-VC diffusion steps must be between 1 and 100.")
        self.installation = installation
        self.artist_root = artist_root.resolve()
        self.artist_id = artist_id
        self.f0_extractor = f0_extractor
        self.diffusion_steps = diffusion_steps
        self.inference_cfg_rate = inference_cfg_rate
        self.fp16 = fp16
        self.timeout_seconds = timeout_seconds
        self.quality_config = quality_config
        self.identity_candidates = identity_candidates
        self.identity_threshold = identity_threshold
        if identity_candidates is not None and not 2 <= identity_candidates <= 5:
            raise SeedVCError("Identity candidate count must be between 2 and 5.")
        if identity_threshold is not None and not -1 <= identity_threshold <= 1:
            raise SeedVCError("Identity threshold must be between -1 and 1.")
        (
            self.training_id,
            self.checkpoint,
            self.checkpoint_sha256,
            self.config,
            self.config_sha256,
        ) = self._resolve_selected()

    def convert(
        self,
        request: VoiceConversionRequest,
        output: Path,
    ) -> VoiceConversionResult:
        """Convert a complete guide vocal with the selected artist fine-tune."""
        if request.engine != "seed-vc":
            raise SeedVCError("Seed-VC received a conversion request for another engine.")
        probe_audio(request.target_audio)
        if not request.target_f0.is_file() or request.target_f0.stat().st_size == 0:
            raise SeedVCError("Seed-VC target F0 is missing.")
        references = tuple(sorted(request.references, key=lambda item: item.reference.id))
        for reference in references:
            if reference.reference.review_status is not VoiceReviewStatus.APPROVED:
                raise SeedVCError(f"Voice reference '{reference.reference.id}' is not approved.")
            for path, digest in (
                (reference.source, reference.reference.source_sha256),
                (reference.audio, reference.reference.audio_sha256),
                (reference.f0, reference.reference.f0_sha256),
            ):
                if not path.is_file() or sha256_file(path) != digest:
                    raise SeedVCError(
                        f"Voice reference artifact failed integrity validation: {path.name}"
                    )
        if self.quality_config is not None:
            return self._convert_identity_gated(request, output, references)
        reference = next(
            (
                item
                for item in references
                if "neutral" in item.reference.tags and "clean-vocal-stem" in item.reference.tags
            ),
            references[0],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        runtime_root = output.parent / "seed-vc-runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        log_path = runtime_root / "inference.log"
        command = [
            str(self.installation.python),
            str(self.installation.runtime_script),
            "--source",
            str(request.target_audio.resolve()),
            "--target",
            str(reference.audio.resolve()),
            "--output",
            str(runtime_root.resolve()),
            "--checkpoint",
            str(self.checkpoint.resolve()),
            "--config",
            str(self.config.resolve()),
            "--seed",
            str(request.seed),
            "--diffusion-steps",
            str(self.diffusion_steps),
            "--inference-cfg-rate",
            str(self.inference_cfg_rate),
        ]
        if self.fp16:
            command.append("--fp16")
        completed = subprocess.run(
            command,
            cwd=self.installation.repository,
            env=self.installation.environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=self.timeout_seconds,
        )
        log_path.write_text(
            completed.stdout + ("\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        generated = sorted(runtime_root.glob("vc_*.wav"))
        if completed.returncode != 0 or len(generated) != 1:
            detail = (completed.stderr or completed.stdout)[-8000:]
            raise SeedVCError(f"Seed-VC conversion failed: {detail}")
        raw_peak_dbfs, peak_reduction_db = _align_and_peak_protect(
            generated[0],
            request.target_audio,
            output,
        )

        cleaned_f0 = output.with_name(f"{output.stem}-cleaned-f0.npy")
        shutil.copyfile(request.target_f0, cleaned_f0)
        converted_f0 = output.with_name(f"{output.stem}-f0.npy")
        self.f0_extractor.extract_f0(output, converted_f0)
        target_f0 = np.load(cleaned_f0, allow_pickle=False).astype(np.float32).reshape(-1)
        generated_f0 = np.load(converted_f0, allow_pickle=False).astype(np.float32).reshape(-1)
        median_error, gross_error = compare_f0(target_f0, generated_f0)
        audio, sample_rate = sf.read(output, dtype="float32", always_2d=True)
        final_peak_dbfs = _amplitude_db(float(np.max(np.abs(audio))))
        dropout_count, dropout_fraction = _count_sustained_dropouts(
            np.mean(audio, axis=1, dtype=np.float32),
            sample_rate,
            target_f0,
        )
        duration = len(audio) / sample_rate
        candidate = VoiceCandidateMetrics(
            candidate_index=0,
            seed=request.seed,
            passed=(dropout_count == 0 and gross_error < 0.05),
            raw_peak_dbfs=raw_peak_dbfs,
            peak_reduction_db=peak_reduction_db,
            sustained_dropout_count=dropout_count,
            dropout_fraction=dropout_fraction,
            spectral_distance_db=0,
            f0_median_error_cents=median_error,
            f0_gross_error_fraction=gross_error,
        )
        phrase = VoicePhraseDecision(
            index=0,
            start_seconds=0,
            end_seconds=duration,
            reference_id=reference.reference.id,
            selected_candidate_index=0,
            candidates=(candidate,),
        )
        metrics = VoiceConversionMetrics(
            phrase_count=1,
            selected_reference_ids=(reference.reference.id,),
            sustained_dropout_count=dropout_count,
            peak_dbfs=final_peak_dbfs,
            clipping_fraction=float(np.mean(np.abs(audio) >= 0.999)),
            f0_median_error_cents=median_error,
            f0_gross_error_fraction=gross_error,
        )
        manifest_path = output.parent / "voice-conversion.json"
        manifest = VoiceConversionManifest(
            engine="seed-vc",
            mode="artist-finetuned-clean-reference",
            created_at=datetime.now(UTC),
            target_audio_sha256=sha256_file(request.target_audio),
            target_f0_sha256=sha256_file(request.target_f0),
            cleaned_f0_sha256=sha256_file(cleaned_f0),
            reference_audio_hashes={reference.reference.id: reference.reference.audio_sha256},
            reference_f0_hashes={reference.reference.id: reference.reference.f0_sha256},
            reference_source_hashes={reference.reference.id: reference.reference.source_sha256},
            reference_parent_source_hashes={
                reference.reference.id: reference.reference.parent_source_sha256
                or reference.reference.source_sha256
            },
            reference_stem_hashes=(
                {reference.reference.id: reference.reference.source_stem_sha256}
                if reference.reference.source_stem_sha256 is not None
                else {}
            ),
            settings={
                "training_id": self.training_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "config_sha256": self.config_sha256,
                "diffusion_steps": self.diffusion_steps,
                "inference_cfg_rate": self.inference_cfg_rate,
                "fp16": self.fp16,
                "f0_condition": True,
                "auto_f0_adjust": False,
                "semi_tone_shift": 0,
            },
            phrases=(phrase,),
            metrics=metrics,
            output_sha256=sha256_file(output),
        )
        dump_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        return VoiceConversionResult(
            audio=output,
            manifest=manifest_path,
            cleaned_f0=cleaned_f0,
            reference_ids=(reference.reference.id,),
            phrases=(phrase,),
            metrics=metrics,
        )

    def _convert_identity_gated(
        self,
        request: VoiceConversionRequest,
        output: Path,
        references: tuple[ResolvedVoiceReference, ...],
    ) -> VoiceConversionResult:
        """Convert register-matched phrases and retain only identity-passing candidates."""
        quality = self.quality_config
        if quality is None:
            raise SeedVCError("Identity-gated conversion requires voice quality settings.")
        output.parent.mkdir(parents=True, exist_ok=True)
        cleaned_f0 = output.with_name(f"{output.stem}-cleaned-f0.npy")
        write_clean_f0(request.target_f0, cleaned_f0, quality.f0_gap_fill_ms)
        target_f0 = np.load(cleaned_f0, allow_pickle=False).astype(np.float32).reshape(-1)
        phrase_plans = plan_phrases(
            target_f0,
            quality.identity_phrase_min_seconds,
            quality.identity_phrase_max_seconds,
            quality.identity_phrase_overlap_seconds,
        )
        try:
            target_audio, sample_rate = sf.read(
                request.target_audio,
                dtype="float32",
                always_2d=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise SeedVCError(f"Unable to read Seed-VC target audio: {error}") from error
        if target_audio.size == 0 or sample_rate <= 0:
            raise SeedVCError("Seed-VC target audio contains no samples.")
        target_mono = np.mean(target_audio, axis=1, dtype=np.float32)

        runtime_root = output.parent / "seed-vc-identity"
        targets_root = runtime_root / "targets"
        targets_root.mkdir(parents=True, exist_ok=False)
        prepared_phrases: list[dict[str, object]] = []
        for phrase in phrase_plans:
            start_sample = round(phrase.start_seconds * sample_rate)
            end_sample = min(len(target_mono), round(phrase.end_seconds * sample_rate))
            if end_sample <= start_sample:
                raise SeedVCError(f"Seed-VC phrase {phrase.index} has no audio samples.")
            target_path = targets_root / f"phrase-{phrase.index:03d}.wav"
            sf.write(
                target_path,
                target_mono[start_sample:end_sample],
                sample_rate,
                subtype="PCM_24",
            )
            ranked_references = rank_references(
                references,
                request.target_audio,
                target_f0,
                phrase,
            )
            reference = ranked_references[0]
            phrase_f0 = target_f0[phrase.start_frame : phrase.end_frame]
            phrase_f0_path = targets_root / f"phrase-{phrase.index:03d}-f0.npy"
            np.save(phrase_f0_path, phrase_f0)
            prepared_phrases.append(
                {
                    "plan": phrase,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "target": target_path,
                    "target_f0": phrase_f0_path,
                    "reference": reference,
                    "references": ranked_references,
                    "is_voiced": bool(np.any(phrase_f0 > 0)),
                }
            )
        voiced_phrases = [item for item in prepared_phrases if item["is_voiced"]]
        if not voiced_phrases:
            raise SeedVCError("Identity-gated conversion found no voiced target phrases.")

        runtime_result = runtime_root / "result.json"
        runtime_log = runtime_root / "inference.log"
        request_path = output.parent / f".seed-vc-identity.{uuid4().hex}.json"
        candidate_count = self.identity_candidates or quality.identity_candidates_per_phrase
        settings = {
            "phrase_min_seconds": quality.identity_phrase_min_seconds,
            "phrase_max_seconds": quality.identity_phrase_max_seconds,
            "phrase_overlap_seconds": quality.identity_phrase_overlap_seconds,
            "candidates_per_phrase": candidate_count,
            "max_candidates_per_phrase": max(
                candidate_count,
                quality.identity_max_candidates_per_phrase,
            ),
            "f0_gap_fill_ms": quality.f0_gap_fill_ms,
            "f0_median_error_limit_cents": quality.f0_median_error_limit_cents,
            "f0_gross_error_limit_fraction": quality.f0_gross_error_limit_fraction,
            "audio_dropout_repair_ms": quality.audio_dropout_repair_ms,
            "audio_dropout_max_gain_db": quality.audio_dropout_max_gain_db,
            "diffusion_steps": self.diffusion_steps,
            "inference_cfg_rate": self.inference_cfg_rate,
            "fp16": self.fp16,
            "f0_condition": True,
            "auto_f0_adjust": False,
            "semi_tone_shift": 0,
            "identity_threshold": self.identity_threshold,
            "identity_similarity_margin": quality.identity_similarity_margin,
            "identity_minimum_similarity": quality.identity_minimum_similarity,
            "identity_window_seconds": quality.identity_window_seconds,
            "identity_window_hop_seconds": quality.identity_window_hop_seconds,
            "identity_window_min_voiced_fraction": (quality.identity_window_min_voiced_fraction),
            "identity_window_similarity_margin": (quality.identity_window_similarity_margin),
        }
        dump_json_atomic(
            request_path,
            {
                "schema_version": 1,
                "runtime_root": str(runtime_root.resolve()),
                "result": str(runtime_result.resolve()),
                "seed": request.seed,
                "settings": settings,
                "phrases": [
                    {
                        "index": item["plan"].index,
                        "source": str(item["target"].resolve()),
                        "target_f0": str(item["target_f0"].resolve()),
                        "reference_id": item["reference"].reference.id,
                        "reference_ids": [
                            reference.reference.id
                            for reference in item["references"][:candidate_count]
                        ],
                    }
                    for item in voiced_phrases
                ],
                "references": [
                    {
                        "id": item.reference.id,
                        "audio": str(item.audio.resolve()),
                    }
                    for item in references
                ],
            },
        )
        command = [
            str(self.installation.python),
            str(self.installation.runtime_script),
            "--batch-request",
            str(request_path.resolve()),
            "--checkpoint",
            str(self.checkpoint.resolve()),
            "--config",
            str(self.config.resolve()),
            "--seed",
            str(request.seed),
            "--diffusion-steps",
            str(self.diffusion_steps),
            "--inference-cfg-rate",
            str(self.inference_cfg_rate),
        ]
        if self.fp16:
            command.append("--fp16")
        try:
            completed = subprocess.run(
                command,
                cwd=self.installation.repository,
                env=self.installation.environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self.timeout_seconds,
            )
        finally:
            request_path.unlink(missing_ok=True)
        runtime_log.write_text(
            completed.stdout + ("\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0 or not runtime_result.is_file():
            detail = (completed.stderr or completed.stdout)[-8000:]
            raise SeedVCError(f"Seed-VC identity conversion failed: {detail}")

        payload = load_json(runtime_result)
        records = payload.get("candidates") if isinstance(payload, dict) else None
        threshold_value = payload.get("identity_threshold") if isinstance(payload, dict) else None
        window_threshold_value = (
            payload.get("identity_window_threshold") if isinstance(payload, dict) else None
        )
        calibration = (
            payload.get("reference_calibration_similarities") if isinstance(payload, dict) else None
        )
        if (
            not isinstance(records, list)
            or not isinstance(threshold_value, (int, float))
            or not isinstance(window_threshold_value, (int, float))
        ):
            raise SeedVCError("Seed-VC identity runtime returned an invalid result.")
        identity_threshold = float(threshold_value)
        identity_window_threshold = float(window_threshold_value)
        if not -1 <= identity_threshold <= 1:
            raise SeedVCError("Seed-VC identity threshold is outside the valid range.")
        if not -1 <= identity_window_threshold <= 1:
            raise SeedVCError("Seed-VC window identity threshold is outside the valid range.")
        candidates_by_phrase: dict[int, list[dict[str, object]]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("phrase_index"), int):
                raise SeedVCError("Seed-VC identity candidate metadata is invalid.")
            candidates_by_phrase.setdefault(record["phrase_index"], []).append(record)

        selected_audio: list[tuple[int, int, np.ndarray]] = []
        decisions: list[VoicePhraseDecision] = []
        selected_similarities: list[float] = []
        selected_window_similarities: list[float] = []
        selected_window_means: list[float] = []
        selected_dropout_count = 0
        peak_limit = 10 ** (-1 / 20)
        references_by_id = {item.reference.id: item for item in references}
        reference_audio_cache: dict[str, tuple[np.ndarray, int]] = {}
        for item in prepared_phrases:
            phrase = item["plan"]
            start_sample = int(item["start_sample"])
            end_sample = int(item["end_sample"])
            reference = item["reference"]
            phrase_f0 = target_f0[phrase.start_frame : phrase.end_frame]
            if not item["is_voiced"]:
                bypass = target_mono[start_sample:end_sample].copy()
                peak = float(np.max(np.abs(bypass))) if bypass.size else 0.0
                metric = VoiceCandidateMetrics(
                    candidate_index=0,
                    seed=request.seed,
                    passed=True,
                    raw_peak_dbfs=_amplitude_db(peak),
                    peak_reduction_db=0,
                    sustained_dropout_count=0,
                    dropout_fraction=0,
                    spectral_distance_db=0,
                )
                selected_audio.append((start_sample, end_sample, bypass))
                decisions.append(
                    VoicePhraseDecision(
                        index=phrase.index,
                        start_seconds=phrase.start_seconds,
                        end_seconds=phrase.end_seconds,
                        reference_id=reference.reference.id,
                        selected_candidate_index=0,
                        candidates=(metric,),
                    )
                )
                continue

            phrase_records = sorted(
                candidates_by_phrase.get(phrase.index, []),
                key=lambda record: int(record.get("candidate_index", -1)),
            )
            maximum_candidate_count = int(settings["max_candidates_per_phrase"])
            if not candidate_count <= len(phrase_records) <= maximum_candidate_count:
                raise SeedVCError(
                    f"Seed-VC phrase {phrase.index} returned an incomplete candidate set."
                )
            candidate_metrics: list[VoiceCandidateMetrics] = []
            candidate_audio: list[np.ndarray] = []
            for record in phrase_records:
                candidate_path = _confined_runtime_path(runtime_root, record.get("audio"))
                candidate_reference_id = record.get("reference_id")
                if (
                    not isinstance(candidate_reference_id, str)
                    or candidate_reference_id not in references_by_id
                ):
                    raise SeedVCError("Seed-VC candidate reference identity is invalid.")
                candidate_reference = references_by_id[candidate_reference_id]
                if candidate_reference_id not in reference_audio_cache:
                    reference_audio, reference_rate = sf.read(
                        candidate_reference.audio,
                        dtype="float32",
                        always_2d=True,
                    )
                    reference_audio_cache[candidate_reference_id] = (
                        np.mean(reference_audio, axis=1, dtype=np.float32),
                        reference_rate,
                    )
                reference_mono, reference_rate = reference_audio_cache[candidate_reference_id]
                values, generated_rate = sf.read(
                    candidate_path,
                    dtype="float32",
                    always_2d=True,
                )
                if generated_rate != sample_rate:
                    raise SeedVCError("Seed-VC phrase conversion changed the target sample rate.")
                mono = _fit_length(
                    np.mean(values, axis=1, dtype=np.float32),
                    end_sample - start_sample,
                )
                mono, local_repaired_dropout_count = _repair_short_dropouts(
                    mono,
                    sample_rate,
                    phrase_f0,
                    quality.audio_dropout_repair_ms,
                    quality.audio_dropout_max_gain_db,
                )
                raw_peak = float(np.max(np.abs(mono))) if mono.size else 0.0
                raw_peak_dbfs = _amplitude_db(raw_peak)
                peak_reduction_db = 0.0
                if raw_peak > peak_limit:
                    peak_reduction_db = raw_peak_dbfs - _amplitude_db(peak_limit)
                    mono *= peak_limit / raw_peak
                dropout_count, dropout_fraction = _count_sustained_dropouts(
                    mono,
                    sample_rate,
                    phrase_f0,
                )
                spectral_distance = abs(
                    _high_band_ratio(mono, sample_rate)
                    - _high_band_ratio(reference_mono, reference_rate)
                )
                similarity_value = record.get("identity_similarity")
                if not isinstance(similarity_value, (int, float)):
                    raise SeedVCError("Seed-VC candidate identity score is invalid.")
                similarity = float(similarity_value)
                reference_similarity_value = record.get("reference_identity_similarity")
                minimum_window_value = record.get("minimum_window_identity_similarity")
                mean_window_value = record.get("mean_window_identity_similarity")
                window_count_value = record.get("identity_window_count")
                if (
                    not isinstance(reference_similarity_value, (int, float))
                    or not isinstance(minimum_window_value, (int, float))
                    or not isinstance(mean_window_value, (int, float))
                    or not isinstance(window_count_value, int)
                    or window_count_value < 1
                ):
                    raise SeedVCError("Seed-VC candidate window identity scores are invalid.")
                reference_similarity = float(reference_similarity_value)
                minimum_window_similarity = float(minimum_window_value)
                mean_window_similarity = float(mean_window_value)
                median_pitch_value = record.get("f0_median_error_cents")
                gross_pitch_value = record.get("f0_gross_error_fraction")
                if not isinstance(median_pitch_value, (int, float)) or not isinstance(
                    gross_pitch_value, (int, float)
                ):
                    raise SeedVCError("Seed-VC candidate pitch scores are invalid.")
                median_pitch_error = float(median_pitch_value)
                gross_pitch_error = float(gross_pitch_value)
                runtime_repaired_dropout_count = record.get("repaired_dropout_count", 0)
                if (
                    not isinstance(runtime_repaired_dropout_count, int)
                    or runtime_repaired_dropout_count < 0
                ):
                    raise SeedVCError("Seed-VC repaired-dropout metadata is invalid.")
                passed = (
                    np.isfinite(similarity)
                    and np.isfinite(minimum_window_similarity)
                    and minimum_window_similarity >= identity_window_threshold
                    and dropout_count == 0
                    and 0 < raw_peak <= 4.0
                    and median_pitch_error <= quality.f0_median_error_limit_cents
                    and gross_pitch_error < quality.f0_gross_error_limit_fraction
                )
                candidate_metrics.append(
                    VoiceCandidateMetrics(
                        candidate_index=int(record["candidate_index"]),
                        reference_id=candidate_reference_id,
                        seed=int(record["seed"]),
                        passed=passed,
                        raw_peak_dbfs=raw_peak_dbfs,
                        peak_reduction_db=peak_reduction_db,
                        sustained_dropout_count=dropout_count,
                        repaired_dropout_count=(
                            runtime_repaired_dropout_count + local_repaired_dropout_count
                        ),
                        dropout_fraction=dropout_fraction,
                        spectral_distance_db=spectral_distance,
                        identity_similarity=similarity,
                        reference_identity_similarity=reference_similarity,
                        identity_threshold=identity_threshold,
                        minimum_window_identity_similarity=minimum_window_similarity,
                        mean_window_identity_similarity=mean_window_similarity,
                        identity_window_threshold=identity_window_threshold,
                        identity_window_count=window_count_value,
                        f0_median_error_cents=median_pitch_error,
                        f0_gross_error_fraction=gross_pitch_error,
                    )
                )
                candidate_audio.append(mono)
            selected_position = _select_identity_candidate(tuple(candidate_metrics))
            selected = candidate_metrics[selected_position]
            if not selected.passed:
                best_similarity = float(
                    selected.identity_similarity if selected.identity_similarity is not None else -1
                )
                raise SeedVCError(
                    f"Every candidate for phrase {phrase.index} failed identity or pitch QC; "
                    f"best whole-phrase identity was {best_similarity:.4f}."
                )
            selected_audio.append((start_sample, end_sample, candidate_audio[selected_position]))
            selected_dropout_count += selected.sustained_dropout_count
            selected_similarities.append(
                float(
                    selected.identity_similarity if selected.identity_similarity is not None else -1
                )
            )
            selected_window_similarities.append(
                float(
                    selected.minimum_window_identity_similarity
                    if selected.minimum_window_identity_similarity is not None
                    else -1
                )
            )
            selected_window_means.append(
                float(
                    selected.mean_window_identity_similarity
                    if selected.mean_window_identity_similarity is not None
                    else -1
                )
            )
            decisions.append(
                VoicePhraseDecision(
                    index=phrase.index,
                    start_seconds=phrase.start_seconds,
                    end_seconds=phrase.end_seconds,
                    reference_id=(selected.reference_id or reference.reference.id),
                    selected_candidate_index=selected.candidate_index,
                    candidates=tuple(candidate_metrics),
                )
            )

        joined, join_gains = _join_phrases(selected_audio, len(target_mono))
        decisions = [
            decision.model_copy(update={"join_gain_db": join_gain})
            for decision, join_gain in zip(decisions, join_gains, strict=True)
        ]
        output_peak = float(np.max(np.abs(joined))) if joined.size else 0.0
        if output_peak > peak_limit:
            joined *= peak_limit / output_peak
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
        sf.write(temporary, joined, sample_rate, subtype="PCM_24")
        os.replace(temporary, output)
        probe_audio(output)

        converted_f0 = output.with_name(f"{output.stem}-f0.npy")
        self.f0_extractor.extract_f0(output, converted_f0)
        converted_qc_f0 = output.with_name(f"{output.stem}-qc-f0.npy")
        write_clean_f0(converted_f0, converted_qc_f0, quality.f0_gap_fill_ms)
        generated_f0 = np.load(converted_qc_f0, allow_pickle=False).astype(np.float32).reshape(-1)
        median_error, gross_error = compare_f0(target_f0, generated_f0)
        dropout_count, _dropout_fraction = _count_sustained_dropouts(
            joined,
            sample_rate,
            target_f0,
        )
        join_samples = [item[0] for item in selected_audio[1:]]
        selected_reference_ids = tuple(sorted({decision.reference_id for decision in decisions}))
        metrics = VoiceConversionMetrics(
            phrase_count=len(decisions),
            selected_reference_ids=selected_reference_ids,
            sustained_dropout_count=max(dropout_count, selected_dropout_count),
            peak_dbfs=_amplitude_db(float(np.max(np.abs(joined)))),
            clipping_fraction=float(np.mean(np.abs(joined) >= 0.999)),
            max_voiced_join_jump_db=_max_join_jump_db(
                joined,
                join_samples,
                target_f0,
                sample_rate,
            ),
            minimum_identity_similarity=min(selected_similarities),
            mean_identity_similarity=float(np.mean(selected_similarities)),
            identity_threshold=identity_threshold,
            minimum_window_identity_similarity=min(selected_window_similarities),
            mean_window_identity_similarity=float(np.mean(selected_window_means)),
            identity_window_threshold=identity_window_threshold,
            f0_median_error_cents=median_error,
            f0_gross_error_fraction=gross_error,
        )
        manifest_path = output.parent / "voice-conversion.json"
        manifest = VoiceConversionManifest(
            engine="seed-vc",
            mode="identity-gated-multi-reference",
            created_at=datetime.now(UTC),
            target_audio_sha256=sha256_file(request.target_audio),
            target_f0_sha256=sha256_file(request.target_f0),
            cleaned_f0_sha256=sha256_file(cleaned_f0),
            reference_audio_hashes={
                item.reference.id: item.reference.audio_sha256 for item in references
            },
            reference_f0_hashes={
                item.reference.id: item.reference.f0_sha256 for item in references
            },
            reference_source_hashes={
                item.reference.id: item.reference.source_sha256 for item in references
            },
            reference_parent_source_hashes={
                item.reference.id: item.reference.parent_source_sha256
                or item.reference.source_sha256
                for item in references
            },
            reference_stem_hashes={
                item.reference.id: item.reference.source_stem_sha256
                for item in references
                if item.reference.source_stem_sha256 is not None
            },
            settings={
                **settings,
                "training_id": self.training_id,
                "checkpoint_sha256": self.checkpoint_sha256,
                "config_sha256": self.config_sha256,
                "reference_calibration_similarities": calibration,
                "reference_window_calibration": payload.get("reference_window_calibration"),
                "converted_f0": converted_f0.name,
                "converted_f0_sha256": sha256_file(converted_f0),
                "converted_qc_f0": converted_qc_f0.name,
                "converted_qc_f0_sha256": sha256_file(converted_qc_f0),
            },
            phrases=tuple(decisions),
            metrics=metrics,
            output_sha256=sha256_file(output),
        )
        dump_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        return VoiceConversionResult(
            audio=output,
            manifest=manifest_path,
            cleaned_f0=cleaned_f0,
            reference_ids=selected_reference_ids,
            phrases=tuple(decisions),
            metrics=metrics,
        )

    def _resolve_selected(self) -> tuple[str, Path, str, Path, str]:
        selected_path = self.artist_root / "voice/training/selected.json"
        selected = load_json(selected_path)
        if not isinstance(selected, dict) or selected.get("artist_id") != self.artist_id:
            raise SeedVCError("Selected voice training identity is invalid.")
        training_id = selected.get("training_id")
        if not isinstance(training_id, str):
            raise SeedVCError("No Seed-VC training checkpoint is selected.")
        run_root = confined_path(
            self.artist_root,
            Path("voice/training/runs") / training_id,
        )
        manifest = load_json(run_root / "training.json")
        if not isinstance(manifest, dict) or manifest.get("status") != "complete":
            raise SeedVCError("Selected Seed-VC training is not complete.")
        checkpoint = run_root / "ft_model.pth"
        checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint_sha256 != selected.get(
            "checkpoint_sha256"
        ) or checkpoint_sha256 != manifest.get("checkpoint", {}).get("sha256"):
            raise SeedVCError("Selected Seed-VC checkpoint failed integrity validation.")
        configs = tuple(run_root.glob("*.yml"))
        if len(configs) != 1:
            raise SeedVCError("Selected Seed-VC training must retain exactly one config.")
        config_sha256 = sha256_file(configs[0])
        if config_sha256 != manifest.get("settings", {}).get("resolved_config_sha256"):
            raise SeedVCError("Selected Seed-VC config failed integrity validation.")
        return training_id, checkpoint, checkpoint_sha256, configs[0], config_sha256


def _confined_runtime_path(root: Path, relative: object) -> Path:
    """Resolve one generated candidate without allowing runtime-result path escape."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise SeedVCError("Seed-VC candidate path is invalid.")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if resolved_root not in candidate.parents or not candidate.is_file():
        raise SeedVCError("Seed-VC candidate escaped or is missing from its runtime root.")
    return candidate


def _select_identity_candidate(candidates: tuple[VoiceCandidateMetrics, ...]) -> int:
    """Prefer the strongest worst-window identity among candidates that pass every gate."""
    if not candidates:
        raise SeedVCError("Seed-VC produced no identity candidates.")
    return min(
        range(len(candidates)),
        key=lambda index: (
            not candidates[index].passed,
            -float(
                candidates[index].minimum_window_identity_similarity
                if candidates[index].minimum_window_identity_similarity is not None
                else candidates[index].identity_similarity
                if candidates[index].identity_similarity is not None
                else -1
            ),
            -float(
                candidates[index].mean_window_identity_similarity
                if candidates[index].mean_window_identity_similarity is not None
                else -1
            ),
            -float(
                candidates[index].identity_similarity
                if candidates[index].identity_similarity is not None
                else -1
            ),
            -float(
                candidates[index].reference_identity_similarity
                if candidates[index].reference_identity_similarity is not None
                else -1
            ),
            candidates[index].sustained_dropout_count,
            candidates[index].dropout_fraction,
            candidates[index].spectral_distance_db,
            candidates[index].peak_reduction_db,
            candidates[index].candidate_index,
        ),
    )


def _fit_length(values: np.ndarray, expected_samples: int) -> np.ndarray:
    """Trim or pad one phrase candidate to exact sample coverage."""
    audio = np.asarray(values, dtype=np.float32).reshape(-1)
    if expected_samples <= 0:
        raise SeedVCError("Seed-VC phrase length must be positive.")
    if len(audio) >= expected_samples:
        return audio[:expected_samples].copy()
    return np.pad(audio, (0, expected_samples - len(audio))).astype(np.float32)


def _join_phrases(
    phrases: list[tuple[int, int, np.ndarray]],
    total_samples: int,
) -> tuple[np.ndarray, list[float]]:
    """Join complete overlapping phrase coverage with gain-matched equal-power fades."""
    if not phrases or total_samples <= 0:
        raise SeedVCError("Seed-VC produced no phrase audio to join.")
    output = np.zeros(total_samples, dtype=np.float32)
    written_until = 0
    join_gains: list[float] = []
    for start, end, raw_values in sorted(phrases, key=lambda item: item[0]):
        if start < 0 or end <= start or start > written_until:
            raise SeedVCError("Seed-VC phrase coverage contains a gap or invalid range.")
        end = min(end, total_samples)
        values = _fit_length(raw_values, end - start)
        overlap_end = min(written_until, end)
        join_gain_db = 0.0
        if overlap_end > start:
            overlap = overlap_end - start
            previous_rms = float(
                np.sqrt(np.mean(np.square(output[start:overlap_end], dtype=np.float64)))
            )
            current_rms = float(np.sqrt(np.mean(np.square(values[:overlap], dtype=np.float64))))
            if previous_rms > 1e-8 and current_rms > 1e-8:
                join_gain_db = float(np.clip(20 * math.log10(previous_rms / current_rms), -6, 6))
                values *= 10 ** (join_gain_db / 20)
            phase = np.linspace(0, math.pi / 2, overlap, endpoint=True)
            output[start:overlap_end] = output[start:overlap_end] * np.cos(phase) + values[
                :overlap
            ] * np.sin(phase)
            output[overlap_end:end] = values[overlap : overlap + end - overlap_end]
        else:
            output[start:end] = values[: end - start]
        join_gains.append(join_gain_db)
        written_until = max(written_until, end)
    if written_until != total_samples:
        raise SeedVCError("Seed-VC phrase coverage did not reach the target duration.")
    return output, join_gains


def _max_join_jump_db(
    audio: np.ndarray,
    join_samples: list[int],
    target_f0: np.ndarray,
    sample_rate: int,
) -> float:
    """Measure level discontinuities only where target voicing crosses a phrase join."""
    window = max(1, round(0.005 * sample_rate))
    jumps = []
    for join in join_samples:
        f0_index = min(len(target_f0) - 1, round(join * 50 / sample_rate))
        if (
            f0_index == 0
            or target_f0[f0_index - 1] <= 0
            or target_f0[f0_index] <= 0
            or join < window
            or join + window > len(audio)
        ):
            continue
        before = float(np.sqrt(np.mean(np.square(audio[join - window : join], dtype=np.float64))))
        after = float(np.sqrt(np.mean(np.square(audio[join : join + window], dtype=np.float64))))
        jumps.append(abs(20 * math.log10(max(after, 1e-12) / max(before, 1e-12))))
    return max(jumps, default=0.0)


def _high_band_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Return high-frequency energy relative to the vocal-body band."""
    usable = np.asarray(audio, dtype=np.float32).reshape(-1)[: sample_rate * 3]
    if usable.size < 2:
        return -120.0
    spectrum = np.square(np.abs(np.fft.rfft(usable * np.hanning(len(usable)))))
    frequencies = np.fft.rfftfreq(len(usable), 1 / sample_rate)
    high = float(np.sum(spectrum[frequencies >= 5000]))
    body = float(np.sum(spectrum[(frequencies >= 100) & (frequencies < 5000)]))
    return 10 * math.log10(max(high, 1e-20) / max(body, 1e-20))


def _tail(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "Training log could not be read."
    return text[-limit:]


def _align_and_peak_protect(
    generated: Path,
    target: Path,
    output: Path,
) -> tuple[float, float]:
    try:
        values, generated_rate = sf.read(generated, dtype="float32", always_2d=True)
        target_info = sf.info(target)
    except (OSError, RuntimeError, ValueError) as error:
        raise SeedVCError(f"Unable to align Seed-VC output: {error}") from error
    if generated_rate != target_info.samplerate:
        raise SeedVCError("Seed-VC output changed the target sample rate.")
    target_frames = int(target_info.frames)
    if len(values) < target_frames:
        values = np.pad(values, ((0, target_frames - len(values)), (0, 0)))
    values = values[:target_frames]
    raw_peak = float(np.max(np.abs(values)))
    raw_peak_dbfs = _amplitude_db(raw_peak)
    peak_limit = 10 ** (-1 / 20)
    peak_reduction_db = 0.0
    if raw_peak > peak_limit:
        peak_reduction_db = raw_peak_dbfs - _amplitude_db(peak_limit)
        values *= peak_limit / raw_peak
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
    sf.write(temporary, values, generated_rate, subtype="PCM_24")
    os.replace(temporary, output)
    probe_audio(output)
    return raw_peak_dbfs, peak_reduction_db


def _count_sustained_dropouts(
    audio: np.ndarray,
    sample_rate: int,
    f0: np.ndarray,
) -> tuple[int, float]:
    frame_count = min(len(f0), round(len(audio) * 50 / sample_rate))
    if frame_count <= 0:
        raise SeedVCError("Converted vocal has no aligned F0 frames.")
    boundaries = np.rint(np.linspace(0, len(audio), frame_count + 1)).astype(np.int64)
    rms = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = audio[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            rms[index] = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    voiced = f0[:frame_count] > 0
    if not np.any(voiced):
        raise SeedVCError("Converted vocal target contains no voiced frames.")
    threshold = max(float(np.median(rms[voiced])) * 0.03, 1e-6)
    dropout = voiced & (rms < threshold)
    count = 0
    index = 0
    while index < frame_count:
        if not dropout[index]:
            index += 1
            continue
        start = index
        while index < frame_count and dropout[index]:
            index += 1
        if index - start >= 6:
            count += 1
    return count, float(np.mean(dropout[voiced]))


def _repair_short_dropouts(
    audio: np.ndarray,
    sample_rate: int,
    f0: np.ndarray,
    maximum_ms: int,
    maximum_gain_db: float,
) -> tuple[np.ndarray, int]:
    """Restore brief low-level holes with a capped envelope while preserving longer failures."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if maximum_ms <= 0 or maximum_gain_db <= 0:
        return values, 0
    frame_count = min(len(f0), round(len(values) * 50 / sample_rate))
    if frame_count <= 0:
        raise SeedVCError("Converted vocal has no aligned F0 frames.")
    boundaries = np.rint(np.linspace(0, len(values), frame_count + 1)).astype(np.int64)
    rms = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = values[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            rms[index] = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    voiced = np.asarray(f0).reshape(-1)[:frame_count] > 0
    if not np.any(voiced):
        return values, 0
    threshold = max(float(np.median(rms[voiced])) * 0.03, 1e-6)
    dropout = voiced & (rms < threshold)
    maximum_frames = max(1, round(maximum_ms * 50 / 1000))
    maximum_gain = 10 ** (maximum_gain_db / 20)
    repaired_count = 0
    index = 0
    while index < frame_count:
        if not dropout[index]:
            index += 1
            continue
        start = index
        while index < frame_count and dropout[index]:
            index += 1
        length = index - start
        if length < 6 or length > maximum_frames:
            continue
        start_sample = int(boundaries[start])
        end_sample = int(boundaries[index])
        segment = values[start_sample:end_sample]
        segment_rms = float(np.sqrt(np.mean(np.square(segment), dtype=np.float64)))
        if segment_rms <= 1e-8:
            continue
        gain = min(maximum_gain, threshold * 1.05 / segment_rms)
        if gain <= 1:
            continue
        values[start_sample:end_sample] *= gain
        fade_samples = min(round(0.02 * sample_rate), start_sample, len(values) - end_sample)
        if fade_samples > 0:
            phase = np.linspace(0, np.pi / 2, fade_samples, endpoint=False)
            fade_in = 1 + (gain - 1) * np.square(np.sin(phase))
            fade_out = fade_in[::-1]
            values[start_sample - fade_samples : start_sample] *= fade_in.astype(np.float32)
            values[end_sample : end_sample + fade_samples] *= fade_out.astype(np.float32)
        repaired_count += 1
    return values, repaired_count


def _amplitude_db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def _find_seed_vc_python() -> Path:
    override = os.environ.get("SEED_VC_PYTHON")
    if override:
        return Path(override)
    candidates = [
        Path.home() / ".conda/envs/seed-vc/python.exe",
        Path("C:/ProgramData/anaconda3/envs/seed-vc/python.exe"),
        Path("C:/ProgramData/miniconda3/envs/seed-vc/python.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    executable = shutil.which("python")
    if executable:
        return Path(executable)
    raise SeedVCError("Unable to find the seed-vc Python; set SEED_VC_PYTHON.")
