"""Isolated SoulX-Singer preprocessing and SVC subprocess adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from legacy_music.audio import probe_audio
from legacy_music.config import VoiceQualityConfig
from legacy_music.domain.voice import (
    VoiceCandidateMetrics,
    VoiceConversionManifest,
    VoiceConversionMetrics,
    VoiceConversionRequest,
    VoiceConversionResult,
    VoicePhraseDecision,
    VoiceReviewStatus,
)
from legacy_music.engines.base import StemResult
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.utils.hashing import sha256_file
from legacy_music.voice_quality import (
    compare_f0,
    plan_phrases,
    select_reference,
    write_clean_f0,
)


class SoulXError(RuntimeError):
    """Raised when a SoulX-Singer runtime operation fails."""


@dataclass(frozen=True, slots=True)
class SoulXInstallation:
    """Resolved files belonging to one reviewed SoulX-Singer installation."""

    python: Path
    repository: Path
    runtime_script: Path
    model_path: Path
    config_path: Path
    separator_model: Path
    separator_config: Path
    dereverb_model: Path
    dereverb_config: Path
    f0_model: Path
    cache_root: Path

    @classmethod
    def from_project(cls, root: Path, python: Path | None = None) -> SoulXInstallation:
        """Resolve the standard project-local checkout, models, and Conda interpreter."""
        project = root.resolve()
        repository = project / "vendor/SoulX-Singer"
        shared = project / "models/shared/soulx"
        preprocess = shared / "SoulX-Singer-Preprocess"
        return cls(
            python=(python or _find_conda_python()).resolve(),
            repository=repository,
            runtime_script=project / "scripts/soulx_runtime.py",
            model_path=shared / "SoulX-Singer/model-svc.pt",
            config_path=repository / "soulxsinger/config/soulxsinger.yaml",
            separator_model=(
                preprocess
                / "mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt"
            ),
            separator_config=(
                preprocess / "mel-band-roformer-karaoke/config_karaoke_becruily.yaml"
            ),
            dereverb_model=(
                preprocess
                / "dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
            ),
            dereverb_config=(
                preprocess
                / "dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml"
            ),
            f0_model=preprocess / "rmvpe/rmvpe.pt",
            cache_root=project / "models/cache",
        )

    def validate(self, include_voice_model: bool = True) -> None:
        """Fail with one actionable message when required runtime files are missing."""
        required = [
            self.python,
            self.repository / ".git",
            self.runtime_script,
            self.config_path,
            self.separator_model,
            self.separator_config,
            self.f0_model,
        ]
        if include_voice_model:
            required.append(self.model_path)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SoulXError("SoulX-Singer is incomplete; missing: " + ", ".join(missing))

    def environment(self) -> dict[str, str]:
        """Return private project-local caches and offline-safe runtime defaults."""
        environment = os.environ.copy()
        cache_paths = {
            "HF_HOME": self.cache_root / "huggingface",
            "NUMBA_CACHE_DIR": self.cache_root / "numba",
            "MPLCONFIGDIR": self.cache_root / "matplotlib",
        }
        for name, path in cache_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            environment[name] = str(path)
        existing_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(self.repository), existing_python_path)
            if part
        )
        environment["WANDB_MODE"] = "disabled"
        environment["HF_HUB_OFFLINE"] = "1"
        return environment

    def validate_dereverb(self) -> None:
        """Require the reviewed dereverberation weights without changing base validation."""
        required = [
            self.python,
            self.repository / ".git",
            self.runtime_script,
            self.dereverb_model,
            self.dereverb_config,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SoulXError("SoulX dereverberation is incomplete; missing: " + ", ".join(missing))


class SoulXRuntime:
    """Run reviewed preprocessing helpers inside SoulX-Singer's isolated environment."""

    def __init__(
        self,
        installation: SoulXInstallation,
        device: str = "cuda",
        timeout_seconds: float = 1800,
    ) -> None:
        installation.validate(include_voice_model=False)
        self.installation = installation
        self.device = device
        self.timeout_seconds = timeout_seconds

    def extract_f0(self, audio: Path, output: Path) -> Path:
        """Extract and validate one SoulX-compatible RMVPE F0 contour."""
        probe_audio(audio)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            "f0",
            "--input",
            str(audio.resolve()),
            "--output",
            str(output.resolve()),
            "--model",
            str(self.installation.f0_model.resolve()),
            "--device",
            self.device,
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise SoulXError("SoulX-Singer did not create an F0 artifact.")
        return output

    def separate(self, audio: Path, output_dir: Path) -> StemResult:
        """Separate lead vocals/accompaniment and extract target F0 in one subprocess."""
        probe_audio(audio)
        output_dir.mkdir(parents=True, exist_ok=True)
        vocals = output_dir / "vocals.wav"
        accompaniment = output_dir / "accompaniment.wav"
        f0 = output_dir / "vocals-f0.npy"
        self._run(
            "separate",
            "--input",
            str(audio.resolve()),
            "--vocals",
            str(vocals.resolve()),
            "--accompaniment",
            str(accompaniment.resolve()),
            "--f0-output",
            str(f0.resolve()),
            "--separator-model",
            str(self.installation.separator_model.resolve()),
            "--separator-config",
            str(self.installation.separator_config.resolve()),
            "--f0-model",
            str(self.installation.f0_model.resolve()),
            "--device",
            self.device,
        )
        probe_audio(vocals)
        probe_audio(accompaniment)
        if not f0.is_file() or f0.stat().st_size == 0:
            raise SoulXError("SoulX-Singer did not create the target F0 artifact.")
        return StemResult(vocals=vocals, accompaniment=accompaniment, f0=f0)

    def dereverb_batch(
        self,
        jobs: tuple[tuple[Path, Path], ...],
        *,
        strength: float = 1.0,
    ) -> tuple[dict[str, object], ...]:
        """Dereverberate vocal stems with one model load and return per-file QC metrics."""
        if not jobs:
            return ()
        if not 0 <= strength <= 1:
            raise SoulXError("Dereverberation strength must be between zero and one.")
        self.installation.validate_dereverb()
        for source, output in jobs:
            probe_audio(source)
            if source.resolve() == output.resolve():
                raise SoulXError("Dereverberation output must not replace its source vocal stem.")
            output.parent.mkdir(parents=True, exist_ok=True)

        request_root = jobs[0][1].parent
        request_path = request_root / f".dereverb-request.{uuid4().hex}.json"
        result_path = request_root / f".dereverb-result.{uuid4().hex}.json"
        dump_json_atomic(
            request_path,
            {
                "schema_version": 1,
                "chunk_seconds": 5.0,
                "dereverb_strength": strength,
                "jobs": [
                    {
                        "input": str(source.resolve()),
                        "output": str(output.resolve()),
                    }
                    for source, output in jobs
                ],
            },
        )
        try:
            self._run(
                "dereverb-batch",
                "--request",
                str(request_path.resolve()),
                "--result",
                str(result_path.resolve()),
                "--model",
                str(self.installation.dereverb_model.resolve()),
                "--config",
                str(self.installation.dereverb_config.resolve()),
                "--device",
                self.device,
            )
            payload = load_json(result_path)
        finally:
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != len(jobs):
            raise SoulXError("SoulX dereverberation returned an incomplete batch result.")
        for (_source, output), item in zip(jobs, results, strict=True):
            if not isinstance(item, dict) or not output.is_file() or output.stat().st_size == 0:
                raise SoulXError("SoulX dereverberation did not create every requested output.")
            probe_audio(output)
        return tuple(results)

    def _run(self, *arguments: str) -> None:
        command = [
            str(self.installation.python),
            str(self.installation.runtime_script),
            *arguments,
        ]
        completed = subprocess.run(
            command,
            cwd=self.installation.repository,
            env=self.installation.environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-4000:] or completed.stdout.strip()[-4000:]
            raise SoulXError(f"SoulX-Singer preprocessing failed: {detail}")


class SoulXVoiceEngine:
    """Invoke the reviewed SoulX-Singer SVC CLI without importing its environment."""

    def __init__(
        self,
        installation: SoulXInstallation,
        device: str = "cuda",
        steps: int = 32,
        cfg: float = 3.0,
        fp16: bool = True,
        timeout_seconds: float = 1800,
        quality_config: VoiceQualityConfig | None = None,
    ) -> None:
        installation.validate(include_voice_model=True)
        self.installation = installation
        self.device = device
        self.steps = steps
        self.cfg = cfg
        self.fp16 = fp16
        self.timeout_seconds = timeout_seconds
        self.quality_config = quality_config

    def convert(
        self,
        request: VoiceConversionRequest,
        output: Path,
    ) -> VoiceConversionResult:
        """Convert a vocal target and return typed audio and quality evidence."""
        probe_audio(request.target_audio)
        for reference in request.references:
            if reference.reference.review_status is not VoiceReviewStatus.APPROVED:
                raise SoulXError(
                    f"Voice reference '{reference.reference.id}' is not approved."
                )
            probe_audio(reference.audio)
            expected = {
                reference.audio: reference.reference.audio_sha256,
                reference.f0: reference.reference.f0_sha256,
                reference.source: reference.reference.source_sha256,
            }
            for path, digest in expected.items():
                if not path.is_file() or sha256_file(path) != digest:
                    raise SoulXError(
                        f"Voice reference artifact failed integrity validation: {path.name}"
                    )
        for f0_path in (request.target_f0, *(item.f0 for item in request.references)):
            if not f0_path.is_file() or f0_path.stat().st_size == 0:
                raise SoulXError(f"Missing SoulX-Singer F0 artifact: {f0_path}")
        if self.quality_config is not None:
            return self._convert_quality(request, output)
        return self._convert_legacy(request, output)

    def _convert_legacy(
        self,
        request: VoiceConversionRequest,
        output: Path,
    ) -> VoiceConversionResult:
        if len(request.references) != 1:
            raise SoulXError("Legacy SoulX conversion accepts exactly one reference.")
        reference = request.references[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.installation.python),
            "cli/inference_svc.py",
            "--device",
            self.device,
            "--model_path",
            str(self.installation.model_path),
            "--config",
            str(self.installation.config_path),
            "--prompt_wav_path",
            str(reference.audio.resolve()),
            "--target_wav_path",
            str(request.target_audio.resolve()),
            "--prompt_f0_path",
            str(reference.f0.resolve()),
            "--target_f0_path",
            str(request.target_f0.resolve()),
            "--save_dir",
            str(output.parent.resolve()),
            "--auto_shift",
            "--n_steps",
            str(self.steps),
            "--cfg",
            str(self.cfg),
        ]
        if self.fp16:
            command.append("--fp16")
        completed = subprocess.run(
            command,
            cwd=self.installation.repository,
            env=self.installation.environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        generated = output.parent / "generated.wav"
        if completed.returncode != 0 or not generated.exists():
            detail = completed.stderr.strip()[-4000:] or completed.stdout.strip()[-4000:]
            raise SoulXError(f"SoulX-Singer conversion failed: {detail}")
        generated.replace(output)
        properties = probe_audio(output)
        cleaned_f0 = output.with_name(f"{output.stem}-cleaned-f0.npy")
        shutil.copyfile(request.target_f0, cleaned_f0)
        audio, _sample_rate = sf.read(output, dtype="float32", always_2d=True)
        peak = float(np.max(np.abs(audio)))
        candidate = VoiceCandidateMetrics(
            candidate_index=0,
            seed=request.seed,
            passed=True,
            raw_peak_dbfs=float(20 * np.log10(max(peak, 1e-12))),
            peak_reduction_db=0,
            sustained_dropout_count=0,
            dropout_fraction=0,
            spectral_distance_db=0,
        )
        phrase = VoicePhraseDecision(
            index=0,
            start_seconds=0,
            end_seconds=properties.duration_seconds,
            reference_id=reference.reference.id,
            selected_candidate_index=0,
            candidates=(candidate,),
        )
        metrics = VoiceConversionMetrics(
            phrase_count=1,
            selected_reference_ids=(reference.reference.id,),
            sustained_dropout_count=0,
            peak_dbfs=candidate.raw_peak_dbfs,
            clipping_fraction=float(np.mean(np.abs(audio) >= 0.999)),
        )
        manifest_path = output.parent / "voice-conversion.json"
        manifest = VoiceConversionManifest(
            engine="soulx",
            mode="legacy-single-reference",
            created_at=datetime.now(UTC),
            target_audio_sha256=sha256_file(request.target_audio),
            target_f0_sha256=sha256_file(request.target_f0),
            cleaned_f0_sha256=sha256_file(cleaned_f0),
            reference_audio_hashes={
                reference.reference.id: reference.reference.audio_sha256,
            },
            reference_f0_hashes={
                reference.reference.id: reference.reference.f0_sha256,
            },
            reference_source_hashes={
                reference.reference.id: reference.reference.source_sha256,
            },
            reference_parent_source_hashes={
                reference.reference.id: reference.reference.parent_source_sha256
                or reference.reference.source_sha256,
            },
            reference_stem_hashes=(
                {reference.reference.id: reference.reference.source_stem_sha256}
                if reference.reference.source_stem_sha256 is not None
                else {}
            ),
            settings={
                "steps": self.steps,
                "cfg": self.cfg,
                "fp16": self.fp16,
                "auto_shift": True,
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

    def _convert_quality(
        self,
        request: VoiceConversionRequest,
        output: Path,
    ) -> VoiceConversionResult:
        quality = self.quality_config
        if quality is None:
            raise SoulXError("Quality conversion requires validated quality configuration.")
        output.parent.mkdir(parents=True, exist_ok=True)
        cleaned_f0 = output.with_name(f"{output.stem}-cleaned-f0.npy")
        write_clean_f0(request.target_f0, cleaned_f0, quality.f0_gap_fill_ms)
        target_f0 = np.load(cleaned_f0, allow_pickle=False).astype(np.float32).reshape(-1)
        phrases = plan_phrases(
            target_f0,
            quality.phrase_min_seconds,
            quality.phrase_max_seconds,
            quality.phrase_overlap_seconds,
        )
        phrase_requests = []
        for phrase in phrases:
            reference = select_reference(
                request.references,
                request.target_audio,
                target_f0,
                phrase,
            )
            phrase_requests.append(
                {
                    "index": phrase.index,
                    "start_frame": phrase.start_frame,
                    "end_frame": phrase.end_frame,
                    "reference_id": reference.reference.id,
                }
            )

        runtime_result = output.parent / "voice-runtime-result.json"
        request_path = output.parent / f".voice-request.{uuid4().hex}.json"
        settings = {
            "phrase_min_seconds": quality.phrase_min_seconds,
            "phrase_max_seconds": quality.phrase_max_seconds,
            "phrase_overlap_seconds": quality.phrase_overlap_seconds,
            "candidates_per_phrase": quality.candidates_per_phrase,
            "f0_gap_fill_ms": quality.f0_gap_fill_ms,
            "steps": self.steps,
            "cfg": self.cfg,
            "fp16": self.fp16,
            "auto_shift": False,
        }
        dump_json_atomic(
            request_path,
            {
                "target_audio": str(request.target_audio.resolve()),
                "target_f0": str(cleaned_f0.resolve()),
                "output": str(output.resolve()),
                "result": str(runtime_result.resolve()),
                "phrases_directory": str((output.parent / "voice-phrases").resolve()),
                "seed": request.seed,
                "settings": settings,
                "phrases": phrase_requests,
                "references": [
                    {
                        "id": item.reference.id,
                        "audio": str(item.audio.resolve()),
                        "f0": str(item.f0.resolve()),
                    }
                    for item in request.references
                ],
            },
        )
        command = [
            str(self.installation.python),
            str(self.installation.runtime_script),
            "quality-convert",
            "--request",
            str(request_path),
            "--model",
            str(self.installation.model_path),
            "--config",
            str(self.installation.config_path),
            "--device",
            self.device,
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
                check=False,
                timeout=self.timeout_seconds,
            )
        finally:
            request_path.unlink(missing_ok=True)
        if completed.returncode != 0 or not output.is_file() or not runtime_result.is_file():
            detail = completed.stderr.strip()[-8000:] or completed.stdout.strip()[-8000:]
            raise SoulXError(f"SoulX-Singer quality conversion failed: {detail}")
        probe_audio(output)

        runtime_payload = load_json(runtime_result)
        decisions = tuple(
            VoicePhraseDecision.model_validate(item)
            for item in runtime_payload.get("phrases", [])
        )
        metrics = VoiceConversionMetrics.model_validate(runtime_payload.get("metrics", {}))
        converted_f0 = output.with_name(f"{output.stem}-f0.npy")
        SoulXRuntime(
            self.installation,
            device=self.device,
            timeout_seconds=self.timeout_seconds,
        ).extract_f0(output, converted_f0)
        median_error, gross_error = compare_f0(
            target_f0,
            np.load(converted_f0, allow_pickle=False),
        )
        metrics = metrics.model_copy(
            update={
                "f0_median_error_cents": median_error,
                "f0_gross_error_fraction": gross_error,
            }
        )
        manifest_path = output.parent / "voice-conversion.json"
        manifest = VoiceConversionManifest(
            engine="soulx",
            mode="quality-multi-reference",
            created_at=datetime.now(UTC),
            target_audio_sha256=sha256_file(request.target_audio),
            target_f0_sha256=sha256_file(request.target_f0),
            cleaned_f0_sha256=sha256_file(cleaned_f0),
            reference_audio_hashes={
                item.reference.id: item.reference.audio_sha256
                for item in request.references
            },
            reference_f0_hashes={
                item.reference.id: item.reference.f0_sha256 for item in request.references
            },
            reference_source_hashes={
                item.reference.id: item.reference.source_sha256
                for item in request.references
            },
            reference_parent_source_hashes={
                item.reference.id: item.reference.parent_source_sha256
                or item.reference.source_sha256
                for item in request.references
            },
            reference_stem_hashes={
                item.reference.id: item.reference.source_stem_sha256
                for item in request.references
                if item.reference.source_stem_sha256 is not None
            },
            settings={**settings, "converted_f0": converted_f0.name},
            phrases=decisions,
            metrics=metrics,
            output_sha256=sha256_file(output),
        )
        dump_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        return VoiceConversionResult(
            audio=output,
            manifest=manifest_path,
            cleaned_f0=cleaned_f0,
            reference_ids=metrics.selected_reference_ids,
            phrases=decisions,
            metrics=metrics,
        )


def _find_conda_python() -> Path:
    override = os.environ.get("SOULX_PYTHON")
    if override:
        return Path(override)
    candidates = [
        Path.home() / ".conda/envs/soulxsinger/python.exe",
        Path("C:/ProgramData/anaconda3/envs/soulxsinger/python.exe"),
        Path("C:/ProgramData/miniconda3/envs/soulxsinger/python.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    executable = shutil.which("python")
    if executable:
        return Path(executable)
    raise SoulXError("Unable to find the soulxsinger Python; set SOULX_PYTHON.")
