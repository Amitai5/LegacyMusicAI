"""Isolated SoulX-Singer preprocessing and SVC subprocess adapters."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from legacy_music.audio import probe_audio
from legacy_music.domain.voice import VoiceConversionRequest
from legacy_music.engines.base import StemResult


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
    ) -> None:
        installation.validate(include_voice_model=True)
        self.installation = installation
        self.device = device
        self.steps = steps
        self.cfg = cfg
        self.fp16 = fp16
        self.timeout_seconds = timeout_seconds

    def convert(self, request: VoiceConversionRequest, output: Path) -> Path:
        """Convert a vocal target with its authorized reference and F0 contours."""
        probe_audio(request.target_audio)
        probe_audio(request.reference_audio)
        for f0_path in (request.target_f0, request.reference_f0):
            if not f0_path.is_file() or f0_path.stat().st_size == 0:
                raise SoulXError(f"Missing SoulX-Singer F0 artifact: {f0_path}")
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
            str(request.reference_audio.resolve()),
            "--target_wav_path",
            str(request.target_audio.resolve()),
            "--prompt_f0_path",
            str(request.reference_f0.resolve()),
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
        probe_audio(output)
        return output


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
