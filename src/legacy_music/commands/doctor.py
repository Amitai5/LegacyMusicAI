"""Local prerequisite diagnostics."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True, slots=True)
class ToolCheck:
    """Availability result for one external prerequisite."""

    name: str
    required_for_scaffold: bool
    path: str | None

    @property
    def is_available(self) -> bool:
        """Return whether the executable is present on PATH."""
        return self.path is not None


def collect_tool_checks() -> list[ToolCheck]:
    """Inspect external tools without mutating the machine."""
    requirements = {
        "git": True,
        "uv": True,
        "ffmpeg": False,
        "ffprobe": False,
        "nvidia-smi": False,
        "conda": False,
    }
    return [
        ToolCheck(
            name=name,
            required_for_scaffold=required,
            path=_find_tool(name),
        )
        for name, required in requirements.items()
    ]


def _find_tool(name: str) -> str | None:
    discovered = shutil.which(name)
    if discovered is not None:
        return discovered
    if name == "conda":
        for candidate in (
            Path("C:/ProgramData/anaconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/miniconda3/Scripts/conda.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def doctor(
    strict: Annotated[
        bool,
        typer.Option(help="Exit nonzero when a GPU runtime or required model artifact is missing."),
    ] = False,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = Path("."),
) -> None:
    """Report the local application, audio, GPU, and environment prerequisites."""
    console = Console()
    table = Table(title="Legacy Music AI System Check")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    python_supported = sys.version_info[:2] == (3, 11)
    table.add_row(
        "Python",
        "ready" if python_supported else "unsupported",
        platform.python_version(),
    )

    checks = collect_tool_checks()
    for check in checks:
        if check.is_available:
            status = "ready"
            details = check.path or ""
        elif check.required_for_scaffold:
            status = "missing"
            details = "required for application development"
        elif check.name in {"ffmpeg", "ffprobe"}:
            status = "optional"
            details = "SoundFile fallback is active"
        else:
            status = "not installed"
            details = "required by the installed GPU model runtime"
        table.add_row(check.name, status, details)

    project = root.resolve()
    model_artifacts = {
        "ACE-Step turbo": project
        / "models/shared/ace-step/acestep-v15-turbo/model.safetensors",
        "ACE-Step VAE": project
        / "models/shared/ace-step/vae/diffusion_pytorch_model.safetensors",
        "SoulX SVC": project / "models/shared/soulx/SoulX-Singer/model-svc.pt",
        "SoulX separator": project
        / (
            "models/shared/soulx/SoulX-Singer-Preprocess/mel-band-roformer-karaoke/"
            "mel_band_roformer_karaoke_becruily.ckpt"
        ),
        "SoulX RMVPE": project
        / "models/shared/soulx/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
    }
    for name, path in model_artifacts.items():
        table.add_row(
            name,
            "ready" if path.is_file() and path.stat().st_size > 0 else "missing",
            str(path),
        )

    console.print(table)

    strict_tools = {"nvidia-smi", "conda"}
    required_missing = not python_supported or any(
        not check.is_available
        and (check.required_for_scaffold or (strict and check.name in strict_tools))
        for check in checks
    )
    missing_model = any(
        not path.is_file() or path.stat().st_size == 0
        for path in model_artifacts.values()
    )
    if strict and missing_model:
        required_missing = True
    if required_missing:
        raise typer.Exit(code=1)
