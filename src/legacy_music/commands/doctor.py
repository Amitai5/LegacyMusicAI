"""Local prerequisite diagnostics."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
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
        ToolCheck(name=name, required_for_scaffold=required, path=shutil.which(name))
        for name, required in requirements.items()
    ]


def doctor(
    strict: Annotated[
        bool,
        typer.Option(help="Exit nonzero when any future model or audio prerequisite is missing."),
    ] = False,
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
        else:
            status = "not installed"
            details = "required by a later audio or model milestone"
        table.add_row(check.name, status, details)

    console.print(table)

    required_missing = not python_supported or any(
        not check.is_available and (strict or check.required_for_scaffold) for check in checks
    )
    if required_missing:
        raise typer.Exit(code=1)
