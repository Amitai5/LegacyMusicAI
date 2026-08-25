"""Generation run inspection commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from legacy_music.paths import ProjectPaths
from legacy_music.repositories import RunRepository

runs_app = typer.Typer(help="Inspect durable generation history.", no_args_is_help=True)
DefaultProjectRoot = Path(".")


@runs_app.command("list")
def list_runs(
    artist_id: Annotated[
        str | None,
        typer.Option("--artist", help="Filter by artist profile."),
    ] = None,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List run IDs, artists, stages, and update times."""
    repository = RunRepository(ProjectPaths.from_root(root).runs)
    try:
        runs = repository.list(artist_id)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(json.dumps([run.model_dump(mode="json") for run in runs], indent=2))
        return
    table = Table("Run", "Artist", "Stage", "Updated (UTC)")
    for run in runs:
        table.add_row(run.run_id, run.artist_id, run.stage.value, run.updated_at.isoformat())
    Console().print(table)


@runs_app.command("show")
def show_run(
    run_id: Annotated[str, typer.Argument(help="Run identifier to inspect.")],
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Print a complete sanitized run manifest."""
    repository = RunRepository(ProjectPaths.from_root(root).runs)
    try:
        run = repository.load(run_id)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(run.model_dump(mode="json"), indent=2))
