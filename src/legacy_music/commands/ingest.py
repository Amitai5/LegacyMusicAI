"""Immutable rights-gated source-audio ingestion command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from legacy_music.paths import ProjectPaths
from legacy_music.pipeline.ingestion import ingest_audio, inventory_audio
from legacy_music.repositories import ArtistRepository

DefaultProjectRoot = Path(".")


def ingest(
    artist_id: Annotated[str, typer.Argument(help="Explicit artist profile identifier.")],
    source: Annotated[Path, typer.Argument(help="Audio file or directory to inspect/import.")],
    inventory_only: Annotated[
        bool,
        typer.Option("--inventory-only", help="Hash and validate sources without importing."),
    ] = False,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Inventory or import only exactly authorized audio recordings."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        rights = artists.get_rights(artist_id)
        if inventory_only:
            results = inventory_audio(source, rights)
        else:
            results = ingest_audio(artist_id, paths.artist_root(artist_id), rights, source)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    values = [result.model_dump(mode="json") for result in results]
    if json_output:
        typer.echo(json.dumps(values, indent=2))
        return
    if not results:
        typer.echo("No supported audio files found.")
        return
    for result in results:
        if inventory_only:
            label = "authorized" if result.is_authorized else "NOT AUTHORIZED"
            typer.echo(f"{result.sha256}  {label}  {result.path.name}")
        else:
            typer.echo(f"{result.status}: {result.song_id} ({result.sha256})")
