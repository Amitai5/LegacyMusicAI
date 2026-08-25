"""Authorized artist-specific singing-voice reference commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from legacy_music.engines.soulx import SoulXInstallation, SoulXRuntime
from legacy_music.paths import ProjectPaths
from legacy_music.repositories import ArtistRepository, VoiceReferenceRepository

voice_app = typer.Typer(
    help="Curate authorized singing-voice references.",
    no_args_is_help=True,
)
DefaultProjectRoot = Path(".")


@voice_app.command("add-reference")
def add_reference(
    artist_id: Annotated[str, typer.Argument(help="Explicit artist profile identifier.")],
    reference_id: Annotated[str, typer.Argument(help="Canonical reference identifier.")],
    source: Annotated[Path, typer.Argument(help="Exact rights-cleared vocal source file.")],
    tags: Annotated[
        str,
        typer.Option("--tags", help="Comma-separated register, energy, or style tags."),
    ] = "",
    soulx_python: Annotated[
        Path | None,
        typer.Option("--soulx-python", help="Override the SoulX Conda Python executable."),
    ] = None,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Add one hash-authorized source and precompute its SoulX F0 contour."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        profile = artists.get(artist_id)
        if not profile.capabilities.singing_voice:
            raise ValueError("Singing voice is disabled in this artist profile.")
        rights = artists.get_rights(artist_id)
        installation = SoulXInstallation.from_project(paths.root, soulx_python)
        runtime = SoulXRuntime(installation)
        repository = VoiceReferenceRepository(paths.artist_root(artist_id), artist_id)
        resolved = repository.create(
            reference_id,
            source,
            rights,
            runtime,
            tags=tuple(tag.strip() for tag in tags.split(",") if tag.strip()),
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Created authorized voice reference '{resolved.reference.id}' "
        f"({resolved.reference.audio_sha256})."
    )


@voice_app.command("list-references")
def list_references(
    artist_id: Annotated[str, typer.Argument(help="Artist profile to inspect.")],
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List private reference metadata without exposing source audio."""
    paths = ProjectPaths.from_root(root)
    try:
        ArtistRepository(paths).get(artist_id)
        references = VoiceReferenceRepository(
            paths.artist_root(artist_id),
            artist_id,
        ).list()
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(
            json.dumps(
                [reference.model_dump(mode="json") for reference in references],
                indent=2,
            )
        )
        return
    table = Table("ID", "Tags", "Created (UTC)", "Audio SHA-256")
    for reference in references:
        table.add_row(
            reference.id,
            ", ".join(reference.tags),
            reference.created_at.isoformat(),
            reference.audio_sha256,
        )
    Console().print(table)
