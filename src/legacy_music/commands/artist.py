"""Artist profile management commands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from legacy_music.authorization import AuthorizationError, require_capability
from legacy_music.paths import ProjectPaths
from legacy_music.repositories import (
    ArtistRepository,
    CatalogRepository,
    VoiceReferenceRepository,
)

artist_app = typer.Typer(help="Create and inspect isolated artist profiles.", no_args_is_help=True)
DefaultProjectRoot = Path(".")


@artist_app.command("create")
def create_artist(
    artist_id: Annotated[str, typer.Argument(help="Canonical lowercase artist identifier.")],
    name: Annotated[str, typer.Option("--name", help="Display name for the artist profile.")],
    singing_voice: Annotated[
        bool,
        typer.Option("--singing-voice", help="Configure a voice reference library."),
    ] = False,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Create a pending artist profile without granting any rights."""
    repository = ArtistRepository(ProjectPaths.from_root(root))
    try:
        profile = repository.create(artist_id, name, singing_voice=singing_voice)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Created artist '{profile.id}'. Review and approve its rights.yaml before use.")


@artist_app.command("list")
def list_artists(
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List configured local artist profiles."""
    repository = ArtistRepository(ProjectPaths.from_root(root))
    profiles = repository.list()
    if json_output:
        typer.echo(json.dumps([profile.model_dump(mode="json") for profile in profiles], indent=2))
        return
    table = Table("ID", "Display name", "Music", "Voice")
    for profile in profiles:
        table.add_row(
            profile.id,
            profile.display_name,
            "configured" if profile.capabilities.music_style else "disabled",
            "configured" if profile.capabilities.singing_voice else "disabled",
        )
    Console().print(table)


@artist_app.command("status")
def artist_status(
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
    """Report authorization, dataset, adapter, and voice-reference readiness."""
    paths = ProjectPaths.from_root(root)
    repository = ArtistRepository(paths)
    try:
        profile = repository.get(artist_id)
        rights = repository.get_rights(artist_id)
        artist_root = paths.artist_root(artist_id)
        catalog = CatalogRepository(artist_root, artist_id).load()
        now = datetime.now(UTC)
        authorization = {}
        for capability in ("training", "music_style", "singing_voice", "commercial_release"):
            try:
                require_capability(rights, artist_id, capability, now)
                authorization[capability] = True
            except AuthorizationError:
                authorization[capability] = False
        selected_adapter = artist_root / "models/music/selected.json"
        reference_count = len(VoiceReferenceRepository(artist_root, artist_id).list())
        status = {
            "artist_id": artist_id,
            "display_name": profile.display_name,
            "rights_status": rights.status.value,
            "authorization": authorization,
            "catalog_songs": len(catalog.songs),
            "music_adapter_selected": selected_adapter.is_file(),
            "voice_references": reference_count,
        }
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(json.dumps(status, indent=2))
        return
    table = Table("Check", "Value")
    for key, value in status.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    Console().print(table)
