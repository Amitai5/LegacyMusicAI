"""Local model-service inspection commands."""

from __future__ import annotations

import json
import os
from typing import Annotated

import typer

from legacy_music.engines.ace_step import AceStepApiEngine

engine_app = typer.Typer(help="Inspect installed local model services.", no_args_is_help=True)


@engine_app.command("status")
def engine_status(
    ace_url: Annotated[
        str,
        typer.Option("--ace-url", help="Loopback ACE-Step API base URL."),
    ] = "http://127.0.0.1:8001",
) -> None:
    """Check ACE-Step health and loaded models."""
    try:
        engine = AceStepApiEngine(ace_url, api_key=os.environ.get("ACESTEP_API_KEY"))
        payload = {"health": engine.health(), "models": engine.models()}
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(payload, indent=2))
