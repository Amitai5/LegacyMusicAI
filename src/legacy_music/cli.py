"""Top-level command-line application."""

from typing import Annotated

import typer

from legacy_music import __version__
from legacy_music.commands.doctor import doctor

app = typer.Typer(
    name="legacy-music",
    help="Authorization-first, local multi-artist music generation.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


def show_version(value: bool) -> None:
    """Print the installed application version and exit."""
    if value:
        typer.echo(f"legacy-music {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=show_version,
            help="Show the installed version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Initialize the Legacy Music AI command group."""


app.command(name="doctor")(doctor)
