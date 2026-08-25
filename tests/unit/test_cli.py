from typer.testing import CliRunner

from legacy_music import __version__
from legacy_music.cli import app

runner = CliRunner()


def test_help_when_requested_lists_foundation_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Authorization-first" in result.stdout
    assert "doctor" in result.stdout


def test_version_when_requested_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"legacy-music {__version__}"
