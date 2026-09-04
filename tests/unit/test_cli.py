from typer.testing import CliRunner

from legacy_music import __version__
from legacy_music.cli import app

runner = CliRunner()


def test_help_when_requested_lists_foundation_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Authorization-first" in result.stdout
    assert "doctor" in result.stdout
    assert "train" in result.stdout
    assert "voice" in result.stdout


def test_version_when_requested_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"legacy-music {__version__}"


def test_generate_help_lists_direct_fine_tuned_voice_engine_options() -> None:
    result = runner.invoke(app, ["generate", "--help"])

    assert result.exit_code == 0
    assert "--voice-engine" in result.stdout
    assert "--seed-vc-python" in result.stdout
    assert "--seed-vc-steps" in result.stdout


def test_voice_remix_help_lists_identity_gated_options() -> None:
    result = runner.invoke(app, ["voice", "remix-run", "--help"])

    assert result.exit_code == 0
    assert "--identity-gated" in result.stdout
    assert "--identity-candid" in result.stdout
    assert "--identity-thresh" in result.stdout
