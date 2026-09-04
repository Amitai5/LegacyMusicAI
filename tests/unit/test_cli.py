import pytest
from rich.text import Text
from typer import rich_utils
from typer.testing import CliRunner

from legacy_music import __version__
from legacy_music.cli import app

runner = CliRunner()


@pytest.fixture(params=[False, True], ids=["plain", "ansi"])
def help_color(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", request.param)
    monkeypatch.setattr(rich_utils, "COLOR_SYSTEM", "standard" if request.param else None)
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", 120)
    return bool(request.param)


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


def test_generate_help_lists_direct_fine_tuned_voice_engine_options(help_color: bool) -> None:
    result = runner.invoke(app, ["generate", "--help"], color=help_color)

    assert result.exit_code == 0
    help_text = Text.from_ansi(result.stdout).plain
    assert "--voice-engine" in help_text
    assert "--seed-vc-python" in help_text
    assert "--seed-vc-steps" in help_text


def test_voice_remix_help_lists_identity_gated_options(help_color: bool) -> None:
    result = runner.invoke(app, ["voice", "remix-run", "--help"], color=help_color)

    assert result.exit_code == 0
    help_text = Text.from_ansi(result.stdout).plain
    assert "--identity-gated" in help_text
    assert "--identity-candid" in help_text
    assert "--identity-thresh" in help_text
