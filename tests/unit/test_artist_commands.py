import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from legacy_music.cli import app
from legacy_music.domain.voice import VoiceReference
from legacy_music.paths import ProjectPaths
from legacy_music.persistence import dump_json_atomic
from legacy_music.repositories import ArtistRepository


def test_artist_status_counts_curated_reference_directories(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    ArtistRepository(paths).create("test-artist", "Test Artist", singing_voice=True)
    reference_root = paths.artist_root("test-artist") / "voice/references/neutral"
    reference = VoiceReference(
        id="neutral",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_file=Path("voice/references/neutral/source.wav"),
        source_sha256="a" * 64,
        audio_file=Path("voice/references/neutral/audio.wav"),
        audio_sha256="b" * 64,
        f0_file=Path("voice/references/neutral/f0.npy"),
        f0_sha256="c" * 64,
    )
    dump_json_atomic(
        reference_root / "reference.json",
        reference.model_dump(mode="json"),
    )

    result = CliRunner().invoke(
        app,
        ["artist", "status", "test-artist", "--root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["voice_references"] == 1
