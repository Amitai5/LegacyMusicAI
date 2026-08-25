from datetime import UTC, datetime
from pathlib import Path

import pytest

from legacy_music.domain.generation import (
    GenerationRequest,
    LyricsRequest,
    MusicGenerationRequest,
    PipelineStage,
)
from legacy_music.repositories.run import RunRepository, RunRepositoryError


def request(lyrics: Path) -> GenerationRequest:
    return GenerationRequest(
        id="generation-test",
        artist_id="test-artist",
        music=MusicGenerationRequest(prompt="Test prompt", duration_seconds=10, seed=42),
        lyrics=LyricsRequest(file=lyrics),
    )


def test_run_repository_persists_legal_transitions(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs")
    run_id = repository.new_run_id(datetime(2026, 8, 24, tzinfo=UTC))
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Instrumental]", encoding="utf-8")

    repository.create(run_id, request(lyrics))
    repository.transition(run_id, PipelineStage.MUSIC_GENERATED, "Draft created.")
    completed = repository.transition(run_id, PipelineStage.COMPLETE, "Complete.")

    assert completed.stage is PipelineStage.COMPLETE
    assert [event.sequence for event in completed.events] == [0, 1, 2]
    assert repository.load(run_id) == completed


def test_run_repository_rejects_invalid_transition(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runs")
    run_id = repository.new_run_id(datetime(2026, 8, 24, tzinfo=UTC))
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Instrumental]", encoding="utf-8")
    repository.create(run_id, request(lyrics))

    with pytest.raises(RunRepositoryError, match="Invalid run transition"):
        repository.transition(run_id, PipelineStage.COMPLETE, "Skipped generation.")
