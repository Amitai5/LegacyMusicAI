from pathlib import Path
from typing import Any

import pytest

from legacy_music.commands.train import TrainingResumeError, _latest_resume_checkpoint
from legacy_music.domain.training import MusicTrainingRequest
from legacy_music.engines.ace_step import AceStepError
from legacy_music.engines.ace_training import AceStepTrainingClient


class RecordingEngine:
    """Record one deterministic API request without opening a socket."""

    def __init__(self) -> None:
        self.path: str | None = None
        self.payload: dict[str, Any] | None = None

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        self.path = path
        self.payload = payload
        return {"method": method}


def _write_checkpoint(root: Path, epoch: int) -> Path:
    checkpoint = root / "output/checkpoints" / f"epoch_{epoch}_loss_0.1234"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (checkpoint / "training_state.pt").write_bytes(b"synthetic training state")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
    return checkpoint


def test_start_passes_verified_resume_checkpoint_to_api(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "training", 8)
    engine = RecordingEngine()
    client = AceStepTrainingClient(engine)  # type: ignore[arg-type]

    client.start(
        MusicTrainingRequest(artist_id="test-artist", epochs=10),
        tmp_path / "tensors",
        tmp_path / "output",
        resume_from=checkpoint,
    )

    assert engine.path == "v1/training/start"
    assert engine.payload is not None
    assert engine.payload["resume_from"] == str(checkpoint.resolve())
    assert engine.payload["train_epochs"] == 10


def test_start_rejects_incomplete_resume_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    with pytest.raises(AceStepError, match="training_state"):
        AceStepTrainingClient(RecordingEngine()).start(  # type: ignore[arg-type]
            MusicTrainingRequest(artist_id="test-artist"),
            tmp_path / "tensors",
            tmp_path / "output",
            resume_from=checkpoint,
        )


def test_latest_resume_checkpoint_selects_highest_epoch_below_target(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    _write_checkpoint(training_root, 6)
    expected = _write_checkpoint(training_root, 8)
    _write_checkpoint(training_root, 10)

    checkpoint, epoch = _latest_resume_checkpoint(training_root, target_epochs=10)

    assert checkpoint == expected.resolve()
    assert epoch == 8


def test_latest_resume_checkpoint_rejects_finalized_training(tmp_path: Path) -> None:
    training_root = tmp_path / "training"
    _write_checkpoint(training_root, 8)
    (training_root / "output/final").mkdir()

    with pytest.raises(TrainingResumeError, match="final model"):
        _latest_resume_checkpoint(training_root, target_epochs=10)
