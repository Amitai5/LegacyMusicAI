"""Official ACE-Step dataset preprocessing and LoRA training API client."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from legacy_music.domain.training import MusicTrainingRequest, TrainingDatasetManifest
from legacy_music.engines.ace_step import AceStepApiEngine, AceStepError


class AceStepTrainingClient:
    """Drive the reviewed local ACE-Step training endpoints synchronously."""

    def __init__(
        self,
        engine: AceStepApiEngine,
        poll_seconds: float = 2,
        timeout_seconds: float = 24 * 60 * 60,
    ) -> None:
        self.engine = engine
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def prepare_tensors(
        self,
        dataset: TrainingDatasetManifest,
        artist_root: Path,
        tensor_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Scan, validate labels, save, and preprocess one prepared dataset."""
        audio_dir = (artist_root / dataset.audio_dir).resolve()
        scan = self.engine._request_json(
            "v1/dataset/scan",
            {
                "audio_dir": str(audio_dir),
                "dataset_name": dataset.dataset_id,
                "custom_tag": dataset.custom_tag,
                "tag_position": "prepend",
                "all_instrumental": dataset.is_instrumental,
            },
        )
        samples = scan.get("samples", []) if isinstance(scan, dict) else []
        if len(samples) != len(dataset.songs):
            raise AceStepError(
                f"ACE-Step scanned {len(samples)} samples; expected {len(dataset.songs)}."
            )
        unlabeled = [
            sample.get("filename", "unknown")
            for sample in samples
            if not sample.get("labeled")
        ]
        if unlabeled:
            raise AceStepError("ACE-Step rejected unlabeled samples: " + ", ".join(unlabeled))

        dataset_json = tensor_dir.parent / "ace-step-dataset.json"
        self.engine._request_json(
            "v1/dataset/save",
            {
                "save_path": str(dataset_json),
                "dataset_name": dataset.dataset_id,
                "custom_tag": dataset.custom_tag,
                "tag_position": "prepend",
                "all_instrumental": dataset.is_instrumental,
            },
        )
        started = self.engine._request_json(
            "v1/dataset/preprocess_async",
            {"output_dir": str(tensor_dir), "skip_existing": True},
        )
        task_id = started.get("task_id") if isinstance(started, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise AceStepError("ACE-Step did not return a preprocessing task ID.")

        deadline = time.monotonic() + self.timeout_seconds
        previous_progress = None
        while True:
            status = self.engine._request_json(
                f"v1/dataset/preprocess_status/{task_id}",
                method="GET",
            )
            state = status.get("status")
            current_progress = status.get("progress")
            if progress is not None and current_progress != previous_progress:
                progress(str(current_progress))
                previous_progress = current_progress
            if state == "completed":
                return status
            if state == "failed":
                raise AceStepError(
                    f"ACE-Step dataset preprocessing failed: {status.get('error')}"
                )
            if time.monotonic() >= deadline:
                raise AceStepError("ACE-Step dataset preprocessing timed out.")
            time.sleep(self.poll_seconds)

    def start(
        self,
        request: MusicTrainingRequest,
        tensor_dir: Path,
        output_dir: Path,
        resume_from: Path | None = None,
    ) -> dict[str, Any]:
        """Start one low-VRAM LoRA training job."""
        payload: dict[str, Any] = {
            "tensor_dir": str(tensor_dir.resolve()),
            "lora_rank": request.rank,
            "lora_alpha": request.rank * 2,
            "lora_dropout": 0.1,
            "learning_rate": 0.0001,
            "train_epochs": request.epochs,
            "train_batch_size": 1,
            "gradient_accumulation": request.gradient_accumulation,
            "save_every_n_epochs": request.save_every,
            "training_shift": 3.0,
            "training_seed": request.seed,
            "lora_output_dir": str(output_dir.resolve()),
            "use_fp8": False,
            "gradient_checkpointing": request.gradient_checkpointing,
        }
        if resume_from is not None:
            checkpoint = resume_from.resolve()
            if not (checkpoint / "training_state.pt").is_file():
                raise AceStepError("Resume checkpoint has no training_state.pt file.")
            if not (checkpoint / "adapter/adapter_config.json").is_file():
                raise AceStepError("Resume checkpoint has no loadable adapter configuration.")
            payload["resume_from"] = str(checkpoint)
        return self.engine._request_json(
            "v1/training/start",
            payload,
        )

    def wait(self, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Wait for the active training job to finish and surface its final error."""
        deadline = time.monotonic() + self.timeout_seconds
        previous = None
        observed_training = False
        while True:
            status = self.status()
            observed_training = observed_training or bool(status.get("is_training"))
            message = str(status.get("status", "Unknown"))
            if progress is not None and message != previous:
                progress(message)
                previous = message
            if status.get("error"):
                raise AceStepError(f"ACE-Step training failed: {status['error']}")
            is_training = bool(status.get("is_training"))
            if not is_training and (observed_training or message not in {"Idle", "Starting..."}):
                return status
            if time.monotonic() >= deadline:
                raise AceStepError("ACE-Step training timed out.")
            time.sleep(self.poll_seconds)

    def status(self) -> dict[str, Any]:
        """Return the active official ACE-Step training status payload."""
        return self.engine._request_json("v1/training/status", method="GET")

    def export(self, training_output: Path, adapter_output: Path) -> dict[str, Any]:
        """Export the final or latest checkpoint into an artist-owned adapter directory."""
        return self.engine._request_json(
            "v1/training/export",
            {
                "lora_output_dir": str(training_output.resolve()),
                "export_path": str(adapter_output.resolve()),
            },
        )
