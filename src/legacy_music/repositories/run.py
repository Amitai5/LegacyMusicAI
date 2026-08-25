"""Immutable-directory and atomic-manifest generation run repository."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from legacy_music.domain.generation import GenerationRequest, PipelineStage
from legacy_music.domain.run import RunEvent, RunManifest
from legacy_music.persistence import PersistenceError, dump_json_atomic, file_lock, load_json


class RunRepositoryError(RuntimeError):
    """Raised when a run transition or persisted record is invalid."""


AllowedTransitions = {
    PipelineStage.CREATED: {PipelineStage.MUSIC_GENERATED, PipelineStage.FAILED},
    PipelineStage.MUSIC_GENERATED: {
        PipelineStage.VOCALS_SEPARATED,
        PipelineStage.MIXED,
        PipelineStage.COMPLETE,
        PipelineStage.FAILED,
    },
    PipelineStage.VOCALS_SEPARATED: {PipelineStage.VOICE_CONVERTED, PipelineStage.FAILED},
    PipelineStage.VOICE_CONVERTED: {PipelineStage.MIXED, PipelineStage.FAILED},
    PipelineStage.MIXED: {PipelineStage.COMPLETE, PipelineStage.FAILED},
    PipelineStage.COMPLETE: set(),
    PipelineStage.FAILED: set(),
}


class RunRepository:
    """Create unique run roots and atomically append validated stage events."""

    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.resolve()

    def new_run_id(self, at: datetime | None = None) -> str:
        """Create a sortable collision-resistant run identifier."""
        instant = (at or datetime.now(UTC)).astimezone(UTC)
        return f"run-{instant:%Y%m%dt%H%M%Sz}-{uuid4().hex[:8]}"

    def create(self, run_id: str, request: GenerationRequest) -> RunManifest:
        """Create a new immutable run directory and initial event."""
        root = self.root(run_id)
        if root.exists():
            raise RunRepositoryError(f"Run '{run_id}' already exists.")
        now = datetime.now(UTC)
        event = RunEvent(
            sequence=0,
            stage=PipelineStage.CREATED,
            occurred_at=now,
            message="Generation request accepted.",
        )
        manifest = RunManifest(
            run_id=run_id,
            artist_id=request.artist_id,
            request=request,
            created_at=now,
            updated_at=now,
            events=(event,),
        )
        try:
            root.mkdir(parents=True)
            for directory in ("inputs", "intermediates", "output", "logs"):
                (root / directory).mkdir()
            dump_json_atomic(root / "request.json", request.model_dump(mode="json"))
            dump_json_atomic(root / "run.json", manifest.model_dump(mode="json"))
        except (OSError, PersistenceError) as error:
            raise RunRepositoryError(f"Unable to create run '{run_id}': {error}") from error
        return manifest

    def load(self, run_id: str) -> RunManifest:
        """Load and validate one run manifest."""
        try:
            return RunManifest.model_validate(load_json(self.root(run_id) / "run.json"))
        except (PersistenceError, ValidationError) as error:
            raise RunRepositoryError(f"Invalid run '{run_id}': {error}") from error

    def transition(
        self,
        run_id: str,
        stage: PipelineStage,
        message: str,
        artifacts: dict[str, str] | None = None,
        hashes: dict[str, str] | None = None,
        error: str | None = None,
    ) -> RunManifest:
        """Atomically append one legal state transition."""
        lock_path = self.root(run_id) / ".run.lock"
        try:
            with file_lock(lock_path):
                current = self.load(run_id)
                if stage not in AllowedTransitions[current.stage]:
                    raise RunRepositoryError(
                        f"Invalid run transition: {current.stage.value} -> {stage.value}."
                    )
                now = datetime.now(UTC)
                event = RunEvent(
                    sequence=len(current.events),
                    stage=stage,
                    occurred_at=now,
                    message=message,
                    artifacts=artifacts or {},
                    hashes=hashes or {},
                )
                updated = current.model_copy(
                    update={
                        "stage": stage,
                        "updated_at": now,
                        "events": (*current.events, event),
                        "error": error,
                    }
                )
                dump_json_atomic(self.root(run_id) / "run.json", updated.model_dump(mode="json"))
                return updated
        except PersistenceError as error_value:
            raise RunRepositoryError(str(error_value)) from error_value

    def list(self, artist_id: str | None = None) -> tuple[RunManifest, ...]:
        """List valid persisted runs, optionally filtered by artist."""
        if not self.runs_root.exists():
            return ()
        runs = []
        for child in sorted(self.runs_root.glob("run-*"), reverse=True):
            if not child.is_dir():
                continue
            manifest = self.load(child.name)
            if artist_id is None or manifest.artist_id == artist_id:
                runs.append(manifest)
        return tuple(runs)

    def root(self, run_id: str) -> Path:
        """Resolve a validated direct run child."""
        if not run_id.startswith("run-") or any(char in run_id for char in "/\\"):
            raise RunRepositoryError(f"Invalid run ID: {run_id}")
        candidate = (self.runs_root / run_id).resolve()
        if candidate.parent != self.runs_root:
            raise RunRepositoryError("Run path escaped the run repository.")
        return candidate
