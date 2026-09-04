"""Authorization-gated ACE-Step dataset and LoRA training commands."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from legacy_music.authorization import require_capability
from legacy_music.domain.training import MusicTrainingRequest
from legacy_music.engines.ace_step import AceStepApiEngine
from legacy_music.engines.ace_training import AceStepTrainingClient
from legacy_music.paths import ProjectPaths
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.repositories import (
    ArtistRepository,
    CatalogRepository,
    TrainingRepository,
)

train_app = typer.Typer(
    help="Prepare and train isolated artist-specific ACE-Step LoRA adapters.",
    no_args_is_help=True,
)
DefaultProjectRoot = Path(".")
DefaultAceUrl = "http://127.0.0.1:8001"
ResumeCheckpointPattern = re.compile(r"^epoch_(\d+)_loss_.+$")


class TrainingResumeError(RuntimeError):
    """Raised when an existing music-training run cannot resume safely."""


def _latest_resume_checkpoint(training_root: Path, target_epochs: int) -> tuple[Path, int]:
    """Resolve the newest complete checkpoint below the requested final epoch."""
    output_root = (training_root / "output").resolve()
    if (output_root / "final").is_dir():
        raise TrainingResumeError("Training already has a final model and does not need resuming.")
    checkpoints_root = (output_root / "checkpoints").resolve()
    if not checkpoints_root.is_dir():
        raise TrainingResumeError("Training has no checkpoint directory to resume from.")

    candidates: list[tuple[int, Path]] = []
    for candidate in checkpoints_root.iterdir():
        match = ResumeCheckpointPattern.fullmatch(candidate.name)
        resolved = candidate.resolve()
        if match is None or not candidate.is_dir() or resolved.parent != checkpoints_root:
            continue
        epoch = int(match.group(1))
        if (
            epoch < target_epochs
            and (resolved / "training_state.pt").is_file()
            and (resolved / "adapter/adapter_config.json").is_file()
            and any((resolved / "adapter").glob("adapter_model.*"))
        ):
            candidates.append((epoch, resolved))
    if not candidates:
        raise TrainingResumeError("Training has no complete checkpoint below the target epoch.")
    epoch, checkpoint = max(candidates, key=lambda item: item[0])
    return checkpoint, epoch


@train_app.command("prepare")
def prepare_dataset(
    artist_id: Annotated[str, typer.Argument(help="Explicit authorized artist profile.")],
    tag: Annotated[
        str,
        typer.Option("--tag", help="Unique activation tag used in every training caption."),
    ],
    caption: Annotated[
        str,
        typer.Option("--caption", help="Truthful musical description shared by the songs."),
    ],
    instrumental: Annotated[
        bool,
        typer.Option("--instrumental/--vocal", help="Whether every song is instrumental."),
    ] = False,
    lyrics_dir: Annotated[
        Path | None,
        typer.Option("--lyrics-dir", help="UTF-8 lyrics named after each source song."),
    ] = None,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Build an immutable labeled dataset from all verified catalog recordings."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        artists.get(artist_id)
        rights = artists.get_rights(artist_id)
        artist_root = paths.artist_root(artist_id)
        catalog = CatalogRepository(artist_root, artist_id).load()
        dataset = TrainingRepository(artist_root, artist_id).prepare_dataset(
            catalog,
            rights,
            tag,
            caption,
            instrumental,
            lyrics_dir,
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Prepared {dataset.dataset_id} with {len(dataset.songs)} songs.")


@train_app.command("run")
def run_training(
    artist_id: Annotated[str, typer.Argument(help="Explicit authorized artist profile.")],
    dataset_id: Annotated[str, typer.Argument(help="Prepared dataset identifier.")],
    epochs: Annotated[
        int,
        typer.Option("--epochs", min=1, max=10_000, help="LoRA training epochs."),
    ] = 10,
    rank: Annotated[
        int,
        typer.Option("--rank", min=1, max=256, help="LoRA rank; 16 is the 8 GB default."),
    ] = 16,
    save_every: Annotated[
        int,
        typer.Option("--save-every", min=1, help="Checkpoint interval in epochs."),
    ] = 5,
    wait: Annotated[
        bool,
        typer.Option("--wait/--no-wait", help="Wait, export, and verify the adapter."),
    ] = True,
    select: Annotated[
        bool,
        typer.Option("--select", help="Explicitly select the completed adapter for generation."),
    ] = False,
    ace_url: Annotated[
        str,
        typer.Option("--ace-url", help="Loopback ACE-Step API base URL."),
    ] = DefaultAceUrl,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Preprocess a prepared dataset and start official ACE-Step LoRA training."""
    if select and not wait:
        typer.echo("Error: --select requires --wait so the adapter can be verified.", err=True)
        raise typer.Exit(1)
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    training_root: Path | None = None
    try:
        rights = artists.get_rights(artist_id)
        require_capability(rights, artist_id, "training")
        artist_root = paths.artist_root(artist_id)
        repository = TrainingRepository(artist_root, artist_id)
        dataset = repository.load_dataset(dataset_id)
        training_id = repository.new_training_id()
        training_root = repository.training_root(training_id)
        training_root.mkdir(parents=True)
        tensor_dir = training_root / "tensors"
        output_dir = training_root / "output"
        adapter_dir = training_root / "adapter"
        request = MusicTrainingRequest(
            artist_id=artist_id,
            epochs=epochs,
            rank=rank,
            save_every=save_every,
        )
        dump_json_atomic(
            training_root / "training.json",
            {
                "schema_version": 1,
                "training_id": training_id,
                "dataset_id": dataset_id,
                "request": request.model_dump(mode="json"),
                "status": "preprocessing",
            },
        )
        engine = AceStepApiEngine(
            base_url=ace_url,
            api_key=os.environ.get("ACESTEP_API_KEY"),
        )
        client = AceStepTrainingClient(engine)
        client.prepare_tensors(dataset, artist_root, tensor_dir, _echo_progress)
        started = client.start(request, tensor_dir, output_dir)
        dump_json_atomic(
            training_root / "training.json",
            {
                "schema_version": 1,
                "training_id": training_id,
                "dataset_id": dataset_id,
                "request": request.model_dump(mode="json"),
                "status": "training",
                "ace_step": started,
            },
        )
        if not wait:
            typer.echo(f"Started {training_id}. Use 'legacy-music train status' to monitor it.")
            return
        final_status = client.wait(_echo_progress)
        exported = client.export(output_dir, adapter_dir)
        dump_json_atomic(
            training_root / "training.json",
            {
                "schema_version": 1,
                "training_id": training_id,
                "dataset_id": dataset_id,
                "request": request.model_dump(mode="json"),
                "status": "complete",
                "ace_step": final_status,
                "export": exported,
            },
        )
        if select:
            repository.select_adapter(training_id, adapter_dir, training_id)
    except Exception as error:
        if training_root is not None and training_root.is_dir():
            try:
                failure_manifest = load_json(training_root / "training.json")
                if isinstance(failure_manifest, dict):
                    failure_manifest["status"] = "failed"
                    failure_manifest["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    dump_json_atomic(training_root / "training.json", failure_manifest)
            except Exception as persistence_error:
                typer.echo(
                    f"Warning: unable to record failed training state: {persistence_error}",
                    err=True,
                )
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed {training_id}; adapter: {adapter_dir}")
    if select:
        typer.echo("Selected this adapter for artist generation.")


@train_app.command("resume")
def resume_training(
    artist_id: Annotated[str, typer.Argument(help="Explicit authorized artist profile.")],
    training_id: Annotated[str, typer.Argument(help="Paused training identifier.")],
    select: Annotated[
        bool,
        typer.Option("--select", help="Select the verified final adapter for generation."),
    ] = False,
    ace_url: Annotated[
        str,
        typer.Option("--ace-url", help="Loopback ACE-Step API base URL."),
    ] = DefaultAceUrl,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Continue a paused LoRA run from its newest verified optimizer checkpoint."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    training_root: Path | None = None
    manifest: dict[str, object] | None = None
    try:
        rights = artists.get_rights(artist_id)
        require_capability(rights, artist_id, "training")
        repository = TrainingRepository(paths.artist_root(artist_id), artist_id)
        training_root = repository.training_root(training_id)
        payload = load_json(training_root / "training.json")
        if not isinstance(payload, dict):
            raise TrainingResumeError("Training lineage manifest is invalid.")
        manifest = payload
        if manifest.get("training_id") != training_id:
            raise TrainingResumeError("Training identity does not match its directory.")
        dataset_id = manifest.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise TrainingResumeError("Training has no dataset lineage.")
        repository.load_dataset(dataset_id)
        request = MusicTrainingRequest.model_validate(manifest.get("request"))
        if request.artist_id != artist_id:
            raise TrainingResumeError("Training request belongs to another artist.")
        checkpoint, checkpoint_epoch = _latest_resume_checkpoint(
            training_root,
            request.epochs,
        )
        resume_history = manifest.get("resume_history", [])
        if not isinstance(resume_history, list):
            raise TrainingResumeError("Training resume history is invalid.")
        resume_history = [
            *resume_history,
            {
                "checkpoint": checkpoint.relative_to(training_root).as_posix(),
                "from_epoch": checkpoint_epoch,
                "resumed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
        ]
        manifest = {
            **manifest,
            "status": "resuming",
            "resume_history": resume_history,
        }
        dump_json_atomic(training_root / "training.json", manifest)

        engine = AceStepApiEngine(
            base_url=ace_url,
            api_key=os.environ.get("ACESTEP_API_KEY"),
        )
        client = AceStepTrainingClient(engine)
        tensor_dir = training_root / "tensors"
        output_dir = training_root / "output"
        adapter_dir = training_root / "adapter"
        started = client.start(
            request,
            tensor_dir,
            output_dir,
            resume_from=checkpoint,
        )
        manifest = {**manifest, "status": "training", "ace_step": started}
        dump_json_atomic(training_root / "training.json", manifest)
        final_status = client.wait(_echo_progress)
        exported = client.export(output_dir, adapter_dir)
        was_stopped = bool(final_status.get("should_stop")) or "stopped" in str(
            final_status.get("status", "")
        ).casefold()
        manifest = {
            **manifest,
            "status": "paused" if was_stopped else "complete",
            "ace_step": final_status,
            "export": exported,
        }
        dump_json_atomic(training_root / "training.json", manifest)
        if select:
            repository.select_adapter(training_id, adapter_dir, training_id)
    except Exception as error:
        if training_root is not None and manifest is not None:
            try:
                dump_json_atomic(
                    training_root / "training.json",
                    {
                        **manifest,
                        "status": "resume_failed",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                )
            except Exception as persistence_error:
                typer.echo(
                    f"Warning: unable to record resume failure: {persistence_error}",
                    err=True,
                )
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Resumed {training_id} from epoch {checkpoint_epoch}.")
    if was_stopped:
        typer.echo("Training paused again after exporting its latest checkpoint.")
    else:
        typer.echo(f"Completed {training_id}; adapter: {adapter_dir}")
    if select:
        typer.echo("Selected this adapter for artist generation.")


@train_app.command("finalize")
def finalize_training(
    artist_id: Annotated[str, typer.Argument(help="Explicit authorized artist profile.")],
    training_id: Annotated[str, typer.Argument(help="Completed training identifier.")],
    select: Annotated[
        bool,
        typer.Option("--select", help="Select the exported adapter for generation."),
    ] = False,
    ace_url: Annotated[
        str,
        typer.Option("--ace-url", help="Loopback ACE-Step API base URL."),
    ] = DefaultAceUrl,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Export a completed background training job and optionally select it."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        rights = artists.get_rights(artist_id)
        require_capability(rights, artist_id, "training")
        repository = TrainingRepository(paths.artist_root(artist_id), artist_id)
        training_root = repository.training_root(training_id)
        engine = AceStepApiEngine(
            base_url=ace_url,
            api_key=os.environ.get("ACESTEP_API_KEY"),
        )
        adapter = training_root / "adapter"
        AceStepTrainingClient(engine).export(training_root / "output", adapter)
        if select:
            repository.select_adapter(training_id, adapter, training_id)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Exported adapter for {training_id}: {adapter}")


@train_app.command("select")
def select_adapter(
    artist_id: Annotated[str, typer.Argument(help="Explicit authorized artist profile.")],
    training_id: Annotated[str, typer.Argument(help="Exported training identifier.")],
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Human-select one exported, integrity-checked adapter for future generation."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        rights = artists.get_rights(artist_id)
        require_capability(rights, artist_id, "training")
        repository = TrainingRepository(paths.artist_root(artist_id), artist_id)
        adapter = repository.training_root(training_id) / "adapter"
        selected = repository.select_adapter(training_id, adapter, training_id)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Selected {selected.adapter_id} ({selected.sha256}).")


@train_app.command("status")
def training_status(
    ace_url: Annotated[
        str,
        typer.Option("--ace-url", help="Loopback ACE-Step API base URL."),
    ] = DefaultAceUrl,
) -> None:
    """Print the active official ACE-Step training status."""
    try:
        engine = AceStepApiEngine(
            base_url=ace_url,
            api_key=os.environ.get("ACESTEP_API_KEY"),
        )
        status = AceStepTrainingClient(engine).status()
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(status, indent=2))


def _echo_progress(message: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_message = message.encode(encoding, errors="replace").decode(encoding)
    typer.echo(safe_message)
