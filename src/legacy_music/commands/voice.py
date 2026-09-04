"""Authorized artist-specific singing-voice reference commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from legacy_music.audio import FfmpegAudioService
from legacy_music.config import load_app_config
from legacy_music.engines.seed_vc import (
    SeedVCInstallation,
    SeedVCTrainingRunner,
    SeedVCVoiceEngine,
)
from legacy_music.engines.soulx import SoulXInstallation, SoulXRuntime, SoulXVoiceEngine
from legacy_music.paths import ProjectPaths
from legacy_music.pipeline.clean_vocals import CleanVocalPipeline
from legacy_music.pipeline.remix import VocalRemixPipeline
from legacy_music.pipeline.stem_preparation import CatalogStemPipeline
from legacy_music.pipeline.voice_bank import VoiceBankBuilder
from legacy_music.pipeline.voice_training import VoiceTrainingDatasetBuilder
from legacy_music.repositories import ArtistRepository, RunRepository, VoiceReferenceRepository
from legacy_music.utils.hashing import sha256_file

voice_app = typer.Typer(
    help="Curate authorized singing-voice references.",
    no_args_is_help=True,
)
DefaultProjectRoot = Path(".")


def _console_safe_text(message: str, encoding: str | None) -> str:
    """Replace progress glyphs that the active Windows console cannot encode."""
    resolved_encoding = encoding or "utf-8"
    return message.encode(resolved_encoding, errors="replace").decode(resolved_encoding)


@voice_app.command("prepare-stems")
def prepare_stems(
    artist_id: Annotated[str, typer.Argument(help="Authorized artist profile identifier.")],
    soulx_python: Annotated[
        Path | None,
        typer.Option("--soulx-python", help="Override the SoulX Conda Python executable."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device used by the stem-separation models."),
    ] = "cuda:0",
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Create reusable source-linked vocal, accompaniment, and F0 artifacts."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        artists.get(artist_id)
        rights = artists.get_rights(artist_id)
        installation = SoulXInstallation.from_project(paths.root, soulx_python)
        result = CatalogStemPipeline(
            paths.artist_root(artist_id),
            artist_id,
            SoulXRuntime(installation, device=device),
        ).execute(rights)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    created = sum(record.status == "created" for record in result.records)
    reused = sum(record.status == "reused" for record in result.records)
    typer.echo(f"Prepared catalog stems: {created} created, {reused} reused.")


@voice_app.command("clean-vocals")
def clean_vocals(
    artist_id: Annotated[str, typer.Argument(help="Authorized artist profile identifier.")],
    soulx_python: Annotated[
        Path | None,
        typer.Option("--soulx-python", help="Override the SoulX Conda Python executable."),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device used by the dereverberation model."),
    ] = "cuda:0",
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Atomically replace tracked clean derivatives when cleanup settings changed.",
        ),
    ] = False,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Create provenance-tracked clean_vocals.wav derivatives for every vocal stem."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        artists.get(artist_id)
        rights = artists.get_rights(artist_id)
        config = load_app_config(paths.config / "app.yaml", paths.config / "local.yaml")
        installation = SoulXInstallation.from_project(paths.root, soulx_python)
        installation.validate_dereverb()
        result = CleanVocalPipeline(
            paths.artist_root(artist_id),
            artist_id,
            SoulXRuntime(installation, device=device),
        ).execute(
            rights,
            model_sha256=sha256_file(installation.dereverb_model),
            config_sha256=sha256_file(installation.dereverb_config),
            dereverb_strength=config.training_vocals.dereverb_strength,
            rebuild=rebuild,
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    created = sum(record.status == "created" for record in result.records)
    reused = sum(record.status == "reused" for record in result.records)
    typer.echo(f"Completed {result.cleaning_id}: {created} created, {reused} reused.")
    typer.echo(str(result.manifest))


@voice_app.command("prepare-training")
def prepare_voice_training(
    artist_id: Annotated[str, typer.Argument(help="Authorized artist profile identifier.")],
    segments_per_song: Annotated[
        int,
        typer.Option(
            "--segments-per-song",
            min=1,
            max=20,
            help="Maximum quality-screened training segments retained per song.",
        ),
    ] = 5,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Prepare a clean-vocal-only Seed-VC fine-tuning dataset."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        artists.get(artist_id)
        rights = artists.get_rights(artist_id)
        config = load_app_config(paths.config / "app.yaml", paths.config / "local.yaml")
        result = VoiceTrainingDatasetBuilder(
            paths.artist_root(artist_id),
            artist_id,
            config.training_vocals,
        ).prepare(rights, training_segments_per_song=segments_per_song)
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    training_count = sum(segment.split == "train" for segment in result.segments)
    validation_count = sum(segment.split == "validation" for segment in result.segments)
    typer.echo(
        f"Prepared {result.dataset_id}: {training_count} training and "
        f"{validation_count} validation segments."
    )
    typer.echo(str(result.manifest))


@voice_app.command("train-model")
def train_voice_model(
    artist_id: Annotated[str, typer.Argument(help="Authorized artist profile identifier.")],
    dataset_id: Annotated[str, typer.Argument(help="Prepared clean-vocal dataset identifier.")],
    steps: Annotated[
        int,
        typer.Option("--steps", min=1, max=100_000, help="Seed-VC optimization steps."),
    ] = 1000,
    save_every: Annotated[
        int,
        typer.Option("--save-every", min=1, help="Checkpoint interval in steps."),
    ] = 500,
    select: Annotated[
        bool,
        typer.Option("--select", help="Select the verified checkpoint for regeneration."),
    ] = False,
    seed_vc_python: Annotated[
        Path | None,
        typer.Option("--seed-vc-python", help="Override the Seed-VC Conda Python executable."),
    ] = None,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Fine-tune the pinned Seed-VC singing model using only clean vocal segments."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        artists.get(artist_id)
        rights = artists.get_rights(artist_id)
        installation = SeedVCInstallation.from_project(paths.root, seed_vc_python)
        result = SeedVCTrainingRunner(
            installation,
            paths.artist_root(artist_id),
            artist_id,
        ).train(
            rights,
            dataset_id,
            max_steps=steps,
            save_every=save_every,
            select=select,
            progress=(
                lambda message: (
                    typer.echo(_console_safe_text(message, getattr(sys.stdout, "encoding", None)))
                    if message
                    else None
                )
            ),
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed {result.training_id} ({result.checkpoint_sha256}).")
    typer.echo(str(result.checkpoint))
    typer.echo(str(result.manifest))


@voice_app.command("add-reference")
def add_reference(
    artist_id: Annotated[str, typer.Argument(help="Explicit artist profile identifier.")],
    reference_id: Annotated[str, typer.Argument(help="Canonical reference identifier.")],
    source: Annotated[Path, typer.Argument(help="Exact rights-cleared vocal source file.")],
    tags: Annotated[
        str,
        typer.Option("--tags", help="Comma-separated register, energy, or style tags."),
    ] = "",
    soulx_python: Annotated[
        Path | None,
        typer.Option("--soulx-python", help="Override the SoulX Conda Python executable."),
    ] = None,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Add one hash-authorized source and precompute its SoulX F0 contour."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        profile = artists.get(artist_id)
        if not profile.capabilities.singing_voice:
            raise ValueError("Singing voice is disabled in this artist profile.")
        rights = artists.get_rights(artist_id)
        installation = SoulXInstallation.from_project(paths.root, soulx_python)
        runtime = SoulXRuntime(installation)
        repository = VoiceReferenceRepository(paths.artist_root(artist_id), artist_id)
        resolved = repository.create(
            reference_id,
            source,
            rights,
            runtime,
            tags=tuple(tag.strip() for tag in tags.split(",") if tag.strip()),
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Created authorized voice reference '{resolved.reference.id}' "
        f"({resolved.reference.audio_sha256})."
    )


@voice_app.command("list-references")
def list_references(
    artist_id: Annotated[str, typer.Argument(help="Artist profile to inspect.")],
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """List private reference metadata without exposing source audio."""
    paths = ProjectPaths.from_root(root)
    try:
        ArtistRepository(paths).get(artist_id)
        references = VoiceReferenceRepository(
            paths.artist_root(artist_id),
            artist_id,
        ).list()
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    if json_output:
        typer.echo(
            json.dumps(
                [reference.model_dump(mode="json") for reference in references],
                indent=2,
            )
        )
        return
    table = Table("ID", "Tags", "Created (UTC)", "Audio SHA-256")
    for reference in references:
        table.add_row(
            reference.id,
            ", ".join(reference.tags),
            reference.created_at.isoformat(),
            reference.audio_sha256,
        )
    Console().print(table)


@voice_app.command("build-bank")
def build_bank(
    artist_id: Annotated[str, typer.Argument(help="Authorized artist profile identifier.")],
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Build an approved five-register bank from existing studio vocal stems."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    try:
        profile = artists.get(artist_id)
        if not profile.capabilities.singing_voice:
            raise ValueError("Singing voice is disabled in this artist profile.")
        rights = artists.get_rights(artist_id)
        config = load_app_config(paths.config / "app.yaml", paths.config / "local.yaml")
        result = VoiceBankBuilder(paths.artist_root(artist_id), artist_id).build(
            rights,
            FfmpegAudioService(config.audio.ffmpeg_executable),
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        "Built voice bank: " + ", ".join(reference.reference.id for reference in result.references)
    )
    typer.echo(str(result.manifest))


@voice_app.command("remix-run")
def remix_run(
    run_id: Annotated[str, typer.Argument(help="Complete parent run to vocal-remix.")],
    reference: Annotated[
        str,
        typer.Option(
            "--reference",
            help="Approved reference ID; auto selects the closest bank member per phrase.",
        ),
    ] = "auto",
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            help="Voice engine: soulx or the selected artist-fine-tuned seed-vc model.",
        ),
    ] = "soulx",
    duration: Annotated[
        int | None,
        typer.Option(
            "--duration",
            min=10,
            max=600,
            help="Optional prefix length for a GPU smoke test; defaults to the full parent.",
        ),
    ] = None,
    soulx_python: Annotated[
        Path | None,
        typer.Option("--soulx-python", help="Override the SoulX Conda Python executable."),
    ] = None,
    soulx_steps: Annotated[
        int,
        typer.Option("--soulx-steps", min=2, max=100, help="SoulX diffusion steps."),
    ] = 32,
    seed_vc_python: Annotated[
        Path | None,
        typer.Option("--seed-vc-python", help="Override the Seed-VC Conda Python executable."),
    ] = None,
    seed_vc_steps: Annotated[
        int,
        typer.Option("--seed-vc-steps", min=1, max=100, help="Seed-VC diffusion steps."),
    ] = 30,
    identity_gated: Annotated[
        bool,
        typer.Option(
            "--identity-gated/--whole-song",
            help="Select and score multiple Seed-VC candidates for each register-matched phrase.",
        ),
    ] = False,
    identity_candidates: Annotated[
        int | None,
        typer.Option(
            "--identity-candidates",
            min=2,
            max=5,
            help="Override the configured Seed-VC candidates generated per phrase.",
        ),
    ] = None,
    identity_threshold: Annotated[
        float | None,
        typer.Option(
            "--identity-threshold",
            min=-1.0,
            max=1.0,
            help="Override automatic CAMPPlus identity-threshold calibration.",
        ),
    ] = None,
    vocal_presence_db: Annotated[
        float | None,
        typer.Option(
            "--vocal-presence-db",
            min=-6.0,
            max=6.0,
            help="Per-song final vocal-bus gain override in dB.",
        ),
    ] = None,
    vocal_balance_target_db: Annotated[
        float | None,
        typer.Option(
            "--vocal-balance-target-db",
            min=-4.0,
            max=6.0,
            help="Per-song automatic active-vocal balance target in dB.",
        ),
    ] = None,
    instrumental_gain_db: Annotated[
        float | None,
        typer.Option(
            "--instrumental-gain-db",
            min=-6.0,
            max=3.0,
            help="Per-song final instrumental-bus gain override in dB.",
        ),
    ] = None,
    vocal_reverb_wet: Annotated[
        float | None,
        typer.Option(
            "--vocal-reverb-wet",
            min=0.0,
            max=0.3,
            help="Per-song restored vocal reverb wet mix; zero keeps the final vocal dry.",
        ),
    ] = None,
    vocal_reverb_pre_delay_ms: Annotated[
        float | None,
        typer.Option(
            "--vocal-reverb-pre-delay-ms",
            min=0.0,
            max=100.0,
            help="Per-song restored vocal reverb pre-delay.",
        ),
    ] = None,
    vocal_reverb_decay: Annotated[
        float | None,
        typer.Option(
            "--vocal-reverb-decay",
            min=0.0,
            max=0.8,
            help="Per-song restored vocal reverb decay factor.",
        ),
    ] = None,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Create an immutable quality-first vocal-remix child run."""
    paths = ProjectPaths.from_root(root)
    runs = RunRepository(paths.runs)
    artists = ArtistRepository(paths)
    try:
        parent = runs.load(run_id)
        profile = artists.get(parent.artist_id)
        if not profile.capabilities.singing_voice:
            raise ValueError("Singing voice is disabled in this artist profile.")
        rights = artists.get_rights(parent.artist_id)
        config = load_app_config(paths.config / "app.yaml", paths.config / "local.yaml")
        audio = FfmpegAudioService(config.audio.ffmpeg_executable)
        mix_updates = {
            key: value
            for key, value in {
                "vocal_presence_gain_db": vocal_presence_db,
                "target_vocal_to_instrumental_db": vocal_balance_target_db,
                "instrumental_gain_db": instrumental_gain_db,
                "reverb_wet": vocal_reverb_wet,
                "reverb_pre_delay_ms": vocal_reverb_pre_delay_ms,
                "reverb_decay": vocal_reverb_decay,
            }.items()
            if value is not None
        }
        mix_config = config.final_vocal_mix.model_copy(update=mix_updates)
        reference_repository = VoiceReferenceRepository(
            paths.artist_root(parent.artist_id),
            parent.artist_id,
        )
        if reference == "auto":
            references = (
                VoiceBankBuilder(
                    paths.artist_root(parent.artist_id),
                    parent.artist_id,
                )
                .build(rights, audio)
                .references
            )
        else:
            references = (reference_repository.resolve(reference),)
        if engine == "soulx":
            if identity_gated:
                raise ValueError("--identity-gated currently requires --engine seed-vc.")
            installation = SoulXInstallation.from_project(paths.root, soulx_python)
            voice_engine = SoulXVoiceEngine(
                installation,
                steps=soulx_steps,
                quality_config=config.voice_quality,
            )
        elif engine == "seed-vc":
            seed_installation = SeedVCInstallation.from_project(paths.root, seed_vc_python)
            soulx_installation = SoulXInstallation.from_project(paths.root, soulx_python)
            voice_engine = SeedVCVoiceEngine(
                seed_installation,
                paths.artist_root(parent.artist_id),
                parent.artist_id,
                SoulXRuntime(soulx_installation),
                diffusion_steps=seed_vc_steps,
                inference_cfg_rate=(
                    config.voice_quality.identity_inference_cfg_rate if identity_gated else 0.7
                ),
                quality_config=(config.voice_quality if identity_gated else None),
                identity_candidates=identity_candidates,
                identity_threshold=identity_threshold,
            )
        else:
            raise ValueError("Voice engine must be 'soulx' or 'seed-vc'.")
        result = VocalRemixPipeline(
            runs,
            voice_engine,
            config.voice_quality,
            audio,
            mix_config=mix_config,
        ).execute(
            run_id,
            rights,
            references,
            reference_label=reference,
            duration_seconds=duration,
            voice_engine_id=engine,
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed {result.run_id}")
    typer.echo(str(result.final_audio))
    typer.echo(str(result.provenance))
