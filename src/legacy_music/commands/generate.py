"""End-to-end local music generation command."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from legacy_music.domain.generation import (
    GenerationRequest,
    LyricsRequest,
    MusicGenerationRequest,
    OutputRequest,
    VoiceGenerationRequest,
)
from legacy_music.engines.ace_managed import ManagedAceStepEngine
from legacy_music.engines.ace_step import AceStepApiEngine
from legacy_music.engines.soulx import SoulXInstallation, SoulXRuntime, SoulXVoiceEngine
from legacy_music.paths import ProjectPaths
from legacy_music.pipeline.generation import GenerationPipeline
from legacy_music.repositories import (
    ArtistRepository,
    RunRepository,
    TrainingRepository,
    VoiceReferenceRepository,
)
from legacy_music.utils.hashing import sha256_file

DefaultProjectRoot = Path(".")


def generate(
    artist_id: Annotated[str, typer.Argument(help="Explicit authorized artist profile.")],
    prompt: Annotated[str, typer.Option("--prompt", help="Music description for ACE-Step.")],
    lyrics: Annotated[
        Path | None,
        typer.Option("--lyrics", help="UTF-8 lyrics file; defaults to an instrumental marker."),
    ] = None,
    duration: Annotated[
        int,
        typer.Option("--duration", min=10, max=600, help="Target duration in seconds."),
    ] = 30,
    bpm: Annotated[int | None, typer.Option("--bpm", min=30, max=300)] = None,
    key: Annotated[
        str | None,
        typer.Option("--key", help="Key and scale, such as C Major."),
    ] = None,
    vocal_language: Annotated[
        str,
        typer.Option(
            "--vocal-language",
            help="ISO 639 language code for supplied lyrics, such as fa or en.",
        ),
    ] = "en",
    seed: Annotated[int | None, typer.Option("--seed", min=0)] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="ACE-Step model already loaded by the local service."),
    ] = "acestep-v15-turbo",
    adapter: Annotated[
        str,
        typer.Option("--adapter", help="auto, selected, or base; arbitrary paths are rejected."),
    ] = "auto",
    thinking: Annotated[
        bool,
        typer.Option("--thinking", help="Use ACE-Step's optional language-model planning."),
    ] = False,
    voice: Annotated[
        bool,
        typer.Option("--voice", help="Enable authorized singing voice conversion."),
    ] = False,
    reference: Annotated[
        str,
        typer.Option("--reference", help="Curated voice reference ID; auto uses artist default."),
    ] = "auto",
    soulx_python: Annotated[
        Path | None,
        typer.Option("--soulx-python", help="Override the SoulX Conda Python executable."),
    ] = None,
    soulx_steps: Annotated[
        int,
        typer.Option("--soulx-steps", min=2, max=100, help="SoulX diffusion steps."),
    ] = 32,
    ace_url: Annotated[
        str,
        typer.Option("--ace-url", help="Loopback ACE-Step API base URL."),
    ] = "http://127.0.0.1:8001",
    managed_ace: Annotated[
        bool,
        typer.Option(
            "--managed-ace/--external-ace",
            help="Own and stop ACE-Step so the next GPU stage receives all VRAM.",
        ),
    ] = True,
    root: Annotated[
        Path,
        typer.Option("--root", help="Project repository root."),
    ] = DefaultProjectRoot,
) -> None:
    """Generate one durable lossless music artifact through ACE-Step."""
    paths = ProjectPaths.from_root(root)
    artists = ArtistRepository(paths)
    runs = RunRepository(paths.runs)
    run_id = runs.new_run_id()
    lyrics_path = lyrics or paths.root / "examples/instrumental.txt"
    try:
        profile = artists.get(artist_id)
        rights = artists.get_rights(artist_id)
        training = TrainingRepository(paths.artist_root(artist_id), artist_id)
        adapter_path = None
        adapter_label = "base"
        if adapter not in {"auto", "selected", "base"}:
            raise ValueError("--adapter must be auto, selected, or base.")
        if adapter in {"auto", "selected"}:
            selected_manifest = paths.artist_root(artist_id) / "models/music/selected.json"
            if selected_manifest.is_file():
                selected, adapter_path = training.resolve_selected_adapter()
                adapter_label = selected.adapter_id
            elif adapter == "selected":
                raise ValueError("This artist has no selected music adapter.")
        resolved_reference = None
        separator = None
        voice_engine = None
        reference_id = reference
        if voice:
            if not profile.capabilities.singing_voice:
                raise ValueError("Singing voice is disabled in this artist profile.")
            reference_id = (
                profile.voice.default_reference if reference == "auto" else reference
            )
            resolved_reference = VoiceReferenceRepository(
                paths.artist_root(artist_id),
                artist_id,
            ).resolve(reference_id)
            installation = SoulXInstallation.from_project(paths.root, soulx_python)
            separator = SoulXRuntime(installation)
            voice_engine = SoulXVoiceEngine(installation, steps=soulx_steps)
        request = GenerationRequest(
            id=f"generation-{uuid4().hex}",
            artist_id=artist_id,
            music=MusicGenerationRequest(
                prompt=prompt,
                adapter=adapter_label,
                duration_seconds=duration,
                bpm=bpm,
                key=key,
                vocal_language=vocal_language.lower(),
                seed=seed if seed is not None else secrets.randbits(63),
            ),
            lyrics=LyricsRequest(file=lyrics_path, sha256=sha256_file(lyrics_path)),
            voice=VoiceGenerationRequest(
                enabled=voice,
                engine=profile.voice.engine,
                reference=reference_id,
            ),
            output=OutputRequest(),
        )
        engine_arguments = {
            "base_url": ace_url,
            "api_key": os.environ.get("ACESTEP_API_KEY"),
            "model": model,
            "thinking": thinking,
            "adapter_path": adapter_path,
            "adapter_name": adapter_label if adapter_path is not None else None,
            "session_lock": paths.models / "cache/ace-step-api.lock",
        }
        if managed_ace:
            engine = ManagedAceStepEngine(
                project_root=paths.root,
                **engine_arguments,
            )
        else:
            engine = AceStepApiEngine(**engine_arguments)
        result = GenerationPipeline(
            runs,
            engine,
            stem_separator=separator,
            voice_engine=voice_engine,
        ).execute(
            run_id,
            request,
            rights,
            voice_reference=resolved_reference,
        )
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Completed {result.run_id}")
    typer.echo(str(result.final_audio))
    typer.echo(str(result.provenance))
