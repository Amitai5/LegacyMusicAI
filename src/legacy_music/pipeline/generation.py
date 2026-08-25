"""Authorization-gated durable end-to-end generation orchestration."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from legacy_music.audio import FfmpegAudioService, probe_audio
from legacy_music.authorization import require_capability, require_source_asset
from legacy_music.domain.generation import (
    GenerationRequest,
    GenerationResult,
    PipelineStage,
)
from legacy_music.domain.rights import RightsManifest
from legacy_music.domain.run import ProvenanceManifest
from legacy_music.domain.voice import ResolvedVoiceReference, VoiceConversionRequest
from legacy_music.engines.base import MusicEngine, StemSeparator, VoiceEngine
from legacy_music.persistence import dump_json_atomic
from legacy_music.repositories.run import RunRepository
from legacy_music.utils.hashing import sha256_file


class GenerationPipelineError(RuntimeError):
    """Raised after a generation failure has been durably recorded."""


class GenerationPipeline:
    """Run stock or voice-converted music generation with durable transitions."""

    def __init__(
        self,
        run_repository: RunRepository,
        music_engine: MusicEngine,
        stem_separator: StemSeparator | None = None,
        voice_engine: VoiceEngine | None = None,
        audio_service: FfmpegAudioService | None = None,
    ) -> None:
        self.runs = run_repository
        self.music_engine = music_engine
        self.stem_separator = stem_separator
        self.voice_engine = voice_engine
        self.audio = audio_service or FfmpegAudioService()

    def execute(
        self,
        run_id: str,
        request: GenerationRequest,
        rights: RightsManifest,
        voice_reference: ResolvedVoiceReference | None = None,
    ) -> GenerationResult:
        """Execute one generation and return its final lossless artifact."""
        require_capability(rights, request.artist_id, "music_style")
        lyrics_source = request.lyrics.file
        if not lyrics_source.is_file():
            raise GenerationPipelineError(f"Lyrics file does not exist: {lyrics_source}")
        lyrics_hash = sha256_file(lyrics_source)
        if request.lyrics.sha256 is not None and request.lyrics.sha256 != lyrics_hash:
            raise GenerationPipelineError("Lyrics changed after the request was created.")

        manifest = self.runs.create(run_id, request)
        try:
            run_root = self.runs.root(run_id)
            lyrics_path = run_root / "inputs/lyrics.txt"
            shutil.copyfile(lyrics_source, lyrics_path)
            draft = run_root / "intermediates/ace-step-draft.wav"
            final_audio = run_root / "output/final.wav"
            lyrics = lyrics_path.read_text(encoding="utf-8")
            self.music_engine.generate(request.music, lyrics, draft)
            probe_audio(draft)
            manifest = self.runs.transition(
                run_id,
                PipelineStage.MUSIC_GENERATED,
                "ACE-Step draft generated and validated.",
                artifacts={"draft": "intermediates/ace-step-draft.wav"},
                hashes={"draft": sha256_file(draft)},
            )

            reference_hash = None
            if request.voice.enabled:
                require_capability(rights, request.artist_id, "singing_voice")
                if self.stem_separator is None or self.voice_engine is None:
                    raise RuntimeError(
                        "Voice generation requires separator and voice engine adapters."
                    )
                if voice_reference is None:
                    raise RuntimeError(
                        "Voice generation requires an explicit authorized reference."
                    )
                require_source_asset(rights, voice_reference.reference.source_sha256)
                reference_hash = sha256_file(voice_reference.audio)
                if reference_hash != voice_reference.reference.audio_sha256:
                    raise RuntimeError("Voice reference audio failed its integrity check.")
                if sha256_file(voice_reference.f0) != voice_reference.reference.f0_sha256:
                    raise RuntimeError("Voice reference F0 failed its integrity check.")
                stems = self.stem_separator.separate(draft, run_root / "intermediates/stems")
                if stems.f0 is None:
                    raise RuntimeError("The separator did not create a target F0 contour.")
                manifest = self.runs.transition(
                    run_id,
                    PipelineStage.VOCALS_SEPARATED,
                    "Draft vocals and accompaniment separated.",
                    artifacts={
                        "vocals": str(stems.vocals.relative_to(run_root).as_posix()),
                        "accompaniment": str(stems.accompaniment.relative_to(run_root).as_posix()),
                        "target_f0": str(stems.f0.relative_to(run_root).as_posix()),
                    },
                    hashes={"target_f0": sha256_file(stems.f0)},
                )
                converted = run_root / "intermediates/converted-vocals.wav"
                conversion = VoiceConversionRequest(
                    target_audio=stems.vocals,
                    target_f0=stems.f0,
                    reference_audio=voice_reference.audio,
                    reference_f0=voice_reference.f0,
                    reference_id=voice_reference.reference.id,
                    engine=request.voice.engine,
                )
                self.voice_engine.convert(conversion, converted)
                manifest = self.runs.transition(
                    run_id,
                    PipelineStage.VOICE_CONVERTED,
                    "Authorized singing voice conversion completed.",
                    artifacts={"converted_vocals": "intermediates/converted-vocals.wav"},
                    hashes={"converted_vocals": sha256_file(converted)},
                )
                self.audio.mix(stems.accompaniment, converted, final_audio)
                manifest = self.runs.transition(
                    run_id,
                    PipelineStage.MIXED,
                    "Converted vocals mixed with the original accompaniment timing.",
                    artifacts={"final": "output/final.wav"},
                    hashes={"final": sha256_file(final_audio)},
                )
            else:
                self.audio.normalize(draft, final_audio)

            metadata = getattr(self.music_engine, "last_metadata", {})
            provenance = ProvenanceManifest(
                run_id=run_id,
                artist_id=request.artist_id,
                created_at=datetime.now(UTC),
                prompt=request.music.prompt,
                lyrics_sha256=lyrics_hash,
                seed=request.music.seed,
                music_engine="ace-step",
                music_model=metadata.get("dit_model"),
                music_adapter=request.music.adapter,
                voice_engine=request.voice.engine if request.voice.enabled else None,
                voice_reference_id=(
                    voice_reference.reference.id if voice_reference is not None else None
                ),
                voice_reference_sha256=reference_hash,
                voice_source_sha256=(
                    voice_reference.reference.source_sha256
                    if voice_reference is not None
                    else None
                ),
                voice_reference_f0_sha256=(
                    voice_reference.reference.f0_sha256
                    if voice_reference is not None
                    else None
                ),
                parent_run_id=request.parent_run_id,
                output=Path("output/final.wav"),
                output_sha256=sha256_file(final_audio),
            )
            provenance_path = run_root / "output/provenance.json"
            dump_json_atomic(provenance_path, provenance.model_dump(mode="json"))
            manifest = self.runs.transition(
                run_id,
                PipelineStage.COMPLETE,
                "Generation completed; output remains blocked from automatic distribution.",
                artifacts={
                    "final": "output/final.wav",
                    "provenance": "output/provenance.json",
                },
                hashes={"final": provenance.output_sha256},
            )
            return GenerationResult(
                run_id=run_id,
                stage=manifest.stage,
                final_audio=final_audio,
                provenance=provenance_path,
            )
        except Exception as error:
            if manifest.stage not in {PipelineStage.COMPLETE, PipelineStage.FAILED}:
                self.runs.transition(
                    run_id,
                    PipelineStage.FAILED,
                    "Generation failed; prior artifacts remain available for inspection.",
                    error=str(error),
                )
            raise GenerationPipelineError(str(error)) from error
