from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from legacy_music.audio import FfmpegAudioService, probe_audio
from legacy_music.authorization import AuthorizationError
from legacy_music.config import VoiceQualityConfig
from legacy_music.domain.generation import (
    GenerationRequest,
    LyricsRequest,
    MusicGenerationRequest,
    PipelineStage,
    VoiceGenerationRequest,
)
from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
    SourceAssetAuthorization,
)
from legacy_music.domain.voice import (
    ResolvedVoiceReference,
    VoiceCandidateMetrics,
    VoiceConversionMetrics,
    VoiceConversionRequest,
    VoiceConversionResult,
    VoicePhraseDecision,
    VoiceReference,
    VoiceReferenceMetrics,
)
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.pipeline.remix import VocalRemixPipeline
from legacy_music.repositories.run import RunRepository
from legacy_music.utils.hashing import sha256_file


class FakeChunkedVoiceEngine:
    def __init__(self, expected_reference_id: str) -> None:
        self.expected_reference_id = expected_reference_id

    def convert(
        self,
        request: VoiceConversionRequest,
        output: Path,
    ) -> VoiceConversionResult:
        assert tuple(item.reference.id for item in request.references) == (
            self.expected_reference_id,
        )
        guide, sample_rate = sf.read(request.target_audio, dtype="float32")
        time = np.arange(len(guide)) / sample_rate
        converted = 0.25 * np.sin(2 * np.pi * 220 * time)
        converted[len(converted) // 3 : 2 * len(converted) // 3] *= 0.2
        sf.write(output, converted, sample_rate, subtype="PCM_24")
        cleaned_f0 = output.with_name("converted-vocals-cleaned-f0.npy")
        values = np.load(request.target_f0, allow_pickle=False)
        np.save(cleaned_f0, values)
        candidate = VoiceCandidateMetrics(
            candidate_index=0,
            seed=request.seed,
            passed=True,
            raw_peak_dbfs=-12,
            peak_reduction_db=0,
            sustained_dropout_count=0,
            dropout_fraction=0,
            spectral_distance_db=1,
            f0_median_error_cents=0,
            f0_gross_error_fraction=0,
        )
        phrases = (
            VoicePhraseDecision(
                index=0,
                start_seconds=0,
                end_seconds=6,
                reference_id=self.expected_reference_id,
                selected_candidate_index=0,
                candidates=(candidate,),
            ),
            VoicePhraseDecision(
                index=1,
                start_seconds=5,
                end_seconds=10,
                reference_id=self.expected_reference_id,
                selected_candidate_index=0,
                candidates=(candidate,),
            ),
        )
        metrics = VoiceConversionMetrics(
            phrase_count=2,
            selected_reference_ids=(self.expected_reference_id,),
            sustained_dropout_count=0,
            peak_dbfs=-12,
            clipping_fraction=0,
            max_voiced_join_jump_db=2,
            f0_median_error_cents=0,
            f0_gross_error_fraction=0,
        )
        manifest = output.parent / "voice-conversion.json"
        dump_json_atomic(
            manifest,
            {
                "schema_version": 2,
                "reference_audio_hashes": {
                    item.reference.id: item.reference.audio_sha256
                    for item in request.references
                },
                "private_source_paths": [],
            },
        )
        return VoiceConversionResult(
            audio=output,
            manifest=manifest,
            cleaned_f0=cleaned_f0,
            reference_ids=(self.expected_reference_id,),
            phrases=phrases,
            metrics=metrics,
        )


def write_tone(path: Path, frequency: float, amplitude: float, seconds: int = 10) -> None:
    sample_rate = 24000
    time = np.arange(sample_rate * seconds) / sample_rate
    sf.write(
        path,
        amplitude * np.sin(2 * np.pi * frequency * time),
        sample_rate,
        subtype="PCM_24",
    )


def create_complete_parent(repository: RunRepository, tmp_path: Path) -> tuple[str, dict[str, str]]:
    lyrics_source = tmp_path / "lyrics.txt"
    lyrics_source.write_text("Synthetic lyrics", encoding="utf-8")
    request = GenerationRequest(
        id="generation-parent",
        artist_id="test-artist",
        music=MusicGenerationRequest(
            prompt="Synthetic vocal quality integration fixture",
            duration_seconds=10,
            seed=42,
        ),
        lyrics=LyricsRequest(file=lyrics_source, sha256=sha256_file(lyrics_source)),
        voice=VoiceGenerationRequest(enabled=True, reference="neutral"),
    )
    run_id = repository.new_run_id(datetime(2026, 8, 25, tzinfo=UTC))
    repository.create(run_id, request)
    root = repository.root(run_id)
    lyrics = root / "inputs/lyrics.txt"
    draft = root / "intermediates/ace-step-draft.wav"
    vocals = root / "intermediates/stems/vocals.wav"
    accompaniment = root / "intermediates/stems/accompaniment.wav"
    target_f0 = root / "intermediates/stems/vocals-f0.npy"
    vocals.parent.mkdir(parents=True, exist_ok=True)
    lyrics.write_text("Synthetic lyrics", encoding="utf-8")
    write_tone(draft, 110, 0.08)
    write_tone(vocals, 220, 0.08)
    write_tone(accompaniment, 110, 0.06)
    np.save(target_f0, np.full(500, 220, dtype=np.float32))
    hashes = {
        "lyrics": sha256_file(lyrics),
        "draft": sha256_file(draft),
        "vocals": sha256_file(vocals),
        "accompaniment": sha256_file(accompaniment),
        "target_f0": sha256_file(target_f0),
    }
    repository.transition(
        run_id,
        PipelineStage.MUSIC_GENERATED,
        "Synthetic draft complete.",
        artifacts={"draft": "intermediates/ace-step-draft.wav"},
        hashes={"draft": hashes["draft"]},
    )
    repository.transition(
        run_id,
        PipelineStage.VOCALS_SEPARATED,
        "Synthetic stems complete.",
        artifacts={
            "vocals": "intermediates/stems/vocals.wav",
            "accompaniment": "intermediates/stems/accompaniment.wav",
            "target_f0": "intermediates/stems/vocals-f0.npy",
        },
        hashes={name: hashes[name] for name in ("vocals", "accompaniment", "target_f0")},
    )
    repository.transition(run_id, PipelineStage.VOICE_CONVERTED, "Legacy voice complete.")
    repository.transition(run_id, PipelineStage.MIXED, "Legacy mix complete.")
    repository.transition(run_id, PipelineStage.COMPLETE, "Legacy run complete.")
    dump_json_atomic(root / "output/provenance.json", {"music_model": "fake-ace-step"})
    return run_id, hashes


def create_reference(tmp_path: Path) -> ResolvedVoiceReference:
    source = tmp_path / "reference-source.wav"
    audio = tmp_path / "reference-audio.wav"
    f0 = tmp_path / "reference-f0.npy"
    write_tone(source, 220, 0.08, seconds=1)
    write_tone(audio, 220, 0.08, seconds=1)
    np.save(f0, np.full(50, 220, dtype=np.float32))
    reference = VoiceReference(
        id="strong",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_file=Path("voice/references/strong/source.wav"),
        source_sha256=sha256_file(source),
        audio_file=Path("voice/references/strong/audio.wav"),
        audio_sha256=sha256_file(audio),
        f0_file=Path("voice/references/strong/f0.npy"),
        f0_sha256=sha256_file(f0),
        metrics=VoiceReferenceMetrics(
            duration_seconds=1,
            voiced_fraction=1,
            median_f0_hz=220,
            f0_p05_hz=220,
            f0_p95_hz=220,
            active_rms_dbfs=-22,
            one_second_level_range_db=0,
            peak_dbfs=-20,
            clipping_fraction=0,
            high_band_ratio_db=-30,
        ),
    )
    return ResolvedVoiceReference(reference=reference, source=source, audio=audio, f0=f0)


def approved_rights(source_hash: str, singing_voice: bool = True) -> RightsManifest:
    return RightsManifest(
        artist_id="test-artist",
        status=RightsStatus.APPROVED,
        permissions=RightsPermissions(music_style=True, singing_voice=singing_voice),
        approvals=(
            ApprovalRecord(
                holder="Synthetic fixture owner",
                scope="Synthetic remix fixture",
                evidence_reference="test://approval",
                approved_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ),
        source_assets=(
            SourceAssetAuthorization(
                sha256=source_hash,
                evidence_reference="test://source",
            ),
        ),
    )


def test_remix_inconsistent_fake_chunks_preserves_parent_and_meets_contract(
    tmp_path: Path,
) -> None:
    repository = RunRepository(tmp_path / "runs")
    parent_run_id, original_hashes = create_complete_parent(repository, tmp_path)
    resolved = create_reference(tmp_path)
    quality = VoiceQualityConfig(phrase_min_seconds=5, phrase_max_seconds=6)
    pipeline = VocalRemixPipeline(
        repository,
        FakeChunkedVoiceEngine("strong"),
        quality,
        FfmpegAudioService(),
    )

    result = pipeline.execute(
        parent_run_id,
        approved_rights(resolved.reference.source_sha256),
        (resolved,),
        reference_label="strong",
    )

    assert result.stage is PipelineStage.COMPLETE
    assert repository.load(parent_run_id).stage is PipelineStage.COMPLETE
    parent_root = repository.root(parent_run_id)
    assert sha256_file(parent_root / "intermediates/ace-step-draft.wav") == original_hashes["draft"]
    child = repository.load(result.run_id)
    assert child.request.parent_run_id == parent_run_id
    assert child.request.voice.reference == "strong"
    assert result.final_audio is not None
    properties = probe_audio(result.final_audio)
    assert abs(properties.duration_seconds - 10) <= 0.02
    final_samples, _ = sf.read(result.final_audio, dtype="float32")
    assert not np.any(np.abs(final_samples) >= 0.999999)
    qc = load_json(repository.root(result.run_id) / "output/quality-control.json")
    assert qc["accepted"] is True
    assert qc["checks"]["phrase_coverage"] is True
    provenance = load_json(repository.root(result.run_id) / "output/provenance.json")
    assert provenance["parent_run_id"] == parent_run_id
    assert provenance["release_approved"] is False


def test_remix_rejects_expired_voice_authorization_before_creating_child(
    tmp_path: Path,
) -> None:
    repository = RunRepository(tmp_path / "runs")
    parent_run_id, _hashes = create_complete_parent(repository, tmp_path)
    resolved = create_reference(tmp_path)
    pipeline = VocalRemixPipeline(
        repository,
        FakeChunkedVoiceEngine("strong"),
        VoiceQualityConfig(phrase_min_seconds=5, phrase_max_seconds=6),
    )

    with pytest.raises(AuthorizationError, match="singing_voice"):
        pipeline.execute(
            parent_run_id,
            approved_rights(resolved.reference.source_sha256, singing_voice=False),
            (resolved,),
            reference_label="strong",
        )

    assert tuple(run.run_id for run in repository.list()) == (parent_run_id,)
