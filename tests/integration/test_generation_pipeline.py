from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from legacy_music.audio import probe_audio
from legacy_music.domain.generation import (
    GenerationRequest,
    LyricsRequest,
    MusicGenerationRequest,
    PipelineStage,
)
from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
)
from legacy_music.pipeline.generation import GenerationPipeline
from legacy_music.repositories.run import RunRepository
from legacy_music.utils.hashing import sha256_file


class FakeMusicEngine:
    last_metadata = {"dit_model": "fake-ace-step"}

    def generate(self, request: MusicGenerationRequest, lyrics: str, output: Path) -> Path:
        assert request.seed == 42
        assert lyrics == "[Instrumental]"
        sf.write(output, np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_16")
        return output


def approved_rights() -> RightsManifest:
    approved_at = datetime(2026, 1, 1, tzinfo=UTC)
    return RightsManifest(
        artist_id="test-artist",
        status=RightsStatus.APPROVED,
        permissions=RightsPermissions(music_style=True),
        approvals=(
            ApprovalRecord(
                holder="Synthetic fixture owner",
                scope="Synthetic fixture generation",
                evidence_reference="test://approval",
                approved_at=approved_at,
            ),
        ),
    )


def test_execute_instrumental_writes_audio_provenance_and_complete_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Instrumental]", encoding="utf-8")
    request = GenerationRequest(
        id="generation-test",
        artist_id="test-artist",
        music=MusicGenerationRequest(prompt="Original test music", duration_seconds=10, seed=42),
        lyrics=LyricsRequest(file=lyrics, sha256=sha256_file(lyrics)),
    )
    repository = RunRepository(tmp_path / "runs")
    run_id = repository.new_run_id(datetime(2026, 8, 24, tzinfo=UTC))
    monkeypatch.setattr("legacy_music.pipeline.generation.probe_audio", lambda _path: None)

    result = GenerationPipeline(repository, FakeMusicEngine()).execute(
        run_id,
        request,
        approved_rights(),
    )

    assert result.stage is PipelineStage.COMPLETE
    assert result.final_audio is not None and result.final_audio.read_bytes()
    final_properties = probe_audio(result.final_audio)
    assert (final_properties.sample_rate, final_properties.channels) == (48000, 2)
    assert final_properties.codec == "PCM_24"
    assert result.provenance is not None and result.provenance.is_file()
    assert repository.load(run_id).stage is PipelineStage.COMPLETE
