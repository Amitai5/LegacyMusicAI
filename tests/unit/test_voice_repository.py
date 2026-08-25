import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
    SourceAssetAuthorization,
)
from legacy_music.repositories.voice import (
    VoiceReferenceRepository,
    VoiceReferenceRepositoryError,
)
from legacy_music.utils.hashing import sha256_file


class CopyAudioService:
    def prepare_voice_reference(self, source: Path, output: Path) -> Path:
        shutil.copyfile(source, output)
        return output


class FakeF0Extractor:
    def extract_f0(self, audio: Path, output: Path) -> Path:
        assert audio.is_file()
        output.write_bytes(b"synthetic-f0")
        return output


def approved_voice_rights(source_hash: str) -> RightsManifest:
    approved_at = datetime(2026, 1, 1, tzinfo=UTC)
    return RightsManifest(
        artist_id="test-artist",
        status=RightsStatus.APPROVED,
        permissions=RightsPermissions(singing_voice=True),
        approvals=(
            ApprovalRecord(
                holder="Synthetic fixture owner",
                scope="Synthetic voice fixture",
                evidence_reference="test://approval",
                approved_at=approved_at,
            ),
        ),
        source_assets=(
            SourceAssetAuthorization(
                sha256=source_hash,
                evidence_reference="test://source",
            ),
        ),
    )


def test_create_and_resolve_voice_reference_validates_all_hashes(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/test-artist"
    (artist_root / "voice/references").mkdir(parents=True)
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2400, dtype=np.float32), 24000, subtype="PCM_16")
    repository = VoiceReferenceRepository(artist_root, "test-artist")

    resolved = repository.create(
        "neutral",
        source,
        approved_voice_rights(sha256_file(source)),
        FakeF0Extractor(),
        CopyAudioService(),
        tags=("neutral",),
    )

    assert resolved.reference.id == "neutral"
    assert resolved.reference.source_sha256 == sha256_file(source)
    assert resolved.audio.is_file()
    assert resolved.f0.read_bytes() == b"synthetic-f0"

    resolved.audio.write_bytes(b"tampered")
    with pytest.raises(VoiceReferenceRepositoryError, match="integrity"):
        repository.resolve("neutral")
