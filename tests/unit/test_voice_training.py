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
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.pipeline.voice_training import (
    VoiceTrainingDatasetBuilder,
    VoiceTrainingDatasetError,
)
from legacy_music.utils.hashing import sha256_file


def test_prepare_uses_clean_vocals_only_and_retains_lineage(tmp_path: Path) -> None:
    artist_root, source_sha256 = _create_clean_stem(tmp_path)
    result = VoiceTrainingDatasetBuilder(
        artist_root,
        "example-artist",
    ).prepare(_rights(source_sha256), training_segments_per_song=1)

    manifest = load_json(result.manifest)
    training = [segment for segment in result.segments if segment.split == "train"]
    assert manifest["source_kind"] == "clean_vocals"
    assert manifest["song_count"] == 1
    assert len(training) == 1
    assert training[0].clean_vocals_sha256 == manifest["sources"][0][
        "clean_vocals_sha256"
    ]
    samples, _sample_rate = sf.read(training[0].audio, dtype="float32")
    assert float(np.max(np.abs(samples))) < 0.2


def test_prepare_fails_closed_without_clean_vocal_metadata(tmp_path: Path) -> None:
    artist_root, source_sha256 = _create_clean_stem(tmp_path)
    stem_manifest = next(artist_root.glob("data/derived/stems/*/stems.json"))
    payload = load_json(stem_manifest)
    payload.pop("clean_vocals")
    dump_json_atomic(stem_manifest, payload)

    with pytest.raises(VoiceTrainingDatasetError, match="clean_vocals"):
        VoiceTrainingDatasetBuilder(artist_root, "example-artist").prepare(
            _rights(source_sha256),
            training_segments_per_song=1,
        )


def test_prepare_rejects_every_candidate_overlapping_reviewed_background_range(
    tmp_path: Path,
) -> None:
    artist_root, source_sha256 = _create_clean_stem(tmp_path)
    policy = artist_root / "data/dataset-policy.yaml"
    policy.write_text(
        """
schema_version: 1
artist_id: example-artist
reject_live_voice_recordings: true
source_exclusions: []
voice_songs:
  - song_id: song-one
    exclude_entire_song: false
    excluded_intervals:
      - start_seconds: 0
        end_seconds: 20
        reason: reviewed background vocal
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = VoiceTrainingDatasetBuilder(artist_root, "example-artist").prepare(
        _rights(source_sha256),
        training_segments_per_song=1,
    )

    assert all(segment.start_seconds >= 20 for segment in result.segments)
    manifest = load_json(result.manifest)
    assert manifest["sources"][0]["excluded_intervals"][0]["reason"] == (
        "reviewed background vocal"
    )


def _create_clean_stem(tmp_path: Path) -> tuple[Path, str]:
    artist_root = tmp_path / "artists/example-artist"
    stem_root = artist_root / "data/derived/stems/song-one"
    stem_root.mkdir(parents=True)
    sample_rate = 44100
    seconds = 45
    time = np.arange(sample_rate * seconds, dtype=np.float64) / sample_rate
    wet = np.stack([0.4 * np.sin(2 * np.pi * 180 * time)] * 2, axis=1)
    clean_mono = 0.1 * np.sin(2 * np.pi * 180 * time)
    clean_mono += 0.003 * np.sin(2 * np.pi * 6000 * time)
    clean = np.stack([clean_mono] * 2, axis=1)
    vocals = stem_root / "vocals.wav"
    clean_vocals = stem_root / "clean_vocals.wav"
    f0_path = stem_root / "vocals-f0.npy"
    sf.write(vocals, wet, sample_rate, subtype="PCM_24")
    sf.write(clean_vocals, clean, sample_rate, subtype="PCM_24")
    f0 = np.full(seconds * 50, 180, dtype=np.float32)
    for start in range(0, len(f0), 100):
        f0[start + 75 : start + 100] = 0
    np.save(f0_path, f0)
    source_sha256 = "1" * 64
    dump_json_atomic(
        stem_root / "stems.json",
        {
            "schema_version": 2,
            "artist_id": "example-artist",
            "song_id": "song-one",
            "source_sha256": source_sha256,
            "vocals": {
                "path": vocals.relative_to(artist_root).as_posix(),
                "sha256": sha256_file(vocals),
            },
            "clean_vocals": {
                "path": clean_vocals.relative_to(artist_root).as_posix(),
                "sha256": sha256_file(clean_vocals),
                "source_vocals_sha256": sha256_file(vocals),
            },
            "f0": {
                "path": f0_path.relative_to(artist_root).as_posix(),
                "sha256": sha256_file(f0_path),
            },
        },
    )
    return artist_root, source_sha256


def _rights(source_sha256: str) -> RightsManifest:
    return RightsManifest(
        artist_id="example-artist",
        status=RightsStatus.APPROVED,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        permissions=RightsPermissions(training=True, singing_voice=True),
        approvals=(
            ApprovalRecord(
                holder="test",
                scope="test",
                evidence_reference="test://approval",
                approved_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ),
        source_assets=(
            SourceAssetAuthorization(
                sha256=source_sha256,
                evidence_reference="test://asset",
            ),
        ),
    )
