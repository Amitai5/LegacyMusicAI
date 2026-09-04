from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from legacy_music.authorization import AuthorizationError
from legacy_music.domain.rights import (
    ApprovalRecord,
    RightsManifest,
    RightsPermissions,
    RightsStatus,
    SourceAssetAuthorization,
)
from legacy_music.persistence import dump_json_atomic, load_json
from legacy_music.pipeline.clean_vocals import CleanVocalError, CleanVocalPipeline
from legacy_music.utils.hashing import sha256_file


class FakeDereverbRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Path, Path], ...]] = []

    def dereverb_batch(
        self,
        jobs: tuple[tuple[Path, Path], ...],
        *,
        strength: float,
    ) -> tuple[dict[str, object], ...]:
        self.calls.append(jobs)
        results = []
        for source, output in jobs:
            audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
            sf.write(output, audio * 0.9, sample_rate, subtype="PCM_24")
            results.append(
                {
                    "clipped_sample_count": 0,
                    "nonfinite_sample_count": 0,
                    "active_level_match_db": 0.5,
                    "peak_reduction_db": 0.0,
                    "active_rms_dbfs": -18.0,
                    "peak_dbfs": -3.0,
                    "dereverb_strength": strength,
                    "source_restoration_ratio": 1 - strength,
                }
            )
        return tuple(results)


def test_execute_creates_clean_pair_without_changing_original(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/example-artist"
    source_hashes = _create_stems(artist_root, 2)
    rights = _rights(tuple(source_hashes))
    runtime = FakeDereverbRuntime()
    pipeline = CleanVocalPipeline(artist_root, "example-artist", runtime)
    originals = {
        path: sha256_file(path)
        for path in artist_root.glob("data/derived/stems/*/vocals.wav")
    }

    result = pipeline.execute(rights, model_sha256="b" * 64, config_sha256="c" * 64)

    assert len(result.records) == 2
    assert len(runtime.calls) == 1
    assert len(runtime.calls[0]) == 2
    for original, digest in originals.items():
        assert sha256_file(original) == digest
        clean = original.with_name("clean_vocals.wav")
        assert clean.is_file()
        manifest = load_json(original.with_name("stems.json"))
        assert manifest["vocals"]["sha256"] == digest
        assert manifest["clean_vocals"]["path"].endswith("/clean_vocals.wav")
        assert manifest["clean_vocals"]["source_vocals_sha256"] == digest
        assert manifest["vocal_cleaning"]["model_sha256"] == "b" * 64
        assert manifest["vocal_cleaning"]["dereverb_strength"] == 0.9
        assert manifest["vocal_cleaning"]["source_restoration_ratio"] == pytest.approx(0.1)


def test_execute_reuses_only_hash_matching_clean_derivatives(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/example-artist"
    source_hashes = _create_stems(artist_root, 1)
    rights = _rights(tuple(source_hashes))
    first_runtime = FakeDereverbRuntime()
    pipeline = CleanVocalPipeline(artist_root, "example-artist", first_runtime)
    pipeline.execute(rights, model_sha256="b" * 64, config_sha256="c" * 64)
    second_runtime = FakeDereverbRuntime()

    result = CleanVocalPipeline(artist_root, "example-artist", second_runtime).execute(
        rights,
        model_sha256="b" * 64,
        config_sha256="c" * 64,
    )

    assert second_runtime.calls == []
    assert result.records[0].status == "reused"


def test_execute_denies_missing_voice_training_authorization(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/example-artist"
    source_hashes = _create_stems(artist_root, 1)
    rights = _rights(tuple(source_hashes), singing_voice=False)

    with pytest.raises(AuthorizationError, match="singing_voice"):
        CleanVocalPipeline(artist_root, "example-artist", FakeDereverbRuntime()).execute(
            rights,
            model_sha256="b" * 64,
            config_sha256="c" * 64,
        )


def test_execute_rebuilds_tracked_derivative_when_strength_changes(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/example-artist"
    source_hashes = _create_stems(artist_root, 1)
    rights = _rights(tuple(source_hashes))
    pipeline = CleanVocalPipeline(artist_root, "example-artist", FakeDereverbRuntime())
    pipeline.execute(
        rights,
        model_sha256="b" * 64,
        config_sha256="c" * 64,
        dereverb_strength=1.0,
    )
    vocals = next(artist_root.glob("data/derived/stems/*/vocals.wav"))
    original_hash = sha256_file(vocals)

    result = CleanVocalPipeline(
        artist_root,
        "example-artist",
        FakeDereverbRuntime(),
    ).execute(
        rights,
        model_sha256="b" * 64,
        config_sha256="c" * 64,
        dereverb_strength=0.9,
        rebuild=True,
    )

    assert result.records[0].status == "created"
    assert sha256_file(vocals) == original_hash
    manifest = load_json(vocals.with_name("stems.json"))
    assert manifest["vocal_cleaning"]["dereverb_strength"] == 0.9


def test_execute_refuses_untracked_clean_vocal_overwrite(tmp_path: Path) -> None:
    artist_root = tmp_path / "artists/example-artist"
    source_hashes = _create_stems(artist_root, 1)
    original = next(artist_root.glob("data/derived/stems/*/vocals.wav"))
    audio, sample_rate = sf.read(original, dtype="float32", always_2d=True)
    sf.write(original.with_name("clean_vocals.wav"), audio, sample_rate)

    with pytest.raises(CleanVocalError, match="Untracked"):
        CleanVocalPipeline(artist_root, "example-artist", FakeDereverbRuntime()).execute(
            _rights(tuple(source_hashes)),
            model_sha256="b" * 64,
            config_sha256="c" * 64,
        )


def _create_stems(artist_root: Path, count: int) -> list[str]:
    source_hashes = []
    for index in range(count):
        song_id = f"song-{index}"
        stem_root = artist_root / "data/derived/stems" / song_id
        stem_root.mkdir(parents=True)
        audio = np.full((4410, 2), 0.1 + index * 0.01, dtype=np.float32)
        vocals = stem_root / "vocals.wav"
        sf.write(vocals, audio, 44100, subtype="PCM_24")
        vocals_sha256 = sha256_file(vocals)
        source_sha256 = f"{index + 1:064x}"
        source_hashes.append(source_sha256)
        dump_json_atomic(
            stem_root / "stems.json",
            {
                "schema_version": 1,
                "artist_id": "example-artist",
                "song_id": song_id,
                "source_sha256": source_sha256,
                "normalized_sha256": "a" * 64,
                "vocals": {
                    "path": vocals.relative_to(artist_root).as_posix(),
                    "sha256": vocals_sha256,
                    "duration_seconds": 0.1,
                },
                "accompaniment": {
                    "path": f"data/derived/stems/{song_id}/accompaniment.wav",
                    "sha256": "d" * 64,
                    "duration_seconds": 0.1,
                },
                "f0": {
                    "path": f"data/derived/stems/{song_id}/vocals-f0.npy",
                    "sha256": "e" * 64,
                },
            },
        )
    return source_hashes


def _rights(
    source_hashes: tuple[str, ...],
    *,
    singing_voice: bool = True,
) -> RightsManifest:
    return RightsManifest(
        artist_id="example-artist",
        status=RightsStatus.APPROVED,
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
        permissions=RightsPermissions(training=True, singing_voice=singing_voice),
        approvals=(
            ApprovalRecord(
                holder="test",
                scope="test",
                evidence_reference="test://approval",
                approved_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        ),
        source_assets=tuple(
            SourceAssetAuthorization(sha256=digest, evidence_reference="test://asset")
            for digest in source_hashes
        ),
    )
