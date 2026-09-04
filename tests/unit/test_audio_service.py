from pathlib import Path

import numpy as np
import soundfile as sf

from legacy_music.audio.ffmpeg import FfmpegAudioService, probe_audio
from legacy_music.persistence import load_json


def write_tone(path: Path, sample_rate: int, duration: float, channels: int = 1) -> None:
    samples = np.arange(round(sample_rate * duration)) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)
    if channels == 2:
        audio = np.column_stack((audio, audio))
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def test_soundfile_fallback_normalizes_and_mixes_lossless_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.wav"
    write_tone(source, 24000, 0.25)
    monkeypatch.setattr("legacy_music.audio.ffmpeg.shutil.which", lambda _name: None)
    service = FfmpegAudioService()

    normalized = service.normalize(source, tmp_path / "normalized.wav")
    reference = service.prepare_voice_reference(source, tmp_path / "reference.wav")
    mixed = service.mix(normalized, reference, tmp_path / "mixed.wav")

    normalized_properties = probe_audio(normalized)
    reference_properties = probe_audio(reference)
    mixed_properties = probe_audio(mixed)
    assert (normalized_properties.sample_rate, normalized_properties.channels) == (48000, 2)
    assert (reference_properties.sample_rate, reference_properties.channels) == (24000, 1)
    assert (mixed_properties.sample_rate, mixed_properties.channels) == (48000, 2)
    assert mixed_properties.codec == "PCM_24"


def test_match_vocal_level_applies_bounded_smoothed_gain(tmp_path: Path) -> None:
    sample_rate = 24000
    seconds = 4
    time = np.arange(sample_rate * seconds) / sample_rate
    guide = 0.08 * np.sin(2 * np.pi * 220 * time)
    converted = 0.32 * np.sin(2 * np.pi * 220 * time)
    guide_path = tmp_path / "guide.wav"
    converted_path = tmp_path / "converted.wav"
    f0_path = tmp_path / "f0.npy"
    sf.write(guide_path, guide, sample_rate, subtype="PCM_24")
    sf.write(converted_path, converted, sample_rate, subtype="PCM_24")
    np.save(f0_path, np.full(seconds * 50, 220, dtype=np.float32))

    result = FfmpegAudioService().match_vocal_level(
        guide_path,
        converted_path,
        f0_path,
        tmp_path / "leveled.wav",
        local_gain_limit_db=6,
    )

    assert -12.2 <= result.global_gain_db <= -11.8
    assert -6 <= result.local_gain_min_db <= result.local_gain_max_db <= 6
    leveled, _ = sf.read(result.audio, dtype="float32")
    assert np.max(np.abs(leveled)) < 1


def test_mix_and_master_uses_bundled_ffmpeg_and_meets_output_contract(tmp_path: Path) -> None:
    sample_rate = 24000
    seconds = 4
    time = np.arange(sample_rate * seconds) / sample_rate
    accompaniment = tmp_path / "accompaniment.wav"
    vocals = tmp_path / "vocals.wav"
    sf.write(
        accompaniment,
        0.08 * np.sin(2 * np.pi * 110 * time),
        sample_rate,
        subtype="PCM_24",
    )
    sf.write(
        vocals,
        0.04 * np.sin(2 * np.pi * 220 * time),
        sample_rate,
        subtype="PCM_24",
    )
    service = FfmpegAudioService()

    result = service.mix_and_master(
        accompaniment,
        vocals,
        tmp_path / "final.wav",
        compressor_threshold_dbfs=-18,
        compressor_ratio=2,
        compressor_attack_ms=15,
        compressor_release_ms=120,
        target_lufs=-14,
        target_lra=9,
        true_peak_dbfs=-1,
    )

    properties = probe_audio(result.final_audio)
    assert service.ffmpeg is not None
    assert (properties.sample_rate, properties.channels, properties.codec) == (
        48000,
        2,
        "PCM_24",
    )
    assert abs(result.integrated_lufs - -14) <= 0.5
    assert result.true_peak_dbfs <= -1
    assert result.premaster_audio.is_file()
    assert result.vocal_bus_audio.is_file()
    assert result.accompaniment_bus_audio.is_file()
    assert -4 <= result.vocal_to_instrumental_db <= 6
    assert abs(result.vocal_to_instrumental_db - 1.5) <= 0.1
    assert result.automatic_balance_correction_db > 0
    assert result.automatic_vocal_adjustment_db == 0
    assert result.automatic_instrumental_adjustment_db < 0
    assert result.manifest.is_file()
    manifest = load_json(result.manifest)
    assert manifest["settings"]["reverb_wet"] == 0.08
    assert manifest["settings"]["vocal_presence_gain_db"] == 1.5
    assert manifest["settings"]["automatic_vocal_balance"] is True
    assert manifest["settings"]["target_vocal_to_instrumental_db"] == 1.5
    assert manifest["vocal_bus_sha256"]
