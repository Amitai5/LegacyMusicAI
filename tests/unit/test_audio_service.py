from pathlib import Path

import numpy as np
import soundfile as sf

from legacy_music.audio.ffmpeg import FfmpegAudioService, probe_audio


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
