"""FFmpeg-backed audio inspection, normalization, and mixing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from legacy_music.domain.catalog import AudioProperties


class AudioProcessingError(RuntimeError):
    """Raised when an external audio operation fails validation."""


def probe_audio(path: Path, ffprobe: str = "ffprobe") -> AudioProperties:
    """Validate an audio file and return stable media properties."""
    if not path.is_file():
        raise AudioProcessingError(f"Audio file does not exist: {path}")
    if shutil.which(ffprobe) is None:
        return _probe_with_soundfile(path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "FFprobe rejected the file."
        raise AudioProcessingError(f"Invalid audio '{path.name}': {detail}")
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        media_format = payload["format"]
        return AudioProperties(
            duration_seconds=float(media_format["duration"]),
            sample_rate=int(stream["sample_rate"]),
            channels=int(stream["channels"]),
            codec=stream["codec_name"],
            format_name=media_format["format_name"],
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioProcessingError(
            f"FFprobe returned incomplete audio metadata: {error}"
        ) from error


class FfmpegAudioService:
    """Write derived audio while never modifying a source path in place."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def normalize(self, source: Path, output: Path) -> Path:
        """Decode to a consistent lossless stereo format without loudness mastering."""
        if shutil.which(self.ffmpeg) is None:
            return self._render_with_soundfile(source, output, 48000, 2, "PCM_24")
        return self._render(
            source,
            output,
            ["-map", "0:a:0", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le"],
        )

    def prepare_voice_reference(self, source: Path, output: Path) -> Path:
        """Create the 24 kHz mono PCM reference expected by SoulX-Singer."""
        if shutil.which(self.ffmpeg) is None:
            return self._render_with_soundfile(source, output, 24000, 1, "PCM_16")
        return self._render(
            source,
            output,
            ["-map", "0:a:0", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le"],
        )

    def mix(self, accompaniment: Path, vocals: Path, output: Path) -> Path:
        """Align and mix accompaniment with converted vocals under a peak limiter."""
        if accompaniment.resolve() == output.resolve() or vocals.resolve() == output.resolve():
            raise AudioProcessingError("Mix output must not replace either source.")
        if shutil.which(self.ffmpeg) is None:
            return self._mix_with_soundfile(accompaniment, vocals, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(accompaniment),
            "-i",
            str(vocals),
            "-filter_complex",
            "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[out]",
            "-map",
            "[out]",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s24le",
            str(temporary),
        ]
        self._run(command, temporary)
        os.replace(temporary, output)
        probe_audio(output, self.ffprobe)
        return output

    def _render(self, source: Path, output: Path, arguments: list[str]) -> Path:
        if source.resolve() == output.resolve():
            raise AudioProcessingError("Derived audio output must not replace its source.")
        probe_audio(source, self.ffprobe)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}{output.suffix}")
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            *arguments,
            str(temporary),
        ]
        self._run(command, temporary)
        os.replace(temporary, output)
        probe_audio(output, self.ffprobe)
        return output

    @staticmethod
    def _run(command: list[str], temporary: Path) -> None:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            temporary.unlink(missing_ok=True)
            detail = completed.stderr.strip() or "FFmpeg failed without diagnostics."
            raise AudioProcessingError(detail)

    def _render_with_soundfile(
        self,
        source: Path,
        output: Path,
        sample_rate: int,
        channels: int,
        subtype: str,
    ) -> Path:
        if source.resolve() == output.resolve():
            raise AudioProcessingError("Derived audio output must not replace its source.")
        probe_audio(source, self.ffprobe)
        audio, source_rate = _read_audio(source)
        audio = _resample(audio, source_rate, sample_rate)
        audio = _set_channels(audio, channels)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}{output.suffix}")
        try:
            sf.write(temporary, audio, sample_rate, subtype=subtype)
            os.replace(temporary, output)
        except (OSError, RuntimeError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise AudioProcessingError(f"Unable to render lossless audio: {error}") from error
        probe_audio(output, self.ffprobe)
        return output

    def _mix_with_soundfile(
        self,
        accompaniment: Path,
        vocals: Path,
        output: Path,
    ) -> Path:
        accompaniment_audio, accompaniment_rate = _read_audio(accompaniment)
        vocal_audio, vocal_rate = _read_audio(vocals)
        accompaniment_audio = _set_channels(
            _resample(accompaniment_audio, accompaniment_rate, 48000),
            2,
        )
        vocal_audio = _set_channels(_resample(vocal_audio, vocal_rate, 48000), 2)
        length = max(len(accompaniment_audio), len(vocal_audio))
        mixed = np.zeros((length, 2), dtype=np.float32)
        mixed[: len(accompaniment_audio)] += accompaniment_audio
        mixed[: len(vocal_audio)] += vocal_audio
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
        if peak > 0.95:
            mixed *= 0.95 / peak

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
        try:
            sf.write(temporary, mixed, 48000, subtype="PCM_24")
            os.replace(temporary, output)
        except (OSError, RuntimeError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise AudioProcessingError(f"Unable to mix lossless audio: {error}") from error
        probe_audio(output, self.ffprobe)
        return output


def _probe_with_soundfile(path: Path) -> AudioProperties:
    try:
        with sf.SoundFile(path) as stream:
            if stream.frames <= 0 or stream.samplerate <= 0 or stream.channels <= 0:
                raise AudioProcessingError(f"Audio file contains no decodable samples: {path}")
            return AudioProperties(
                duration_seconds=stream.frames / stream.samplerate,
                sample_rate=stream.samplerate,
                channels=stream.channels,
                codec=stream.subtype or "unknown",
                format_name=stream.format or path.suffix.lstrip(".") or "unknown",
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise AudioProcessingError(f"Invalid audio '{path.name}': {error}") from error


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise AudioProcessingError(f"Unable to decode '{path.name}': {error}") from error
    if audio.size == 0:
        raise AudioProcessingError(f"Audio file contains no samples: {path}")
    return audio, sample_rate


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    target_frames = max(1, round(len(audio) * target_rate / source_rate))
    source_positions = np.arange(len(audio), dtype=np.float64)
    target_positions = np.linspace(0, len(audio) - 1, target_frames, dtype=np.float64)
    channels = [
        np.interp(target_positions, source_positions, audio[:, index])
        for index in range(audio.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def _set_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return audio[:, :2]
