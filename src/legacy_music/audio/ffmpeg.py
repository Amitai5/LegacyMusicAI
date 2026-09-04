"""FFmpeg-backed audio inspection, normalization, and mixing."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import soundfile as sf

from legacy_music.domain.catalog import AudioProperties
from legacy_music.persistence import dump_json_atomic
from legacy_music.utils.hashing import sha256_file


class AudioProcessingError(RuntimeError):
    """Raised when an external audio operation fails validation."""


@dataclass(frozen=True, slots=True)
class VocalLevelMatchResult:
    """Leveled vocal artifact and reproducible gain measurements."""

    audio: Path
    global_gain_db: float
    local_gain_min_db: float
    local_gain_max_db: float
    peak_dbfs: float


@dataclass(frozen=True, slots=True)
class AudioMasteringResult:
    """Final mastered audio and its retained evidence artifacts."""

    final_audio: Path
    premaster_audio: Path
    vocal_bus_audio: Path
    accompaniment_bus_audio: Path
    manifest: Path
    integrated_lufs: float
    loudness_range_lu: float
    true_peak_dbfs: float
    vocal_to_instrumental_db: float
    pre_balance_vocal_to_instrumental_db: float
    automatic_balance_correction_db: float
    automatic_vocal_adjustment_db: float
    automatic_instrumental_adjustment_db: float


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
            codec=_normalized_codec(str(stream["codec_name"])),
            format_name=media_format["format_name"],
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AudioProcessingError(
            f"FFprobe returned incomplete audio metadata: {error}"
        ) from error


class FfmpegAudioService:
    """Write derived audio while never modifying a source path in place."""

    def __init__(
        self,
        ffmpeg: str | Path | None = None,
        ffprobe: str = "ffprobe",
    ) -> None:
        self.ffmpeg = _resolve_ffmpeg(ffmpeg)
        self.ffprobe = ffprobe

    def normalize(self, source: Path, output: Path) -> Path:
        """Decode to a consistent lossless stereo format without loudness mastering."""
        if self.ffmpeg is None:
            return self._render_with_soundfile(source, output, 48000, 2, "PCM_24")
        return self._render(
            source,
            output,
            ["-map", "0:a:0", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le"],
        )

    def prepare_voice_reference(self, source: Path, output: Path) -> Path:
        """Create the 24 kHz mono PCM reference expected by SoulX-Singer."""
        if self.ffmpeg is None:
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
        if self.ffmpeg is None:
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

    def match_vocal_level(
        self,
        guide_vocals: Path,
        converted_vocals: Path,
        target_f0: Path,
        output: Path,
        local_gain_limit_db: float = 6.0,
    ) -> VocalLevelMatchResult:
        """Match converted-vocal dynamics to the guide with smoothed gain automation."""
        if output.resolve() in {guide_vocals.resolve(), converted_vocals.resolve()}:
            raise AudioProcessingError("Leveled vocal output must not replace a source.")
        guide, guide_rate = _read_audio(guide_vocals)
        converted, converted_rate = _read_audio(converted_vocals)
        guide_mono = np.mean(guide, axis=1, dtype=np.float32)
        converted_mono = np.mean(converted, axis=1, dtype=np.float32)
        try:
            f0 = np.load(target_f0, allow_pickle=False).astype(np.float32).reshape(-1)
        except (OSError, ValueError) as error:
            raise AudioProcessingError(f"Unable to load target F0: {error}") from error
        if f0.size == 0 or not np.any(f0 > 0):
            raise AudioProcessingError("Target F0 contains no voiced frames.")

        guide_rms = _aligned_frame_rms(guide_mono, guide_rate, len(f0))
        converted_rms = _aligned_frame_rms(converted_mono, converted_rate, len(f0))
        usable = min(len(f0), len(guide_rms), len(converted_rms))
        voiced = f0[:usable] > 0
        valid = voiced & (guide_rms[:usable] > 1e-8) & (converted_rms[:usable] > 1e-8)
        if not np.any(valid):
            raise AudioProcessingError("No voiced frames were available for gain matching.")

        difference_db = _db(guide_rms[:usable]) - _db(converted_rms[:usable])
        global_gain_db = float(np.median(difference_db[valid]))
        residual_db = np.zeros(usable, dtype=np.float64)
        residual_db[valid] = np.clip(
            difference_db[valid] - global_gain_db,
            -local_gain_limit_db,
            local_gain_limit_db,
        )
        smoothing_frames = 21
        kernel = np.hanning(smoothing_frames)
        kernel /= np.sum(kernel)
        smoothed_db = np.convolve(residual_db, kernel, mode="same")
        frame_positions = np.linspace(0, len(converted_mono) - 1, usable)
        sample_positions = np.arange(len(converted_mono), dtype=np.float64)
        local_db = np.interp(sample_positions, frame_positions, smoothed_db)
        gain = np.power(10.0, (global_gain_db + local_db) / 20.0)
        leveled = converted_mono.astype(np.float64) * gain
        peak = float(np.max(np.abs(leveled)))
        peak_ceiling = 10 ** (-3 / 20)
        if peak > peak_ceiling:
            reduction_db = 20 * np.log10(peak / peak_ceiling)
            leveled *= peak_ceiling / peak
            global_gain_db -= float(reduction_db)
            peak = peak_ceiling

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
        try:
            sf.write(
                temporary,
                leveled.astype(np.float32),
                converted_rate,
                subtype="PCM_24",
            )
            os.replace(temporary, output)
        except (OSError, RuntimeError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise AudioProcessingError(f"Unable to write leveled vocals: {error}") from error
        probe_audio(output, self.ffprobe)
        return VocalLevelMatchResult(
            audio=output,
            global_gain_db=global_gain_db,
            local_gain_min_db=float(np.min(smoothed_db[valid])),
            local_gain_max_db=float(np.max(smoothed_db[valid])),
            peak_dbfs=float(_db(peak)),
        )

    def mix_and_master(
        self,
        accompaniment: Path,
        leveled_vocals: Path,
        output: Path,
        *,
        compressor_threshold_dbfs: float,
        compressor_ratio: float,
        compressor_attack_ms: float,
        compressor_release_ms: float,
        target_lufs: float,
        target_lra: float,
        true_peak_dbfs: float,
        vocal_presence_gain_db: float = 1.5,
        instrumental_gain_db: float = -0.75,
        presence_eq_frequency_hz: float = 2800.0,
        presence_eq_gain_db: float = 1.5,
        reverb_wet: float = 0.08,
        reverb_pre_delay_ms: float = 24.0,
        reverb_decay: float = 0.22,
        automatic_vocal_balance: bool = True,
        target_vocal_to_instrumental_db: float = 1.5,
        maximum_automatic_balance_correction_db: float = 12.0,
    ) -> AudioMasteringResult:
        """Mix stems and perform deterministic two-pass EBU R128 mastering."""
        if self.ffmpeg is None:
            raise AudioProcessingError(
                "Quality mastering requires FFmpeg; install the locked imageio-ffmpeg dependency."
            )
        if output.resolve() in {accompaniment.resolve(), leveled_vocals.resolve()}:
            raise AudioProcessingError("Master output must not replace a source stem.")
        probe_audio(accompaniment, self.ffprobe)
        probe_audio(leveled_vocals, self.ffprobe)
        if not 0 <= reverb_wet <= 0.3:
            raise AudioProcessingError("Final vocal reverb wet mix must be between 0 and 0.3.")
        if not 0 <= reverb_decay <= 0.8:
            raise AudioProcessingError("Final vocal reverb decay must be between 0 and 0.8.")
        if maximum_automatic_balance_correction_db < 0:
            raise AudioProcessingError("Automatic vocal-balance correction cannot be negative.")
        output.parent.mkdir(parents=True, exist_ok=True)
        premaster = output.parent / "premaster.wav"
        vocal_bus = output.parent / "vocal-bus.wav"
        accompaniment_bus = output.parent / "accompaniment-bus.wav"
        threshold = 10 ** (compressor_threshold_dbfs / 20)
        limiter = 10 ** (true_peak_dbfs / 20)
        vocal_base = (
            "[0:a]aresample=48000:resampler=swr:filter_size=64:phase_shift=10:"
            "linear_interp=0,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            "highpass=f=75,"
            f"equalizer=f={presence_eq_frequency_hz}:t=q:w=1:g={presence_eq_gain_db},"
            f"volume={vocal_presence_gain_db}dB,"
            f"acompressor=threshold={threshold:.9f}:ratio={compressor_ratio}:"
            f"attack={compressor_attack_ms}:release={compressor_release_ms}:makeup=1"
        )
        if reverb_wet > 0:
            delays = (
                max(1.0, reverb_pre_delay_ms),
                max(2.0, reverb_pre_delay_ms + 23.0),
                max(3.0, reverb_pre_delay_ms + 47.0),
            )
            decays = (reverb_decay, reverb_decay * 0.65, reverb_decay * 0.4)
            delay_text = "|".join(f"{value:.3f}" for value in delays)
            decay_text = "|".join(f"{value:.6f}" for value in decays)
            vocal_filter = (
                f"{vocal_base}[base];"
                "[base]asplit=2[dry][wetin];"
                f"[wetin]aecho=0:1:{delay_text}:{decay_text},"
                f"volume={reverb_wet:.6f}[wet];"
                "[dry][wet]amix=inputs=2:duration=first:normalize=0,"
                "alimiter=limit=0.95:attack=5:release=50:level=disabled[vocal]"
            )
        else:
            vocal_filter = f"{vocal_base}[vocal]"
        self._run_external(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(leveled_vocals),
                "-filter_complex",
                vocal_filter,
                "-map",
                "[vocal]",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
                str(vocal_bus),
            ],
            vocal_bus,
        )
        self._render(
            accompaniment,
            accompaniment_bus,
            [
                "-map",
                "0:a:0",
                "-af",
                (
                    "aresample=48000:resampler=swr:filter_size=64:phase_shift=10:"
                    f"linear_interp=0,volume={instrumental_gain_db}dB"
                ),
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
            ],
        )
        pre_balance_vocal_to_instrumental_db = _active_vocal_balance_db(
            vocal_bus,
            accompaniment_bus,
        )
        automatic_balance_correction_db = 0.0
        automatic_vocal_adjustment_db = 0.0
        automatic_instrumental_adjustment_db = 0.0
        if automatic_vocal_balance:
            requested_correction = (
                target_vocal_to_instrumental_db - pre_balance_vocal_to_instrumental_db
            )
            automatic_balance_correction_db = float(
                np.clip(
                    requested_correction,
                    -maximum_automatic_balance_correction_db,
                    maximum_automatic_balance_correction_db,
                )
            )
            if automatic_balance_correction_db < -0.01:
                automatic_vocal_adjustment_db = automatic_balance_correction_db
                self._attenuate_bus(vocal_bus, automatic_vocal_adjustment_db)
            elif automatic_balance_correction_db > 0.01:
                automatic_instrumental_adjustment_db = -automatic_balance_correction_db
                self._attenuate_bus(
                    accompaniment_bus,
                    automatic_instrumental_adjustment_db,
                )
        mix_filter = (
            "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0,"
            f"alimiter=limit={limiter:.9f}:attack=5:release=50:level=disabled[mix]"
        )
        self._run_external(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(accompaniment_bus),
                "-i",
                str(vocal_bus),
                "-filter_complex",
                mix_filter,
                "-map",
                "[mix]",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
                str(premaster),
            ],
            premaster,
        )
        vocal_to_instrumental_db = _active_vocal_balance_db(vocal_bus, accompaniment_bus)

        analysis_filter = (
            f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={true_peak_dbfs}:"
            "print_format=json"
        )
        first_pass = subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                str(premaster),
                "-af",
                analysis_filter,
                "-f",
                "null",
                "NUL" if os.name == "nt" else "/dev/null",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if first_pass.returncode != 0:
            raise AudioProcessingError(first_pass.stderr.strip() or "FFmpeg loudness scan failed.")
        measured = _parse_loudnorm(first_pass.stderr)
        second_filter = (
            f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={true_peak_dbfs}:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:"
            "linear=true:print_format=json"
        )
        temporary = output.with_name(f".{output.stem}.{uuid4().hex}.wav")
        second_pass = subprocess.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-nostats",
                "-y",
                "-i",
                str(premaster),
                "-af",
                second_filter,
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if second_pass.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            raise AudioProcessingError(
                second_pass.stderr.strip() or "FFmpeg loudness mastering failed."
            )
        mastered = _parse_loudnorm(second_pass.stderr)
        os.replace(temporary, output)
        probe_audio(output, self.ffprobe)
        integrated_lufs = float(mastered["output_i"])
        loudness_range = float(mastered["output_lra"])
        true_peak = float(mastered["output_tp"])
        manifest_path = output.parent / "mastering.json"
        dump_json_atomic(
            manifest_path,
            {
                "schema_version": 1,
                "created_at": datetime.now(UTC).isoformat(),
                "settings": {
                    "sample_rate": 48000,
                    "channels": 2,
                    "sample_format": "pcm_s24le",
                    "compressor_threshold_dbfs": compressor_threshold_dbfs,
                    "compressor_ratio": compressor_ratio,
                    "compressor_attack_ms": compressor_attack_ms,
                    "compressor_release_ms": compressor_release_ms,
                    "vocal_presence_gain_db": vocal_presence_gain_db,
                    "instrumental_gain_db": instrumental_gain_db,
                    "presence_eq_frequency_hz": presence_eq_frequency_hz,
                    "presence_eq_gain_db": presence_eq_gain_db,
                    "reverb_wet": reverb_wet,
                    "reverb_pre_delay_ms": reverb_pre_delay_ms,
                    "reverb_decay": reverb_decay,
                    "automatic_vocal_balance": automatic_vocal_balance,
                    "target_vocal_to_instrumental_db": target_vocal_to_instrumental_db,
                    "maximum_automatic_balance_correction_db": (
                        maximum_automatic_balance_correction_db
                    ),
                    "target_lufs": target_lufs,
                    "target_lra": target_lra,
                    "true_peak_dbfs": true_peak_dbfs,
                },
                "first_pass": measured,
                "second_pass": mastered,
                "premaster_sha256": sha256_file(premaster),
                "vocal_bus_sha256": sha256_file(vocal_bus),
                "accompaniment_bus_sha256": sha256_file(accompaniment_bus),
                "pre_balance_vocal_to_instrumental_db": (
                    pre_balance_vocal_to_instrumental_db
                ),
                "automatic_balance_correction_db": automatic_balance_correction_db,
                "automatic_vocal_adjustment_db": automatic_vocal_adjustment_db,
                "automatic_instrumental_adjustment_db": (
                    automatic_instrumental_adjustment_db
                ),
                "vocal_to_instrumental_db": vocal_to_instrumental_db,
                "output_sha256": sha256_file(output),
            },
        )
        return AudioMasteringResult(
            final_audio=output,
            premaster_audio=premaster,
            vocal_bus_audio=vocal_bus,
            accompaniment_bus_audio=accompaniment_bus,
            manifest=manifest_path,
            integrated_lufs=integrated_lufs,
            loudness_range_lu=loudness_range,
            true_peak_dbfs=true_peak,
            vocal_to_instrumental_db=vocal_to_instrumental_db,
            pre_balance_vocal_to_instrumental_db=(
                pre_balance_vocal_to_instrumental_db
            ),
            automatic_balance_correction_db=automatic_balance_correction_db,
            automatic_vocal_adjustment_db=automatic_vocal_adjustment_db,
            automatic_instrumental_adjustment_db=automatic_instrumental_adjustment_db,
        )

    def _attenuate_bus(self, bus: Path, gain_db: float) -> None:
        """Apply a non-positive automatic balance correction to one derived bus."""
        if gain_db > 0:
            raise AudioProcessingError("Automatic bus correction must not boost audio.")
        adjusted = bus.with_name(f".{bus.stem}-balanced-{uuid4().hex}.wav")
        self._render(
            bus,
            adjusted,
            [
                "-map",
                "0:a:0",
                "-af",
                f"volume={gain_db:.6f}dB",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-c:a",
                "pcm_s24le",
            ],
        )
        os.replace(adjusted, bus)

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
    def _run_external(command: list[str], expected: Path) -> None:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not expected.is_file():
            expected.unlink(missing_ok=True)
            detail = completed.stderr.strip() or "FFmpeg failed without diagnostics."
            raise AudioProcessingError(detail)

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


def _normalized_codec(codec: str) -> str:
    """Return stable PCM names across FFprobe and SoundFile backends."""
    return {
        "pcm_s16le": "PCM_16",
        "pcm_s24le": "PCM_24",
        "pcm_s32le": "PCM_32",
        "pcm_f32le": "FLOAT",
        "pcm_f64le": "DOUBLE",
    }.get(codec.casefold(), codec)


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


def _resolve_ffmpeg(candidate: str | Path | None) -> str | None:
    if candidate is not None:
        value = str(candidate)
        path = Path(value)
        if path.is_file():
            return str(path.resolve())
        configured = shutil.which(value)
        if configured is not None:
            return configured
    system = shutil.which("ffmpeg")
    if system is not None:
        return system
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, OSError, RuntimeError):
        return None
    return str(bundled.resolve()) if bundled.is_file() else None


def _aligned_frame_rms(audio: np.ndarray, sample_rate: int, frame_count: int) -> np.ndarray:
    boundaries = np.rint(np.linspace(0, len(audio), frame_count + 1)).astype(np.int64)
    values = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = audio[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            values[index] = np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
    return values


def _active_vocal_balance_db(vocals: Path, accompaniment: Path) -> float:
    """Measure median vocal-to-instrumental balance during active vocal frames."""
    vocal_audio, vocal_rate = _read_audio(vocals)
    accompaniment_audio, accompaniment_rate = _read_audio(accompaniment)
    vocal_mono = np.mean(vocal_audio, axis=1, dtype=np.float32)
    accompaniment_mono = np.mean(accompaniment_audio, axis=1, dtype=np.float32)
    duration = min(
        len(vocal_mono) / vocal_rate,
        len(accompaniment_mono) / accompaniment_rate,
    )
    frame_count = max(1, round(duration * 10))
    vocal_rms = _aligned_frame_rms(vocal_mono, vocal_rate, frame_count)
    accompaniment_rms = _aligned_frame_rms(accompaniment_mono, accompaniment_rate, frame_count)
    active_values = vocal_rms[vocal_rms > 10 ** (-60 / 20)]
    if active_values.size == 0:
        raise AudioProcessingError("Final vocal bus contains no active audio.")
    threshold = max(10 ** (-45 / 20), float(np.percentile(active_values, 60)) * 0.1)
    active = (vocal_rms >= threshold) & (accompaniment_rms > 1e-8)
    if not np.any(active):
        raise AudioProcessingError("No active frames were available for vocal-balance QC.")
    ratios = _db(vocal_rms[active]) - _db(accompaniment_rms[active])
    return float(np.median(ratios))


def _db(value: np.ndarray | float) -> np.ndarray | float:
    converted = 20 * np.log10(np.maximum(value, 1e-12))
    return float(converted) if np.isscalar(value) else converted


def _parse_loudnorm(stderr: str) -> dict[str, Any]:
    matches = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    if not matches:
        raise AudioProcessingError("FFmpeg loudnorm did not return JSON measurements.")
    try:
        payload = json.loads(matches[-1])
    except json.JSONDecodeError as error:
        raise AudioProcessingError(f"Invalid FFmpeg loudnorm measurements: {error}") from error
    required = {
        "input_i",
        "input_tp",
        "input_lra",
        "input_thresh",
        "target_offset",
        "output_i",
        "output_tp",
        "output_lra",
    }
    missing = required.difference(payload)
    if missing:
        raise AudioProcessingError(
            "FFmpeg loudnorm measurements are incomplete: " + ", ".join(sorted(missing))
        )
    return payload
