"""Deterministic vocal analysis, pitch cleanup, phrase planning, and prompt matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from legacy_music.domain.voice import ResolvedVoiceReference, VoiceReferenceMetrics

F0RateHz = 50


class VoiceQualityError(RuntimeError):
    """Raised when vocal quality inputs cannot be analyzed safely."""


@dataclass(frozen=True, slots=True)
class VoicePhrase:
    """One overlapping target interval expressed in F0-frame coordinates."""

    index: int
    start_frame: int
    end_frame: int

    @property
    def start_seconds(self) -> float:
        """Return the phrase start in seconds."""
        return self.start_frame / F0RateHz

    @property
    def end_seconds(self) -> float:
        """Return the phrase end in seconds."""
        return self.end_frame / F0RateHz


def clean_f0(f0: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    """Fill short internal unvoiced gaps and repair isolated octave outliers."""
    values = np.asarray(f0, dtype=np.float32).reshape(-1).copy()
    if maximum_gap_frames < 0:
        raise VoiceQualityError("maximum_gap_frames cannot be negative.")
    if values.size == 0:
        raise VoiceQualityError("F0 contour is empty.")
    values[~np.isfinite(values) | (values < 0)] = 0

    index = 0
    while index < len(values):
        if values[index] > 0:
            index += 1
            continue
        start = index
        while index < len(values) and values[index] <= 0:
            index += 1
        end = index
        if (
            0 < start
            and end < len(values)
            and end - start <= maximum_gap_frames
            and values[start - 1] > 0
            and values[end] > 0
        ):
            endpoint_distance = abs(1200 * np.log2(values[end] / values[start - 1]))
            if endpoint_distance <= 700:
                interpolation = np.geomspace(
                    float(values[start - 1]),
                    float(values[end]),
                    end - start + 2,
                    dtype=np.float64,
                )
                values[start:end] = interpolation[1:-1].astype(np.float32)

    for index in range(1, len(values) - 1):
        previous, current, following = values[index - 1 : index + 2]
        if min(previous, current, following) <= 0:
            continue
        neighbor_distance = abs(1200 * np.log2(following / previous))
        current_distance = min(
            abs(1200 * np.log2(current / previous)),
            abs(1200 * np.log2(current / following)),
        )
        if neighbor_distance <= 100 and current_distance >= 700:
            values[index] = np.sqrt(previous * following)
    return values


def write_clean_f0(source: Path, output: Path, maximum_gap_ms: int) -> Path:
    """Persist a cleaned copy of an immutable target F0 contour."""
    if not source.is_file():
        raise VoiceQualityError(f"F0 contour does not exist: {source}")
    try:
        values = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise VoiceQualityError(f"Unable to load F0 contour: {error}") from error
    maximum_gap_frames = round(maximum_gap_ms * F0RateHz / 1000)
    cleaned = clean_f0(values, maximum_gap_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, cleaned)
    return output


def plan_phrases(
    f0: np.ndarray,
    minimum_seconds: float,
    maximum_seconds: float,
    overlap_seconds: float,
) -> tuple[VoicePhrase, ...]:
    """Cover an F0 contour with uniform, exactly overlapping quality-first phrases."""
    values = np.asarray(f0).reshape(-1)
    if values.size == 0:
        raise VoiceQualityError("Cannot plan phrases for an empty F0 contour.")
    if not 0 < overlap_seconds < minimum_seconds <= maximum_seconds:
        raise VoiceQualityError("Phrase lengths and overlap are inconsistent.")

    duration_seconds = len(values) / F0RateHz
    if duration_seconds <= maximum_seconds:
        return (VoicePhrase(index=0, start_frame=0, end_frame=len(values)),)

    count = int(np.ceil((duration_seconds - overlap_seconds) / (maximum_seconds - overlap_seconds)))
    phrase_seconds = (duration_seconds + (count - 1) * overlap_seconds) / count
    if phrase_seconds < minimum_seconds:
        count = max(
            1,
            int(
                np.floor((duration_seconds - overlap_seconds) / (minimum_seconds - overlap_seconds))
            ),
        )
        phrase_seconds = (duration_seconds + (count - 1) * overlap_seconds) / count
    if not minimum_seconds <= phrase_seconds <= maximum_seconds:
        raise VoiceQualityError("Unable to satisfy configured phrase-length bounds.")

    hop_frames = (phrase_seconds - overlap_seconds) * F0RateHz
    phrases = []
    for index in range(count):
        start = 0 if index == 0 else round(index * hop_frames)
        end = len(values) if index == count - 1 else round(start + phrase_seconds * F0RateHz)
        phrases.append(VoicePhrase(index=index, start_frame=start, end_frame=end))
    return tuple(phrases)


def analyze_reference(audio_path: Path, f0_path: Path) -> VoiceReferenceMetrics:
    """Measure one reference using stable 50 Hz pitch-aligned windows."""
    audio, sample_rate = _read_mono(audio_path)
    try:
        f0 = np.load(f0_path, allow_pickle=False).astype(np.float32).reshape(-1)
    except (OSError, ValueError) as error:
        raise VoiceQualityError(f"Unable to load reference F0: {error}") from error
    return analyze_reference_samples(audio, sample_rate, f0)


def analyze_reference_samples(
    audio: np.ndarray,
    sample_rate: int,
    f0: np.ndarray,
) -> VoiceReferenceMetrics:
    """Measure in-memory prompt audio and its aligned F0 contour."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1 or audio.size == 0 or sample_rate <= 0:
        raise VoiceQualityError("Reference audio samples are invalid.")
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    if f0.size == 0 or not np.any(f0 > 0):
        raise VoiceQualityError("Reference contains no voiced F0 frames.")

    frame_rms = _frame_rms(audio, sample_rate, len(f0))
    usable = min(len(frame_rms), len(f0))
    f0 = f0[:usable]
    frame_rms = frame_rms[:usable]
    voiced = f0 > 0
    voiced_f0 = f0[voiced]
    active_rms = float(np.sqrt(np.mean(np.square(frame_rms[voiced], dtype=np.float64))))

    second_count = usable // F0RateHz
    if second_count:
        second_rms = np.sqrt(
            np.mean(
                np.square(frame_rms[: second_count * F0RateHz], dtype=np.float64).reshape(
                    second_count,
                    F0RateHz,
                ),
                axis=1,
            )
        )
        second_voice = voiced[: second_count * F0RateHz].reshape(
            second_count,
            F0RateHz,
        )
        active_seconds = second_rms[np.mean(second_voice, axis=1) >= 0.2]
    else:
        active_seconds = np.array([active_rms], dtype=np.float64)
    if active_seconds.size == 0:
        raise VoiceQualityError("Reference contains no sustained voiced interval.")
    active_seconds_db = _amplitude_db(active_seconds)

    return VoiceReferenceMetrics(
        duration_seconds=min(len(audio) / sample_rate, len(f0) / F0RateHz),
        voiced_fraction=float(np.mean(voiced)),
        median_f0_hz=float(np.median(voiced_f0)),
        f0_p05_hz=float(np.percentile(voiced_f0, 5)),
        f0_p95_hz=float(np.percentile(voiced_f0, 95)),
        active_rms_dbfs=float(_amplitude_db(active_rms)),
        one_second_level_range_db=float(
            min(
                120,
                np.percentile(active_seconds_db, 95) - np.percentile(active_seconds_db, 5),
            )
        ),
        peak_dbfs=float(_amplitude_db(np.max(np.abs(audio)))),
        clipping_fraction=float(np.mean(np.abs(audio) >= 0.999)),
        high_band_ratio_db=_high_band_ratio(audio, sample_rate),
    )


def select_reference(
    references: tuple[ResolvedVoiceReference, ...],
    target_audio: Path,
    target_f0: np.ndarray,
    phrase: VoicePhrase,
) -> ResolvedVoiceReference:
    """Select the closest approved prompt by phrase register and active energy."""
    return rank_references(references, target_audio, target_f0, phrase)[0]


def rank_references(
    references: tuple[ResolvedVoiceReference, ...],
    target_audio: Path,
    target_f0: np.ndarray,
    phrase: VoicePhrase,
) -> tuple[ResolvedVoiceReference, ...]:
    """Rank approved prompts by phrase register and active energy."""
    if not references:
        raise VoiceQualityError("Voice conversion requires at least one reference.")
    if len(references) == 1:
        return references

    phrase_f0 = np.asarray(target_f0).reshape(-1)[phrase.start_frame : phrase.end_frame]
    voiced = phrase_f0[phrase_f0 > 0]
    if voiced.size == 0:
        return tuple(sorted(references, key=lambda item: item.reference.id))
    target_pitch = float(np.median(voiced))

    audio, sample_rate = _read_mono(target_audio)
    start_sample = round(phrase.start_seconds * sample_rate)
    end_sample = min(len(audio), round(phrase.end_seconds * sample_rate))
    target_level = float(
        _amplitude_db(np.sqrt(np.mean(np.square(audio[start_sample:end_sample], dtype=np.float64))))
    )

    ranked: list[tuple[float, str, ResolvedVoiceReference]] = []
    for resolved in references:
        metrics = resolved.reference.metrics
        if metrics is None or resolved.reference.review_status.value != "approved":
            continue
        pitch_distance = abs(1200 * np.log2(target_pitch / metrics.median_f0_hz)) / 600
        energy_distance = abs(target_level - metrics.active_rms_dbfs) / 12
        range_penalty = 0.0
        if target_pitch < metrics.f0_p05_hz:
            range_penalty = abs(1200 * np.log2(target_pitch / metrics.f0_p05_hz)) / 600
        elif target_pitch > metrics.f0_p95_hz:
            range_penalty = abs(1200 * np.log2(target_pitch / metrics.f0_p95_hz)) / 600
        ranked.append(
            (
                pitch_distance + 0.35 * energy_distance + range_penalty,
                resolved.reference.id,
                resolved,
            )
        )
    if not ranked:
        return tuple(sorted(references, key=lambda item: item.reference.id))
    return tuple(item[2] for item in sorted(ranked, key=lambda item: (item[0], item[1])))


def compare_f0(target: np.ndarray, converted: np.ndarray) -> tuple[float, float]:
    """Return median cents error and gross error fraction on target-voiced frames."""
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    converted_values = np.asarray(converted, dtype=np.float64).reshape(-1)
    usable = min(len(target_values), len(converted_values))
    target_values = target_values[:usable]
    converted_values = converted_values[:usable]
    target_voiced = target_values > 0
    if not np.any(target_voiced):
        raise VoiceQualityError("Target F0 contains no voiced frames for comparison.")
    both_voiced = target_voiced & (converted_values > 0)
    cents = np.full(usable, np.inf, dtype=np.float64)
    cents[both_voiced] = np.abs(
        1200 * np.log2(converted_values[both_voiced] / target_values[both_voiced])
    )
    median_error = float(np.median(cents[both_voiced])) if np.any(both_voiced) else float("inf")
    gross_error = float(np.mean(cents[target_voiced] > 200))
    return median_error, gross_error


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise VoiceQualityError(f"Audio does not exist: {path}")
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise VoiceQualityError(f"Unable to read audio '{path.name}': {error}") from error
    if audio.size == 0 or sample_rate <= 0:
        raise VoiceQualityError(f"Audio contains no samples: {path}")
    return np.mean(audio, axis=1, dtype=np.float32), sample_rate


def _frame_rms(audio: np.ndarray, sample_rate: int, frame_count: int) -> np.ndarray:
    boundaries = np.rint(np.linspace(0, len(audio), frame_count + 1)).astype(np.int64)
    values = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = audio[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            values[index] = np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
    return values


def _high_band_ratio(audio: np.ndarray, sample_rate: int) -> float:
    usable = audio[: min(len(audio), sample_rate * 3)]
    if usable.size < 2:
        return -120.0
    spectrum = np.square(np.abs(np.fft.rfft(usable * np.hanning(len(usable)))))
    frequencies = np.fft.rfftfreq(len(usable), 1 / sample_rate)
    high = float(np.sum(spectrum[frequencies >= 5000]))
    body = float(np.sum(spectrum[(frequencies >= 100) & (frequencies < 5000)]))
    ratio = 10 * np.log10(max(high, 1e-20) / max(body, 1e-20))
    return float(np.clip(ratio, -120, 60))


def _amplitude_db(value: np.ndarray | float) -> np.ndarray | float:
    result = 20 * np.log10(np.maximum(value, 1e-12))
    if np.isscalar(value):
        return float(result)
    return result
