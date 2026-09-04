"""Rank one prepared accompaniment dataset by tempo and rhythmic energy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


@dataclass(frozen=True)
class TempoCandidate:
    """One periodic-pulse candidate measured from the transient envelope."""

    bpm: float
    correlation: float


@dataclass(frozen=True)
class SongTempoAnalysis:
    """Reproducible full-track tempo and rhythmic-energy measurements."""

    song_id: str
    title: str
    duration_seconds: float
    pulse_bpm: float
    pulse_confidence: float
    onset_density_per_second: float
    transient_activity: float
    active_fraction: float
    tempo_candidates: tuple[TempoCandidate, ...]
    upbeat_score: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prepared dataset.json to analyze.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def _frame_rms(values: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute frame RMS in linear time without materializing overlapping frames."""
    squared = np.square(values, dtype=np.float64)
    cumulative = np.empty(len(squared) + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(squared, out=cumulative[1:])
    starts = np.arange(0, max(len(values) - frame_length + 1, 1), hop_length)
    ends = np.minimum(starts + frame_length, len(values))
    energies = (cumulative[ends] - cumulative[starts]) / np.maximum(ends - starts, 1)
    return np.sqrt(np.maximum(energies, 0.0))


def _robust_scale(values: np.ndarray) -> np.ndarray:
    """Scale a non-negative envelope against its robust upper range."""
    upper = float(np.percentile(values, 95)) if len(values) else 0.0
    return np.clip(values / max(upper, 1e-12), 0.0, 2.0)


def _peak_count(values: np.ndarray, frame_rate: float) -> int:
    """Count separated, prominent transient peaks."""
    if len(values) < 3:
        return 0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + max(1.5 * mad, 0.08)
    candidates = np.flatnonzero(
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] >= threshold)
    ) + 1
    minimum_gap = max(int(round(frame_rate * 0.12)), 1)
    selected: list[int] = []
    for candidate in candidates:
        if not selected or candidate - selected[-1] >= minimum_gap:
            selected.append(int(candidate))
        elif values[candidate] > values[selected[-1]]:
            selected[-1] = int(candidate)
    return len(selected)


def _tempo_candidates(values: np.ndarray, frame_rate: float) -> tuple[TempoCandidate, ...]:
    """Return distinct autocorrelation peaks between 55 and 190 BPM."""
    centered = values - float(np.mean(values))
    candidates: list[TempoCandidate] = []
    for bpm in np.linspace(55.0, 190.0, 541):
        lag = max(int(round(frame_rate * 60.0 / bpm)), 1)
        left = centered[:-lag]
        right = centered[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
        candidates.append(TempoCandidate(bpm=bpm, correlation=correlation))
    ranked = sorted(candidates, key=lambda item: item.correlation, reverse=True)
    distinct: list[TempoCandidate] = []
    for candidate in ranked:
        if all(abs(candidate.bpm - existing.bpm) >= 4.0 for existing in distinct):
            distinct.append(candidate)
        if len(distinct) == 5:
            break
    return tuple(distinct)


def analyze_song(path: Path, song_id: str, title: str) -> SongTempoAnalysis:
    """Analyze one complete accompaniment file without changing it."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    duration = len(mono) / sample_rate
    frame_length = max(int(round(sample_rate * 0.046)), 32)
    hop_length = max(int(round(sample_rate * 0.012)), 8)
    frame_rate = sample_rate / hop_length

    rms = _frame_rms(mono, frame_length, hop_length)
    derivative = np.diff(mono, prepend=mono[0])
    derivative_rms = _frame_rms(derivative, frame_length, hop_length)
    energy_flux = np.maximum(np.diff(np.log(rms + 1e-7), prepend=np.log(rms[0] + 1e-7)), 0.0)
    detail_flux = np.maximum(
        np.diff(
            np.log(derivative_rms + 1e-7),
            prepend=np.log(derivative_rms[0] + 1e-7),
        ),
        0.0,
    )
    transient = 0.6 * _robust_scale(energy_flux) + 0.4 * _robust_scale(detail_flux)
    tempo_candidates = _tempo_candidates(transient, frame_rate)
    primary = tempo_candidates[0] if tempo_candidates else TempoCandidate(0.0, 0.0)
    baseline_correlation = (
        float(np.median([item.correlation for item in tempo_candidates]))
        if tempo_candidates
        else 0.0
    )
    confidence = max(primary.correlation - baseline_correlation, 0.0)
    onset_count = _peak_count(transient, frame_rate)
    rms_threshold = max(float(np.percentile(rms, 20)), 1e-7)

    return SongTempoAnalysis(
        song_id=song_id,
        title=title,
        duration_seconds=duration,
        pulse_bpm=primary.bpm,
        pulse_confidence=confidence,
        onset_density_per_second=onset_count / max(duration, 1.0),
        transient_activity=float(np.mean(transient)),
        active_fraction=float(np.mean(rms > rms_threshold)),
        tempo_candidates=tempo_candidates,
    )


def _with_scores(rows: list[SongTempoAnalysis]) -> list[SongTempoAnalysis]:
    """Add a robust relative score while retaining all raw evidence."""
    fields = (
        ("pulse_bpm", 0.45),
        ("onset_density_per_second", 0.25),
        ("pulse_confidence", 0.15),
        ("transient_activity", 0.15),
    )
    scores = np.zeros(len(rows), dtype=np.float64)
    for field, weight in fields:
        values = np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        median = float(np.median(values))
        scale = 1.4826 * float(np.median(np.abs(values - median)))
        if scale < 1e-12:
            scale = float(np.std(values)) or 1.0
        scores += weight * (values - median) / scale
    return [
        replace(row, upbeat_score=float(score))
        for row, score in zip(rows, scores, strict=True)
    ]


def main() -> None:
    """Analyze and rank every song in one prepared dataset."""
    args = parse_args()
    dataset_path = args.dataset.resolve()
    artist_root = dataset_path.parents[3]
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    catalog = yaml.safe_load((artist_root / "data/catalog.yaml").read_text(encoding="utf-8"))
    titles = {song["id"]: song["title"] for song in catalog["songs"]}

    rows = []
    for index, record in enumerate(dataset["songs"], 1):
        song_id = record["song_id"]
        source = artist_root / Path(record["source_artifact"])
        row = analyze_song(source, song_id, titles.get(song_id, song_id))
        rows.append(row)
        print(f"Measured {index:02d}/{len(dataset['songs']):02d}: {song_id}", flush=True)

    ranked = sorted(_with_scores(rows), key=lambda row: row.upbeat_score, reverse=True)
    payload = {
        "schema_version": 1,
        "analyzed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset_id": dataset["dataset_id"],
        "method": "full-track frame-energy and detail-transient autocorrelation",
        "songs": [asdict(row) for row in ranked],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("\nRANKING")
    for row in ranked:
        candidates = "/".join(f"{item.bpm:.0f}" for item in row.tempo_candidates[:3])
        print(
            f"{row.upbeat_score:6.2f} | {row.pulse_bpm:6.1f} BPM [{candidates:>11}] | "
            f"onsets {row.onset_density_per_second:.2f}/s | "
            f"confidence {row.pulse_confidence:.3f} | {row.song_id}"
        )


if __name__ == "__main__":
    main()
