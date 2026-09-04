"""Minimal reviewed entry point for SoulX-Singer preprocessing operations."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf


def extract_f0(arguments: argparse.Namespace) -> None:
    """Extract a 24 kHz/480-hop RMVPE F0 contour."""
    from preprocess.tools.f0_extraction import F0Extractor

    extractor = F0Extractor(
        model_path=arguments.model,
        device=arguments.device,
        is_half=arguments.device.startswith("cuda"),
        verbose=False,
    )
    extractor.process(arguments.input, f0_path=arguments.output, verbose=False)


def separate(arguments: argparse.Namespace) -> None:
    """Separate vocals/accompaniment and create the vocal F0 contour."""
    from preprocess.tools.f0_extraction import F0Extractor
    from preprocess.tools.vocal_separation.model import VocalSeparator

    separator = VocalSeparator(
        sep_model_path=arguments.separator_model,
        sep_config_path=arguments.separator_config,
        der_model_path=arguments.separator_model,
        der_config_path=arguments.separator_config,
        chunk_length_sec=5,
        use_der=False,
        device=arguments.device,
        verbose=False,
    )
    result = separator.process(arguments.input, verbose=False)
    vocals = Path(arguments.vocals)
    accompaniment = Path(arguments.accompaniment)
    vocals.parent.mkdir(parents=True, exist_ok=True)
    accompaniment.parent.mkdir(parents=True, exist_ok=True)
    sf.write(vocals, result.vocals_dereverbed.T, result.sample_rate)
    sf.write(accompaniment, result.accompaniment.T, result.sample_rate)

    extractor = F0Extractor(
        model_path=arguments.f0_model,
        device=arguments.device,
        is_half=arguments.device.startswith("cuda"),
        verbose=False,
    )
    extractor.process(str(vocals), f0_path=arguments.f0_output, verbose=False)


def dereverb_batch(arguments: argparse.Namespace) -> None:
    """Dereverberate existing vocal stems with one reviewed model load."""
    import librosa
    import torch

    request = json.loads(Path(arguments.request).read_text(encoding="utf-8"))
    jobs = request.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Dereverberation request must contain at least one job.")
    dereverb_strength = float(request.get("dereverb_strength", 1.0))
    if not 0 <= dereverb_strength <= 1:
        raise ValueError("Dereverberation strength must be between zero and one.")

    repository = Path.cwd().resolve()
    package_paths = {
        "preprocess": repository / "preprocess",
        "preprocess.tools": repository / "preprocess/tools",
        "preprocess.tools.vocal_separation": repository / "preprocess/tools/vocal_separation",
    }
    for package_name, package_path in package_paths.items():
        package = ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package

    from preprocess.tools.vocal_separation.model import process
    from preprocess.tools.vocal_separation.utils.settings import get_model_from_config

    model_arguments = SimpleNamespace(
        model_type="mel_band_roformer",
        config_path=str(Path(arguments.config).resolve()),
        start_check_point=str(Path(arguments.model).resolve()),
        disable_detailed_pbar=True,
        lora_checkpoint=None,
    )
    initializer_names = (
        "constant_",
        "dirac_",
        "eye_",
        "kaiming_normal_",
        "kaiming_uniform_",
        "normal_",
        "ones_",
        "orthogonal_",
        "sparse_",
        "trunc_normal_",
        "uniform_",
        "xavier_normal_",
        "xavier_uniform_",
        "zeros_",
    )
    original_initializers = {
        name: getattr(torch.nn.init, name)
        for name in initializer_names
        if hasattr(torch.nn.init, name)
    }

    def skip_initialization(
        tensor: torch.Tensor,
        *_arguments: object,
        **_keywords: object,
    ) -> torch.Tensor:
        return tensor

    try:
        for name in original_initializers:
            setattr(torch.nn.init, name, skip_initialization)
        model, config = get_model_from_config(
            model_arguments.model_type,
            model_arguments.config_path,
        )
    finally:
        for name, initializer in original_initializers.items():
            setattr(torch.nn.init, name, initializer)

    state_dict = torch.load(
        model_arguments.start_check_point,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state_dict, assign=True)
    del state_dict
    gc.collect()
    sample_rate = int(config.audio.sample_rate)
    config.inference.chunk_size = round(float(request.get("chunk_seconds", 5.0)) * sample_rate)
    device = torch.device(arguments.device)
    model = model.half() if device.type == "cuda" else model.float()
    model = model.to(device).eval()

    results = []
    for index, job in enumerate(jobs):
        input_path = Path(job["input"]).resolve()
        output_path = Path(job["output"]).resolve()
        audio, _ = librosa.load(input_path, sr=sample_rate, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        with torch.inference_mode():
            cleaned = process(audio, model, model_arguments, config, device)

        source = audio.T.astype(np.float32, copy=False)
        cleaned = cleaned.T.astype(np.float32, copy=False)
        expected_frames = source.shape[0]
        if cleaned.shape[0] < expected_frames:
            cleaned = np.pad(cleaned, ((0, expected_frames - cleaned.shape[0]), (0, 0)))
        cleaned = cleaned[:expected_frames]
        if not np.all(np.isfinite(cleaned)):
            raise ValueError(f"Dereverberation produced non-finite samples for job {index}.")

        cleaned = dereverb_strength * cleaned + (1 - dereverb_strength) * source

        source_active_rms = _active_rms(source)
        cleaned_active_rms = _active_rms(cleaned)
        gain_db = float(
            np.clip(_amplitude_db(source_active_rms) - _amplitude_db(cleaned_active_rms), -12, 12)
        )
        cleaned *= 10 ** (gain_db / 20)
        peak_limit = 10 ** (-1 / 20)
        peak = float(np.max(np.abs(cleaned)))
        peak_reduction_db = 0.0
        if peak > peak_limit:
            peak_reduction_db = _amplitude_db(peak_limit) - _amplitude_db(peak)
            cleaned *= 10 ** (peak_reduction_db / 20)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, cleaned, sample_rate, subtype="PCM_24")
        results.append(
            {
                "index": index,
                "duration_seconds": cleaned.shape[0] / sample_rate,
                "sample_rate": sample_rate,
                "channels": int(cleaned.shape[1]),
                "dereverb_strength": dereverb_strength,
                "source_restoration_ratio": 1 - dereverb_strength,
                "active_level_match_db": gain_db,
                "peak_reduction_db": peak_reduction_db,
                "active_rms_dbfs": _amplitude_db(_active_rms(cleaned)),
                "peak_dbfs": _amplitude_db(float(np.max(np.abs(cleaned)))),
                "clipped_sample_count": int(np.count_nonzero(np.abs(cleaned) >= 1)),
                "nonfinite_sample_count": int(np.count_nonzero(~np.isfinite(cleaned))),
            }
        )

    Path(arguments.result).write_text(
        json.dumps({"schema_version": 1, "results": results}, indent=2),
        encoding="utf-8",
    )


def _active_rms(audio: np.ndarray, threshold_dbfs: float = -45.0) -> float:
    mono = np.mean(audio, axis=1) if audio.ndim == 2 else audio
    frame_length = 2048
    usable = len(mono) - (len(mono) % frame_length)
    if usable == 0:
        return float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    frames = mono[:usable].reshape(-1, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, dtype=np.float64))
    active = frames[frame_rms >= 10 ** (threshold_dbfs / 20)]
    selected = active if active.size else mono
    return float(np.sqrt(np.mean(np.square(selected), dtype=np.float64)))


def _amplitude_db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def quality_convert(arguments: argparse.Namespace) -> None:
    """Convert overlapping phrases with one model load and deterministic candidate QC."""
    import torch
    from cli.inference_svc import build_model
    from soulxsinger.utils.audio_utils import load_wav
    from soulxsinger.utils.file_utils import load_config

    request = json.loads(Path(arguments.request).read_text(encoding="utf-8"))
    config = load_config(arguments.config)
    sample_rate = int(config.audio.sample_rate)
    hop_size = int(config.audio.hop_size)
    f0_rate = sample_rate // hop_size
    if f0_rate <= 0:
        raise ValueError("SoulX configuration has an invalid F0 rate.")

    model = build_model(
        model_path=arguments.model,
        config=config,
        device=arguments.device,
        use_fp16=arguments.fp16,
    )
    target_wav = load_wav(request["target_audio"], sample_rate).to(arguments.device)
    target_f0_np = np.load(request["target_f0"], allow_pickle=False).astype(np.float32).reshape(-1)
    target_f0 = torch.from_numpy(target_f0_np).unsqueeze(0).to(arguments.device)
    references = {item["id"]: item for item in request["references"]}
    loaded_references: dict[str, tuple[torch.Tensor, torch.Tensor, np.ndarray]] = {}

    def load_reference(reference_id: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        cached = loaded_references.get(reference_id)
        if cached is not None:
            return cached
        record = references[reference_id]
        waveform = load_wav(record["audio"], sample_rate).to(arguments.device)
        f0_values = np.load(record["f0"], allow_pickle=False).astype(np.float32).reshape(-1)
        f0_tensor = torch.from_numpy(f0_values).unsqueeze(0).to(arguments.device)
        cached = (waveform, f0_tensor, waveform.squeeze().float().cpu().numpy())
        loaded_references[reference_id] = cached
        return cached

    phrases_root = Path(request["phrases_directory"])
    phrases_root.mkdir(parents=True, exist_ok=True)
    decisions: list[dict[str, Any]] = []
    selected_audio: list[tuple[int, int, np.ndarray]] = []
    selected_dropout_count = 0
    settings = request["settings"]
    candidate_count = int(settings["candidates_per_phrase"])
    steps = int(settings["steps"])
    cfg = float(settings["cfg"])
    base_seed = int(request["seed"])

    for phrase in request["phrases"]:
        phrase_index = int(phrase["index"])
        start_frame = int(phrase["start_frame"])
        end_frame = int(phrase["end_frame"])
        reference_id = str(phrase["reference_id"])
        start_sample = round(start_frame * sample_rate / f0_rate)
        end_sample = min(target_wav.shape[-1], round(end_frame * sample_rate / f0_rate))
        phrase_wav = target_wav[:, start_sample:end_sample]
        phrase_f0 = target_f0[:, start_frame:end_frame]
        prompt_wav, prompt_f0, prompt_numpy = load_reference(reference_id)
        candidate_records = []
        candidate_audio = []

        for candidate_index in range(candidate_count):
            seed = _candidate_seed(base_seed, phrase_index, candidate_index)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                generated, _shift = model.infer(
                    pt_wav=prompt_wav,
                    gt_wav=phrase_wav,
                    pt_f0=prompt_f0,
                    gt_f0=phrase_f0,
                    auto_shift=False,
                    pitch_shift=0,
                    n_steps=steps,
                    cfg=cfg,
                    use_fp16=arguments.fp16,
                )
            values = generated.squeeze().float().cpu().numpy().astype(np.float32)
            expected_samples = end_sample - start_sample
            values = _fit_length(values, expected_samples)
            raw_peak = float(np.max(np.abs(values))) if values.size else 0.0
            if not np.isfinite(raw_peak) or raw_peak <= 0:
                raise RuntimeError(f"SoulX returned invalid audio for phrase {phrase_index}.")
            peak_reduction_db = max(0.0, 20 * math.log10(raw_peak / 0.95))
            if raw_peak > 0.95:
                values *= 0.95 / raw_peak

            dropout_count, dropout_fraction = _dropout_metrics(
                values,
                phrase_f0.squeeze().float().cpu().numpy(),
                sample_rate,
                f0_rate,
            )
            spectral_distance = abs(
                _high_band_ratio(values, sample_rate)
                - _high_band_ratio(prompt_numpy, sample_rate)
            )
            f0_median_error, f0_gross_error = _pitch_metrics(
                values,
                phrase_f0.squeeze().float().cpu().numpy(),
                sample_rate,
                f0_rate,
            )
            passed = (
                dropout_count == 0
                and raw_peak <= 4.0
                and f0_median_error <= 100
                and f0_gross_error <= 0.20
            )
            record = {
                "candidate_index": candidate_index,
                "seed": seed,
                "passed": passed,
                "raw_peak_dbfs": 20 * math.log10(max(raw_peak, 1e-12)),
                "peak_reduction_db": peak_reduction_db,
                "sustained_dropout_count": dropout_count,
                "dropout_fraction": dropout_fraction,
                "spectral_distance_db": spectral_distance,
                "f0_median_error_cents": f0_median_error,
                "f0_gross_error_fraction": f0_gross_error,
            }
            candidate_path = phrases_root / (
                f"phrase-{phrase_index:03d}-candidate-{candidate_index}.wav"
            )
            sf.write(candidate_path, values, sample_rate, subtype="PCM_24")
            candidate_records.append(record)
            candidate_audio.append(values)

        selected_index = min(
            range(candidate_count),
            key=lambda index: (
                not candidate_records[index]["passed"],
                candidate_records[index]["sustained_dropout_count"],
                candidate_records[index]["dropout_fraction"],
                candidate_records[index]["f0_gross_error_fraction"],
                candidate_records[index]["f0_median_error_cents"],
                candidate_records[index]["spectral_distance_db"],
                candidate_records[index]["peak_reduction_db"],
                index,
            ),
        )
        if not candidate_records[selected_index]["passed"]:
            raise RuntimeError(
                f"Every candidate for phrase {phrase_index} failed sustained-dropout QC."
            )
        selected_dropout_count += int(
            candidate_records[selected_index]["sustained_dropout_count"]
        )
        selected_audio.append((start_sample, end_sample, candidate_audio[selected_index]))
        decisions.append(
            {
                "index": phrase_index,
                "start_seconds": start_frame / f0_rate,
                "end_seconds": end_frame / f0_rate,
                "reference_id": reference_id,
                "selected_candidate_index": selected_index,
                "candidates": candidate_records,
            }
        )

    joined, join_gains = _join_phrases(selected_audio, target_wav.shape[-1])
    for decision, join_gain_db in zip(decisions, join_gains, strict=True):
        decision["join_gain_db"] = join_gain_db
    output_peak = float(np.max(np.abs(joined))) if joined.size else 0.0
    if output_peak > 0.95:
        joined *= 0.95 / output_peak
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, joined, sample_rate, subtype="PCM_24")
    result = {
        "settings": settings,
        "phrases": decisions,
        "metrics": {
            "phrase_count": len(decisions),
            "selected_reference_ids": sorted(
                {decision["reference_id"] for decision in decisions}
            ),
            "sustained_dropout_count": selected_dropout_count,
            "peak_dbfs": 20 * math.log10(max(float(np.max(np.abs(joined))), 1e-12)),
            "clipping_fraction": float(np.mean(np.abs(joined) >= 0.999)),
            "max_voiced_join_jump_db": _max_join_jump_db(
                joined,
                [item[0] for item in selected_audio[1:]],
                target_f0_np,
                sample_rate,
                f0_rate,
            ),
            "f0_median_error_cents": None,
            "f0_gross_error_fraction": None,
        },
    }
    result_path = Path(request["result"])
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, result_path)


def _fit_length(values: np.ndarray, expected_samples: int) -> np.ndarray:
    """Trim or zero-pad a generated phrase to its exact target duration."""
    if len(values) >= expected_samples:
        return values[:expected_samples]
    return np.pad(values, (0, expected_samples - len(values)))


def _candidate_seed(base_seed: int, phrase_index: int, candidate_index: int) -> int:
    """Derive a stable, non-overlapping seed for one phrase candidate."""
    return base_seed + phrase_index * 1009 + candidate_index


def _dropout_metrics(
    audio: np.ndarray,
    f0: np.ndarray,
    sample_rate: int,
    f0_rate: int,
) -> tuple[int, float]:
    """Count candidate silence lasting at least 300 ms during target voicing."""
    frame_count = min(len(f0), round(len(audio) * f0_rate / sample_rate))
    if frame_count <= 0:
        return 0, 0.0
    boundaries = np.rint(np.linspace(0, len(audio), frame_count + 1)).astype(np.int64)
    rms = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = audio[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            rms[index] = np.sqrt(np.mean(np.square(frame, dtype=np.float64)))
    voiced = np.asarray(f0[:frame_count]) > 0
    if not np.any(voiced):
        return 0, 0.0
    active_median = float(np.median(rms[voiced]))
    threshold = max(active_median * (10 ** (-35 / 20)), 10 ** (-60 / 20))
    dropout = voiced & (rms < threshold)
    sustained = 0
    index = 0
    minimum_frames = round(0.3 * f0_rate)
    while index < frame_count:
        if not dropout[index]:
            index += 1
            continue
        start = index
        while index < frame_count and dropout[index]:
            index += 1
        if index - start >= minimum_frames:
            sustained += 1
    return sustained, float(np.sum(dropout) / max(np.sum(voiced), 1))


def _high_band_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Return high-frequency energy relative to the vocal body band."""
    usable = np.asarray(audio, dtype=np.float32).reshape(-1)[: sample_rate * 3]
    if usable.size < 2:
        return -120.0
    spectrum = np.square(np.abs(np.fft.rfft(usable * np.hanning(len(usable)))))
    frequencies = np.fft.rfftfreq(len(usable), 1 / sample_rate)
    high = float(np.sum(spectrum[frequencies >= 5000]))
    body = float(np.sum(spectrum[(frequencies >= 100) & (frequencies < 5000)]))
    return 10 * math.log10(max(high, 1e-20) / max(body, 1e-20))


def _pitch_metrics(
    audio: np.ndarray,
    f0: np.ndarray,
    sample_rate: int,
    f0_rate: int,
) -> tuple[float, float]:
    """Estimate phrase pitch adherence without loading another GPU model."""
    target = np.asarray(f0, dtype=np.float64).reshape(-1)
    voiced_indices = np.flatnonzero(target > 0)[::5]
    if voiced_indices.size == 0:
        return 0.0, 0.0
    half_window = round(0.04 * sample_rate)
    errors = []
    for frame_index in voiced_indices:
        center = round((frame_index + 0.5) * sample_rate / f0_rate)
        start = max(0, center - half_window)
        end = min(len(audio), center + half_window)
        segment = np.asarray(audio[start:end], dtype=np.float64)
        if segment.size < half_window or np.sqrt(np.mean(np.square(segment))) < 1e-5:
            errors.append(float("inf"))
            continue
        segment = (segment - np.mean(segment)) * np.hanning(len(segment))
        fft_size = 1 << (2 * len(segment) - 1).bit_length()
        spectrum = np.fft.rfft(segment, fft_size)
        autocorrelation = np.fft.irfft(np.square(np.abs(spectrum)), fft_size)[: len(segment)]
        if autocorrelation[0] <= 1e-12:
            errors.append(float("inf"))
            continue
        autocorrelation /= autocorrelation[0]
        expected_lag = sample_rate / target[frame_index]
        minimum_lag = max(1, round(expected_lag / (2 ** (100 / 1200))))
        maximum_lag = min(
            len(autocorrelation) - 1,
            round(expected_lag * (2 ** (100 / 1200))),
        )
        if maximum_lag <= minimum_lag:
            errors.append(float("inf"))
            continue
        local_offset = int(np.argmax(autocorrelation[minimum_lag : maximum_lag + 1]))
        estimated_lag = minimum_lag + local_offset
        local_strength = float(autocorrelation[estimated_lag])
        octave_strengths = []
        for multiplier in (0.5, 2.0):
            octave_lag = round(expected_lag * multiplier)
            if 1 <= octave_lag < len(autocorrelation):
                octave_strengths.append(float(autocorrelation[octave_lag]))
        if local_strength < 0.12 or any(
            strength > local_strength * 1.15 for strength in octave_strengths
        ):
            errors.append(1200.0)
            continue
        estimated_f0 = sample_rate / estimated_lag
        errors.append(abs(1200 * math.log2(estimated_f0 / target[frame_index])))
    finite = np.asarray(errors, dtype=np.float64)
    finite_errors = finite[np.isfinite(finite)]
    median = float(np.median(finite_errors)) if finite_errors.size else 1200.0
    gross = float(np.mean(finite > 200))
    return median, gross


def _max_join_jump_db(
    audio: np.ndarray,
    join_samples: list[int],
    target_f0: np.ndarray,
    sample_rate: int,
    f0_rate: int,
) -> float:
    """Measure instantaneous level discontinuity only across continuously voiced joins."""
    window = max(1, round(0.005 * sample_rate))
    jumps = []
    for join in join_samples:
        f0_index = min(len(target_f0) - 1, round(join * f0_rate / sample_rate))
        if (
            f0_index == 0
            or target_f0[f0_index - 1] <= 0
            or target_f0[f0_index] <= 0
            or join < window
            or join + window > len(audio)
        ):
            continue
        before = float(np.sqrt(np.mean(np.square(audio[join - window : join], dtype=np.float64))))
        after = float(np.sqrt(np.mean(np.square(audio[join : join + window], dtype=np.float64))))
        jumps.append(abs(20 * math.log10(max(after, 1e-12) / max(before, 1e-12))))
    return max(jumps, default=0.0)


def _join_phrases(
    phrases: list[tuple[int, int, np.ndarray]],
    total_samples: int,
) -> tuple[np.ndarray, list[float]]:
    """Join sorted overlapping phrases with equal-power crossfades."""
    if not phrases:
        raise ValueError("No converted phrases were produced.")
    output = np.zeros(total_samples, dtype=np.float32)
    written_until = 0
    join_gains = []
    for start, end, values in phrases:
        end = min(end, total_samples)
        values = _fit_length(values, end - start).copy()
        overlap_end = min(written_until, end)
        join_gain_db = 0.0
        if overlap_end > start:
            overlap = overlap_end - start
            previous_rms = float(
                np.sqrt(np.mean(np.square(output[start:overlap_end], dtype=np.float64)))
            )
            current_rms = float(
                np.sqrt(np.mean(np.square(values[:overlap], dtype=np.float64)))
            )
            if previous_rms > 1e-8 and current_rms > 1e-8:
                join_gain_db = float(
                    np.clip(20 * math.log10(previous_rms / current_rms), -6, 6)
                )
                values *= 10 ** (join_gain_db / 20)
            phase = np.linspace(0, math.pi / 2, overlap, endpoint=True)
            output[start:overlap_end] = (
                output[start:overlap_end] * np.cos(phase)
                + values[:overlap] * np.sin(phase)
            )
            output[overlap_end:end] = values[overlap : overlap + end - overlap_end]
        else:
            output[start:end] = values[: end - start]
        join_gains.append(join_gain_db)
        written_until = max(written_until, end)
    if written_until < total_samples:
        output[written_until:] = 0
    return output, join_gains


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated helper command parser."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    f0 = commands.add_parser("f0")
    f0.add_argument("--input", required=True)
    f0.add_argument("--output", required=True)
    f0.add_argument("--model", required=True)
    f0.add_argument("--device", default="cuda")
    f0.set_defaults(handler=extract_f0)

    separation = commands.add_parser("separate")
    separation.add_argument("--input", required=True)
    separation.add_argument("--vocals", required=True)
    separation.add_argument("--accompaniment", required=True)
    separation.add_argument("--f0-output", required=True)
    separation.add_argument("--separator-model", required=True)
    separation.add_argument("--separator-config", required=True)
    separation.add_argument("--f0-model", required=True)
    separation.add_argument("--device", default="cuda")
    separation.set_defaults(handler=separate)

    dereverb = commands.add_parser("dereverb-batch")
    dereverb.add_argument("--request", required=True)
    dereverb.add_argument("--result", required=True)
    dereverb.add_argument("--model", required=True)
    dereverb.add_argument("--config", required=True)
    dereverb.add_argument("--device", default="cuda")
    dereverb.set_defaults(handler=dereverb_batch)

    conversion = commands.add_parser("quality-convert")
    conversion.add_argument("--request", required=True)
    conversion.add_argument("--model", required=True)
    conversion.add_argument("--config", required=True)
    conversion.add_argument("--device", default="cuda")
    conversion.add_argument("--fp16", action="store_true")
    conversion.set_defaults(handler=quality_convert)
    return parser


def main() -> None:
    """Run exactly one requested preprocessing operation."""
    arguments = build_parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
