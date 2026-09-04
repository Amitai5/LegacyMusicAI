"""Minimal deterministic entry point for the pinned Seed-VC inference module."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch


def convert(arguments: argparse.Namespace) -> None:
    """Run one deterministic singing-voice conversion with explicit settings."""
    sys.path.insert(0, str(Path.cwd().resolve()))
    from inference import main

    random.seed(arguments.seed)
    np.random.seed(arguments.seed % (2**32))
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.use_deterministic_algorithms(False)
    main(
        SimpleNamespace(
            source=arguments.source,
            target=arguments.target,
            output=arguments.output,
            diffusion_steps=arguments.diffusion_steps,
            length_adjust=1.0,
            inference_cfg_rate=arguments.inference_cfg_rate,
            f0_condition=True,
            auto_f0_adjust=False,
            semi_tone_shift=0,
            checkpoint=arguments.checkpoint,
            config=arguments.config,
            fp16=arguments.fp16,
        )
    )


def batch_convert(arguments: argparse.Namespace) -> None:
    """Convert identity-gated phrase candidates while loading Seed-VC exactly once."""
    sys.path.insert(0, str(Path.cwd().resolve()))
    import torchaudio
    from inference import load_models, main

    request_path = Path(arguments.batch_request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    runtime_root = Path(request["runtime_root"]).resolve()
    result_path = Path(request["result"]).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    phrases = request.get("phrases")
    references = request.get("references")
    settings = request.get("settings")
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("Seed-VC batch request contains no phrases.")
    if not isinstance(references, list) or not references:
        raise ValueError("Seed-VC batch request contains no references.")
    if not isinstance(settings, dict):
        raise ValueError("Seed-VC batch request settings are invalid.")

    references_by_id = {str(item["id"]): Path(item["audio"]).resolve() for item in references}
    first_phrase = phrases[0]
    first_reference_ids = first_phrase.get("reference_ids") or [first_phrase["reference_id"]]
    first_reference = references_by_id[str(first_reference_ids[0])]
    base_arguments = SimpleNamespace(
        source=str(Path(first_phrase["source"]).resolve()),
        target=str(first_reference),
        output=str(runtime_root),
        diffusion_steps=int(settings["diffusion_steps"]),
        length_adjust=1.0,
        inference_cfg_rate=float(settings["inference_cfg_rate"]),
        f0_condition=True,
        auto_f0_adjust=False,
        semi_tone_shift=0,
        checkpoint=arguments.checkpoint,
        config=arguments.config,
        fp16=arguments.fp16,
    )
    _set_seed(int(request["seed"]))
    loaded_models = load_models(base_arguments)
    f0_fn = loaded_models[2]
    campplus_model = loaded_models[4]
    device = next(campplus_model.parameters()).device
    reference_embeddings = {
        reference_id: _speaker_embedding(audio, campplus_model, device, torchaudio)
        for reference_id, audio in references_by_id.items()
    }
    ordered_ids = sorted(reference_embeddings)
    ordered_embeddings = torch.stack(
        [reference_embeddings[reference_id] for reference_id in ordered_ids]
    )
    centroid = torch.nn.functional.normalize(
        ordered_embeddings.mean(dim=0),
        dim=0,
    )
    calibration = _leave_one_out_similarities(ordered_ids, reference_embeddings, torch)
    configured_threshold = settings.get("identity_threshold")
    threshold = (
        float(configured_threshold)
        if configured_threshold is not None
        else _adaptive_identity_threshold(
            tuple(calibration.values()),
            float(settings["identity_similarity_margin"]),
            float(settings["identity_minimum_similarity"]),
        )
    )
    window_seconds = float(settings["identity_window_seconds"])
    window_hop_seconds = float(settings["identity_window_hop_seconds"])
    window_min_voiced_fraction = float(settings["identity_window_min_voiced_fraction"])
    reference_window_calibration = {}
    reference_window_scores = []
    for reference_id in ordered_ids:
        window_embeddings = _identity_window_embeddings(
            references_by_id[reference_id],
            campplus_model,
            device,
            torchaudio,
            window_seconds,
            window_hop_seconds,
        )
        scores = _combined_identity_scores(
            window_embeddings,
            centroid,
            reference_embeddings[reference_id],
        )
        reference_window_scores.extend(scores)
        reference_window_calibration[reference_id] = {
            "count": len(scores),
            "minimum": min(scores),
            "p10": float(np.quantile(scores, 0.1)),
            "mean": float(np.mean(scores)),
        }
    window_threshold = _adaptive_window_identity_threshold(
        tuple(reference_window_scores),
        float(settings["identity_window_similarity_margin"]),
        float(settings["identity_minimum_similarity"]),
    )

    import inference

    original_loader = inference.load_models
    inference.load_models = lambda _arguments: loaded_models
    records = []
    initial_candidate_count = int(settings["candidates_per_phrase"])
    maximum_candidate_count = int(settings["max_candidates_per_phrase"])
    minimum_passing_candidates = 2
    try:
        for phrase in phrases:
            phrase_index = int(phrase["index"])
            source = Path(phrase["source"]).resolve()
            candidate_reference_ids = [
                str(reference_id)
                for reference_id in (phrase.get("reference_ids") or [phrase["reference_id"]])
            ]
            if not candidate_reference_ids:
                raise ValueError(f"Seed-VC phrase {phrase_index} has no references.")
            target_f0, target_duration = _pitch_contour(
                source,
                f0_fn,
                device,
                torchaudio,
            )
            authoritative_f0_path = Path(phrase["target_f0"]).resolve()
            authoritative_f0 = np.load(
                authoritative_f0_path,
                allow_pickle=False,
            ).astype(np.float32)
            phrase_records = []
            for candidate_index in range(maximum_candidate_count):
                if candidate_index < initial_candidate_count:
                    reference_id = candidate_reference_ids[
                        candidate_index % len(candidate_reference_ids)
                    ]
                else:
                    retry_reference_ids = _retry_reference_ids(
                        phrase_records,
                        candidate_reference_ids,
                    )
                    reference_id = retry_reference_ids[
                        (candidate_index - initial_candidate_count)
                        % min(3, len(retry_reference_ids))
                    ]
                reference = references_by_id[reference_id]
                seed = _candidate_seed(int(request["seed"]), phrase_index, candidate_index)
                _set_seed(seed)
                candidate_root = runtime_root / (
                    f"phrase-{phrase_index:03d}-candidate-{candidate_index}"
                )
                candidate_root.mkdir(parents=True, exist_ok=False)
                candidate_arguments = SimpleNamespace(
                    **{
                        **vars(base_arguments),
                        "source": str(source),
                        "target": str(reference),
                        "output": str(candidate_root),
                    }
                )
                main(candidate_arguments)
                generated = sorted(candidate_root.glob("vc_*.wav"))
                if len(generated) != 1:
                    raise RuntimeError(
                        f"Seed-VC created {len(generated)} outputs for phrase {phrase_index}."
                    )
                repaired_dropout_count = _repair_short_dropouts(
                    generated[0],
                    authoritative_f0,
                    int(settings["audio_dropout_repair_ms"]),
                    float(settings["audio_dropout_max_gain_db"]),
                )
                embedding = _speaker_embedding(
                    generated[0],
                    campplus_model,
                    device,
                    torchaudio,
                )
                similarity = float(torch.dot(embedding, centroid).detach().cpu())
                reference_similarity = float(
                    torch.dot(embedding, reference_embeddings[reference_id]).detach().cpu()
                )
                window_embeddings = _identity_window_embeddings(
                    generated[0],
                    campplus_model,
                    device,
                    torchaudio,
                    window_seconds,
                    window_hop_seconds,
                    target_f0=target_f0,
                    target_duration=target_duration,
                    minimum_voiced_fraction=window_min_voiced_fraction,
                )
                window_scores = _combined_identity_scores(
                    window_embeddings,
                    centroid,
                    reference_embeddings[reference_id],
                )
                candidate_f0, _candidate_duration = _pitch_contour(
                    generated[0],
                    f0_fn,
                    device,
                    torchaudio,
                )
                median_pitch_error, gross_pitch_error = _pitch_metrics(
                    target_f0,
                    candidate_f0,
                    target_duration,
                    int(settings["f0_gap_fill_ms"]),
                )
                dropout_count, dropout_fraction = _candidate_dropout_metrics(
                    generated[0],
                    authoritative_f0,
                    target_duration,
                    torchaudio,
                )
                record = {
                    "phrase_index": phrase_index,
                    "candidate_index": candidate_index,
                    "reference_id": reference_id,
                    "seed": seed,
                    "audio": generated[0].relative_to(runtime_root).as_posix(),
                    "identity_similarity": similarity,
                    "reference_identity_similarity": reference_similarity,
                    "minimum_window_identity_similarity": min(window_scores),
                    "mean_window_identity_similarity": float(np.mean(window_scores)),
                    "identity_window_count": len(window_scores),
                    "f0_median_error_cents": median_pitch_error,
                    "f0_gross_error_fraction": gross_pitch_error,
                    "sustained_dropout_count": dropout_count,
                    "repaired_dropout_count": repaired_dropout_count,
                    "dropout_fraction": dropout_fraction,
                }
                phrase_records.append(record)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (
                    candidate_index + 1 >= initial_candidate_count
                    and sum(
                        _runtime_candidate_passes(item, settings, window_threshold)
                        for item in phrase_records
                    )
                    >= minimum_passing_candidates
                ):
                    break
            records.extend(phrase_records)
    finally:
        inference.load_models = original_loader

    result = {
        "schema_version": 2,
        "identity_threshold": threshold,
        "identity_window_threshold": window_threshold,
        "reference_calibration_similarities": calibration,
        "reference_window_calibration": reference_window_calibration,
        "candidates": records,
    }
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, result_path)


def _set_seed(seed: int) -> None:
    """Reset every stochastic source used by upstream inference."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _candidate_seed(base_seed: int, phrase_index: int, candidate_index: int) -> int:
    """Derive one stable candidate seed without overflowing signed 64-bit storage."""
    return (base_seed + phrase_index * 1009 + candidate_index) % (2**63 - 1)


def _speaker_embedding(path: Path, model, device, torchaudio):
    """Return one normalized CAMPPlus embedding for identity comparison."""
    waveform = _load_waveform_16k(path, torchaudio)
    return _embedding_for_waveform(waveform, model, device, torchaudio)


def _load_waveform_16k(path: Path, torchaudio):
    """Load one mono waveform at the identity encoder's required rate."""
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True).float()
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    return waveform


def _embedding_for_waveform(waveform, model, device, torchaudio):
    """Encode one already-resampled waveform with CAMPPlus."""
    if waveform.shape[-1] < 1600:
        waveform = torch.nn.functional.pad(waveform, (0, 1600 - waveform.shape[-1]))
    waveform = waveform.to(device)
    features = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    features = features - features.mean(dim=0, keepdim=True)
    with torch.no_grad():
        embedding = model(features.unsqueeze(0)).squeeze(0)
    return torch.nn.functional.normalize(embedding.float(), dim=0)


def _identity_window_embeddings(
    path: Path,
    model,
    device,
    torchaudio,
    window_seconds: float,
    hop_seconds: float,
    *,
    target_f0: np.ndarray | None = None,
    target_duration: float | None = None,
    minimum_voiced_fraction: float = 0.0,
):
    """Encode overlapping active windows so brief identity drift cannot hide in an average."""
    waveform = _load_waveform_16k(path, torchaudio)
    total_samples = waveform.shape[-1]
    window_samples = max(1600, round(window_seconds * 16000))
    hop_samples = max(1, round(hop_seconds * 16000))
    if total_samples <= window_samples:
        starts = [0]
    else:
        starts = list(range(0, total_samples - window_samples + 1, hop_samples))
        final_start = total_samples - window_samples
        if starts[-1] != final_start:
            starts.append(final_start)
    embeddings = []
    for start in starts:
        end = min(total_samples, start + window_samples)
        if target_f0 is not None and target_duration is not None and target_duration > 0:
            start_frame = min(
                len(target_f0),
                round((start / 16000) * len(target_f0) / target_duration),
            )
            end_frame = min(
                len(target_f0),
                round((end / 16000) * len(target_f0) / target_duration),
            )
            voiced_fraction = (
                float(np.mean(target_f0[start_frame:end_frame] > 0))
                if end_frame > start_frame
                else 0.0
            )
            if voiced_fraction < minimum_voiced_fraction:
                continue
        window = waveform[:, start:end]
        if float(torch.sqrt(torch.mean(torch.square(window)))) < 1e-5:
            continue
        embeddings.append(_embedding_for_waveform(window, model, device, torchaudio))
    if not embeddings:
        embeddings.append(_embedding_for_waveform(waveform, model, device, torchaudio))
    return embeddings


def _combined_identity_scores(embeddings, centroid, reference_embedding) -> list[float]:
    """Blend artist-centroid and register-matched-reference identity evidence."""
    return [
        float(
            (0.7 * torch.dot(embedding, centroid) + 0.3 * torch.dot(embedding, reference_embedding))
            .detach()
            .cpu()
        )
        for embedding in embeddings
    ]


def _candidate_dropout_metrics(
    path: Path,
    target_f0: np.ndarray,
    target_duration: float,
    torchaudio,
) -> tuple[int, float]:
    """Measure sustained silence during target voicing without loading another model."""
    waveform = _load_waveform_16k(path, torchaudio).squeeze(0).cpu().numpy()
    frame_count = min(
        len(target_f0),
        round(len(waveform) * len(target_f0) / 16000 / target_duration),
    )
    if frame_count <= 0:
        return 1, 1.0
    boundaries = np.rint(np.linspace(0, len(waveform), frame_count + 1)).astype(np.int64)
    rms = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = waveform[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            rms[index] = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    voiced = target_f0[:frame_count] > 0
    if not np.any(voiced):
        return 0, 0.0
    threshold = max(float(np.median(rms[voiced])) * 0.03, 1e-6)
    dropout = voiced & (rms < threshold)
    minimum_run = max(1, round(0.12 * frame_count / target_duration))
    count = 0
    index = 0
    while index < frame_count:
        if not dropout[index]:
            index += 1
            continue
        start = index
        while index < frame_count and dropout[index]:
            index += 1
        if index - start >= minimum_run:
            count += 1
    return count, float(np.mean(dropout[voiced]))


def _repair_short_dropouts(
    path: Path,
    target_f0: np.ndarray,
    maximum_ms: int,
    maximum_gain_db: float,
) -> int:
    """Repair capped 120 ms waveform holes in-place before candidate scoring."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if maximum_ms <= 0 or maximum_gain_db <= 0:
        return 0
    frame_count = min(len(target_f0), round(len(mono) * 50 / sample_rate))
    if frame_count <= 0:
        return 0
    boundaries = np.rint(np.linspace(0, len(mono), frame_count + 1)).astype(np.int64)
    rms = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        frame = mono[boundaries[index] : boundaries[index + 1]]
        if frame.size:
            rms[index] = np.sqrt(np.mean(np.square(frame), dtype=np.float64))
    voiced = np.asarray(target_f0).reshape(-1)[:frame_count] > 0
    if not np.any(voiced):
        return 0
    threshold = max(float(np.median(rms[voiced])) * 0.03, 1e-6)
    dropout = voiced & (rms < threshold)
    maximum_frames = max(1, round(maximum_ms * 50 / 1000))
    maximum_gain = 10 ** (maximum_gain_db / 20)
    gains = np.ones(len(mono), dtype=np.float32)
    repaired_count = 0
    index = 0
    while index < frame_count:
        if not dropout[index]:
            index += 1
            continue
        start = index
        while index < frame_count and dropout[index]:
            index += 1
        length = index - start
        if length < 6 or length > maximum_frames:
            continue
        start_sample = int(boundaries[start])
        end_sample = int(boundaries[index])
        segment_rms = float(
            np.sqrt(np.mean(np.square(mono[start_sample:end_sample]), dtype=np.float64))
        )
        if segment_rms <= 1e-8:
            continue
        gain = min(maximum_gain, threshold * 1.05 / segment_rms)
        if gain <= 1:
            continue
        gains[start_sample:end_sample] *= gain
        fade_samples = min(round(0.02 * sample_rate), start_sample, len(mono) - end_sample)
        if fade_samples > 0:
            phase = np.linspace(0, np.pi / 2, fade_samples, endpoint=False)
            fade_in = 1 + (gain - 1) * np.square(np.sin(phase))
            gains[start_sample - fade_samples : start_sample] *= fade_in.astype(np.float32)
            gains[end_sample : end_sample + fade_samples] *= fade_in[::-1].astype(np.float32)
        repaired_count += 1
    if repaired_count:
        repaired = audio * gains[:, np.newaxis]
        temporary = path.with_name(f".{path.stem}.dropout-repair.wav")
        sf.write(temporary, repaired, sample_rate, subtype="PCM_24")
        os.replace(temporary, path)
    return repaired_count


def _runtime_candidate_passes(record, settings, window_threshold: float) -> bool:
    """Apply every runtime-available candidate gate before deciding on retries."""
    return bool(
        float(record["minimum_window_identity_similarity"]) >= window_threshold
        and float(record["f0_median_error_cents"]) <= float(settings["f0_median_error_limit_cents"])
        and float(record["f0_gross_error_fraction"])
        < float(settings["f0_gross_error_limit_fraction"])
        and int(record["sustained_dropout_count"]) == 0
    )


def _retry_reference_ids(records, fallback_ids: list[str]) -> list[str]:
    """Retry complementary continuity-safe and identity-strong reference roles."""
    continuity_safe = sorted(
        (record for record in records if int(record["sustained_dropout_count"]) == 0),
        key=lambda record: (
            -float(record["minimum_window_identity_similarity"]),
            -float(record["mean_window_identity_similarity"]),
            int(record["candidate_index"]),
        ),
    )
    identity_strong = sorted(
        records,
        key=lambda record: (
            -float(record["minimum_window_identity_similarity"]),
            -float(record["mean_window_identity_similarity"]),
            int(record["sustained_dropout_count"]),
            float(record["f0_gross_error_fraction"]),
            int(record["candidate_index"]),
        ),
    )
    result = []
    for ranked in (continuity_safe, identity_strong):
        for record in ranked:
            reference_id = str(record["reference_id"])
            if reference_id not in result:
                result.append(reference_id)
    for reference_id in fallback_ids:
        if reference_id not in result:
            result.append(reference_id)
    return result


def _pitch_contour(path: Path, f0_fn, device, torchaudio) -> tuple[np.ndarray, float]:
    """Extract one RMVPE contour with the already-loaded Seed-VC pitch model."""
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True).float()
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    duration = waveform.shape[-1] / 16000
    with torch.no_grad():
        values = f0_fn(waveform.to(device)[0], thred=0.03)
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1), duration


def _pitch_metrics(
    target: np.ndarray,
    converted: np.ndarray,
    duration_seconds: float,
    maximum_gap_ms: int,
) -> tuple[float, float]:
    """Measure pitch after repairing only short detector gaps and isolated octaves."""
    usable = min(len(target), len(converted))
    if usable <= 0 or duration_seconds <= 0:
        raise ValueError("Seed-VC candidate pitch contour is empty.")
    frame_rate = usable / duration_seconds
    maximum_gap_frames = round(maximum_gap_ms * frame_rate / 1000)
    target_values = _clean_pitch(target[:usable], maximum_gap_frames)
    converted_values = _clean_pitch(converted[:usable], maximum_gap_frames)
    target_voiced = target_values > 0
    if not np.any(target_voiced):
        raise ValueError("Seed-VC target phrase contains no measurable pitch.")
    both_voiced = target_voiced & (converted_values > 0)
    cents = np.full(usable, np.inf, dtype=np.float64)
    cents[both_voiced] = np.abs(
        1200 * np.log2(converted_values[both_voiced] / target_values[both_voiced])
    )
    median_error = float(np.median(cents[both_voiced])) if np.any(both_voiced) else 1_000_000.0
    return median_error, float(np.mean(cents[target_voiced] > 200))


def _clean_pitch(values: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    """Repair short F0 detector gaps while retaining genuine longer rests."""
    cleaned = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    index = 0
    while index < len(cleaned):
        if cleaned[index] > 0:
            index += 1
            continue
        start = index
        while index < len(cleaned) and cleaned[index] <= 0:
            index += 1
        end = index
        if (
            end - start <= maximum_gap_frames
            and start > 0
            and end < len(cleaned)
            and cleaned[start - 1] > 0
            and cleaned[end] > 0
        ):
            endpoint_distance = abs(1200 * np.log2(cleaned[end] / cleaned[start - 1]))
            if endpoint_distance <= 700:
                interpolation = np.geomspace(
                    float(cleaned[start - 1]),
                    float(cleaned[end]),
                    end - start + 2,
                    dtype=np.float64,
                )
                cleaned[start:end] = interpolation[1:-1].astype(np.float32)
    for index in range(1, len(cleaned) - 1):
        previous, current, following = cleaned[index - 1 : index + 2]
        if min(previous, current, following) <= 0:
            continue
        neighbor_distance = abs(1200 * np.log2(following / previous))
        current_distance = min(
            abs(1200 * np.log2(current / previous)),
            abs(1200 * np.log2(current / following)),
        )
        if neighbor_distance <= 100 and current_distance >= 700:
            cleaned[index] = np.sqrt(previous * following)
    return cleaned


def _leave_one_out_similarities(ids, embeddings, torch_module) -> dict[str, float]:
    """Measure each real reference against the centroid of the remaining references."""
    if len(ids) == 1:
        return {ids[0]: 1.0}
    similarities = {}
    for reference_id in ids:
        others = torch_module.stack([embeddings[item] for item in ids if item != reference_id])
        centroid = torch_module.nn.functional.normalize(others.mean(dim=0), dim=0)
        similarities[reference_id] = float(
            torch_module.dot(embeddings[reference_id], centroid).detach().cpu()
        )
    return similarities


def _adaptive_identity_threshold(
    similarities: tuple[float, ...],
    margin: float,
    minimum: float,
) -> float:
    """Set a conservative identity floor from real-reference variation."""
    if not similarities:
        return minimum
    return float(np.clip(min(similarities) - margin, minimum, 1.0))


def _adaptive_window_identity_threshold(
    similarities: tuple[float, ...],
    margin: float,
    minimum: float,
) -> float:
    """Require generated windows to exceed the weakest real reference by a margin."""
    if not similarities:
        return minimum
    return float(np.clip(min(similarities) + margin, minimum, 1.0))


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated deterministic inference parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--output")
    parser.add_argument("--batch-request")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--fp16", action="store_true")
    return parser


def main() -> None:
    """Run one Seed-VC operation."""
    arguments = build_parser().parse_args()
    if arguments.batch_request:
        batch_convert(arguments)
        return
    missing = [name for name in ("source", "target", "output") if getattr(arguments, name) is None]
    if missing:
        raise ValueError("Missing required single-conversion arguments: " + ", ".join(missing))
    convert(arguments)


if __name__ == "__main__":
    main()
