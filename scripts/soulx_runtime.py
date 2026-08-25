"""Minimal reviewed entry point for SoulX-Singer preprocessing operations."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    return parser


def main() -> None:
    """Run exactly one requested preprocessing operation."""
    arguments = build_parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
