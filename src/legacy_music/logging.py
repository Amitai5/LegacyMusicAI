"""Application logging configuration."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure stable timestamped logging for the orchestration process."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
