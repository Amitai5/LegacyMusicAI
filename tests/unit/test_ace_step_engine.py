from pathlib import Path
from typing import Any

import pytest

from legacy_music.engines.ace_step import AceStepApiEngine, AceStepError


def test_configure_adapter_when_requested_adapter_is_active_reuses_it(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    engine = AceStepApiEngine(
        adapter_path=adapter,
        adapter_name="training-test",
    )
    calls: list[tuple[str, dict[str, Any] | None]] = []

    monkeypatch.setattr(engine, "health", lambda: {"models_initialized": True})
    monkeypatch.setattr(
        engine,
        "lora_status",
        lambda: {
            "lora_loaded": True,
            "active_adapter": "training-test",
        },
    )
    monkeypatch.setattr(
        engine,
        "unload_lora",
        lambda: calls.append(("unload", None)),
    )
    monkeypatch.setattr(
        engine,
        "load_lora",
        lambda path, adapter_name: calls.append(
            ("load", {"path": str(path), "adapter_name": adapter_name})
        ),
    )
    monkeypatch.setattr(
        engine,
        "_request_json",
        lambda endpoint, payload=None, method="POST": calls.append((endpoint, payload)),
    )

    engine._configure_adapter()

    assert calls == [
        (
            "v1/lora/scale",
            {"scale": 1.0, "adapter_name": "training-test"},
        ),
        ("v1/lora/toggle", {"use_lora": True}),
    ]


def test_configure_base_when_training_state_cannot_unload_explains_restart(
    monkeypatch: Any,
) -> None:
    def fail_unload() -> None:
        raise AceStepError("ACE-Step HTTP 400: Base decoder backup not found")

    engine = AceStepApiEngine()
    monkeypatch.setattr(engine, "health", lambda: {"models_initialized": True})
    monkeypatch.setattr(
        engine,
        "lora_status",
        lambda: {"lora_loaded": True, "active_adapter": "training-test"},
    )
    monkeypatch.setattr(engine, "unload_lora", fail_unload)

    with pytest.raises(AceStepError, match="Restart the external ACE-Step service"):
        engine._configure_adapter()
