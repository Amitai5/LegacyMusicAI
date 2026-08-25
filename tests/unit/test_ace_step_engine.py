import json
from pathlib import Path
from typing import Any

import pytest

from legacy_music.domain.generation import MusicGenerationRequest
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


def test_generate_forwards_vocal_language_to_ace_step(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine = AceStepApiEngine()
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def request_json(
        endpoint: str,
        payload: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> Any:
        calls.append((endpoint, payload))
        if endpoint == "release_task":
            return {"task_id": "task-test"}
        if endpoint == "query_result":
            return [
                {
                    "status": 1,
                    "result": json.dumps([{"file": "/v1/audio?path=test.wav"}]),
                }
            ]
        raise AssertionError(f"Unexpected endpoint: {endpoint} ({method})")

    monkeypatch.setattr(engine, "_request_json", request_json)
    monkeypatch.setattr(
        engine,
        "_download_audio",
        lambda audio_url, output: output,
    )
    output = tmp_path / "generated.wav"
    request = MusicGenerationRequest(
        prompt="Persian pop with a lead vocal",
        duration_seconds=30,
        vocal_language="fa",
        seed=42,
    )

    result = engine._generate_unlocked(request, "[Verse]\nمتن تازه", output)

    assert result == output
    assert calls[0][0] == "release_task"
    assert calls[0][1] is not None
    assert calls[0][1]["vocal_language"] == "fa"
