"""ACE-Step 1.5 local asynchronous API adapter."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from legacy_music.audio import probe_audio
from legacy_music.domain.generation import MusicGenerationRequest
from legacy_music.persistence import file_lock


class AceStepError(RuntimeError):
    """Raised when the local ACE-Step service cannot complete a request."""


class AceStepApiEngine:
    """Generate music through the official loopback-only ACE-Step API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        api_key: str | None = None,
        model: str = "acestep-v15-turbo",
        inference_steps: int = 8,
        thinking: bool = False,
        timeout_seconds: float = 1800,
        poll_seconds: float = 2,
        adapter_path: Path | None = None,
        adapter_name: str | None = None,
        adapter_scale: float = 1.0,
        session_lock: Path | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("ACE-Step API must use an HTTP loopback address.")
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key
        self.model = model
        self.inference_steps = inference_steps
        self.thinking = thinking
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.adapter_path = adapter_path.resolve() if adapter_path is not None else None
        self.adapter_name = adapter_name
        if not 0 <= adapter_scale <= 1:
            raise ValueError("ACE-Step adapter scale must be between 0 and 1.")
        self.adapter_scale = adapter_scale
        self.session_lock = session_lock
        self.last_metadata: dict[str, Any] = {}

    def health(self) -> dict[str, Any]:
        """Return the local service health payload."""
        return self._request_json("health", method="GET")

    def models(self) -> dict[str, Any]:
        """Return models available in the running local service."""
        return self._request_json("v1/models", method="GET")

    def generate(self, request: MusicGenerationRequest, lyrics: str, output: Path) -> Path:
        """Submit, poll, download, and validate one deterministic WAV generation."""
        lock = file_lock(self.session_lock) if self.session_lock is not None else nullcontext()
        with lock:
            self._configure_adapter()
            return self._generate_unlocked(request, lyrics, output)

    def load_lora(self, path: Path, adapter_name: str | None = None) -> dict[str, Any]:
        """Load one reviewed local LoRA directory into the running service."""
        resolved = path.resolve()
        if not resolved.exists():
            raise AceStepError(f"ACE-Step adapter does not exist: {resolved}")
        payload: dict[str, Any] = {"lora_path": str(resolved)}
        if adapter_name:
            payload["adapter_name"] = adapter_name
        return self._request_json("v1/lora/load", payload)

    def unload_lora(self) -> dict[str, Any]:
        """Restore base-model generation and clear prior artist adapter state."""
        return self._request_json("v1/lora/unload", {})

    def lora_status(self) -> dict[str, Any]:
        """Return the local service's current LoRA state."""
        return self._request_json("v1/lora/status", method="GET")

    def _configure_adapter(self) -> None:
        health = self.health()
        if not health.get("models_initialized", False):
            self._request_json(
                "v1/init",
                {
                    "model": self.model,
                    "slot": 1,
                    "init_llm": self.thinking,
                },
            )
        status = self.lora_status()
        is_loaded = bool(status.get("lora_loaded", False))
        if self.adapter_path is None:
            if is_loaded:
                self.unload_lora()
            return
        is_requested_adapter_active = (
            is_loaded
            and self.adapter_name is not None
            and status.get("active_adapter") == self.adapter_name
        )
        if not is_requested_adapter_active:
            if is_loaded:
                self.unload_lora()
            self.load_lora(self.adapter_path, self.adapter_name)
        self._request_json(
            "v1/lora/scale",
            {
                "scale": self.adapter_scale,
                "adapter_name": self.adapter_name,
            },
        )
        self._request_json("v1/lora/toggle", {"use_lora": True})

    def _generate_unlocked(
        self,
        request: MusicGenerationRequest,
        lyrics: str,
        output: Path,
    ) -> Path:
        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "lyrics": lyrics,
            "audio_format": "wav",
            "audio_duration": request.duration_seconds,
            "inference_steps": self.inference_steps,
            "use_random_seed": False,
            "seed": request.seed,
            "batch_size": 1,
            "model": self.model,
            "thinking": self.thinking,
            "use_cot_caption": self.thinking,
            "use_cot_language": self.thinking,
        }
        if request.bpm is not None:
            payload["bpm"] = request.bpm
        if request.key is not None:
            payload["key_scale"] = request.key

        submission = self._request_json("release_task", payload)
        task_id = submission.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise AceStepError("ACE-Step did not return a task ID.")

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status_payload = self._request_json(
                "query_result",
                {"task_id_list": [task_id]},
            )
            if not isinstance(status_payload, list) or not status_payload:
                raise AceStepError("ACE-Step returned an empty task status response.")
            task = status_payload[0]
            status = task.get("status")
            if status == 1:
                result_value = task.get("result")
                try:
                    results = (
                        json.loads(result_value)
                        if isinstance(result_value, str)
                        else result_value
                    )
                    result = results[0]
                    audio_url = result["file"]
                except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                    raise AceStepError(f"ACE-Step returned an invalid result: {error}") from error
                self.last_metadata = {
                    "task_id": task_id,
                    "dit_model": result.get("dit_model", self.model),
                    "lm_model": result.get("lm_model"),
                    "seed_value": result.get("seed_value"),
                    "metas": result.get("metas", {}),
                    "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                    "adapter_name": self.adapter_name,
                    "adapter_scale": self.adapter_scale if self.adapter_path else None,
                }
                return self._download_audio(audio_url, output)
            if status == 2:
                detail = task.get("result", "unknown error")
                raise AceStepError(f"ACE-Step generation failed: {detail}")
            if time.monotonic() >= deadline:
                raise AceStepError(f"ACE-Step generation timed out for task {task_id}.")
            time.sleep(self.poll_seconds)

    def _request_json(
        self,
        endpoint: str,
        payload: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            urljoin(self.base_url, endpoint),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=30) as response:
                wrapper = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise AceStepError(f"ACE-Step HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise AceStepError(f"Unable to call the local ACE-Step service: {error}") from error
        if isinstance(wrapper, dict) and "code" not in wrapper:
            return wrapper
        if not isinstance(wrapper, dict) or wrapper.get("code") != 200:
            detail = wrapper.get("error") if isinstance(wrapper, dict) else "invalid response"
            raise AceStepError(f"ACE-Step API error: {detail}")
        return wrapper.get("data")

    def _download_audio(self, audio_url: str, output: Path) -> Path:
        resolved_url = urljoin(self.base_url, audio_url)
        parsed_base = urlparse(self.base_url)
        parsed_audio = urlparse(resolved_url)
        if (parsed_audio.scheme, parsed_audio.hostname, parsed_audio.port) != (
            parsed_base.scheme,
            parsed_base.hostname,
            parsed_base.port,
        ):
            raise AceStepError("ACE-Step returned an audio URL outside its local service.")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(resolved_url, headers=headers, method="GET")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            with urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            probe_audio(temporary)
            os.replace(temporary, output)
            return output
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            temporary.unlink(missing_ok=True)
            raise AceStepError(f"Unable to download ACE-Step output: {error}") from error
