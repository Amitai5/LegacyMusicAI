"""Short-lived ACE-Step API process for low-VRAM sequential pipelines."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from legacy_music.domain.generation import MusicGenerationRequest
from legacy_music.engines.ace_step import AceStepApiEngine, AceStepError


class ManagedAceStepEngine:
    """Start ACE-Step for one draft, then release all of its GPU allocations."""

    def __init__(
        self,
        project_root: Path,
        base_url: str,
        api_key: str | None,
        model: str,
        thinking: bool,
        adapter_path: Path | None,
        adapter_name: str | None,
        session_lock: Path,
        startup_timeout_seconds: float = 300,
    ) -> None:
        self.project_root = project_root.resolve()
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.thinking = thinking
        self.adapter_path = adapter_path
        self.adapter_name = adapter_name
        self.session_lock = session_lock
        self.startup_timeout_seconds = startup_timeout_seconds
        self.last_metadata: dict[str, object] = {}

    def generate(self, request: MusicGenerationRequest, lyrics: str, output: Path) -> Path:
        """Generate through an owned loopback process and always release it afterward."""
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise AceStepError("Managed ACE-Step requires an HTTP IPv4 loopback URL.")
        if _is_reachable(self.base_url):
            raise AceStepError(
                "Managed ACE-Step cannot own the configured port because a server is already "
                "running; stop it or pass --external-ace."
            )

        repository = self.project_root / "vendor/ACE-Step-1.5"
        executable = repository / ".venv/Scripts/acestep-api.exe"
        checkpoint = repository / "checkpoints/acestep-v15-turbo/model.safetensors"
        if not executable.is_file() or not checkpoint.is_file():
            raise AceStepError(
                "ACE-Step is incomplete; run scripts/install-models.ps1 before generation."
            )

        port = parsed.port or 80
        log_path = output.parent / "ace-step-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "ACESTEP_API_HOST": "127.0.0.1",
                "ACESTEP_API_PORT": str(port),
                "ACESTEP_PROJECT_ROOT": str(repository),
                "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
                "ACESTEP_DEVICE": "cuda",
                "ACESTEP_OFFLOAD_TO_CPU": "true",
                "ACESTEP_COMPILE_MODEL": "false",
                "ACESTEP_NO_INIT": "false",
                "ACESTEP_INIT_LLM": "true" if self.thinking else "false",
                "HF_HOME": str(self.project_root / "models/cache/huggingface"),
                "NUMBA_CACHE_DIR": str(self.project_root / "models/cache/numba"),
                "MPLCONFIGDIR": str(self.project_root / "models/cache/matplotlib"),
            }
        )
        if self.api_key:
            environment["ACESTEP_API_KEY"] = self.api_key
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(executable), "--host", "127.0.0.1", "--port", str(port)],
                cwd=self.project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creation_flags,
            )
            try:
                self._wait_until_ready(process, log_path)
                engine = AceStepApiEngine(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    model=self.model,
                    thinking=self.thinking,
                    adapter_path=self.adapter_path,
                    adapter_name=self.adapter_name,
                    session_lock=self.session_lock,
                )
                generated = engine.generate(request, lyrics, output)
                self.last_metadata = engine.last_metadata
                return generated
            finally:
                _stop_process(process)

    def _wait_until_ready(self, process: subprocess.Popen[str], log_path: Path) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise AceStepError(f"Managed ACE-Step exited during startup: {detail}")
            if _is_reachable(self.base_url):
                return
            time.sleep(1)
        raise AceStepError("Managed ACE-Step did not become ready before its timeout.")


def _is_reachable(base_url: str) -> bool:
    try:
        with urlopen(base_url.rstrip("/") + "/health", timeout=0.5) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
