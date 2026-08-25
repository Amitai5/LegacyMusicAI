# ACE-Step environment

`scripts/install-models.ps1` checks out the reviewed ACE-Step commit in `config/upstreams.yaml`, runs its locked `uv` installation, downloads the official weights into `models/shared/ace-step/`, and creates the required `vendor/ACE-Step-1.5/checkpoints` junction.

Generation starts a short-lived loopback service by default. Use `scripts/start-ace-step.ps1` for training or repeated `--external-ace` generation. The script binds only to `127.0.0.1`, enables CPU offload, disables compilation, and leaves the optional language model disabled unless requested.
