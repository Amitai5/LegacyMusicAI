# ACE-Step environment

`scripts/install-models.ps1` checks out the reviewed ACE-Step commit in `config/upstreams.yaml`, runs its locked `uv` installation, downloads the official weights into `models/shared/ace-step/`, and creates the required `vendor/ACE-Step-1.5/checkpoints` junction.

Generation starts a short-lived loopback service by default. Use `scripts/start-ace-step.ps1` for training or repeated `--external-ace` generation. The script binds only to `127.0.0.1`, enables CPU offload, disables compilation, and leaves the optional language model disabled unless requested.

The registered `config/patches/ace-step-windows-training.patch` is applied after verifying
the source revision. It supports low-VRAM Windows training and progress reporting. The
installer refuses unrelated local upstream edits rather than discarding them.

ACE-Step LoRAs model musical style; they are separate from optional Seed-VC voice-identity
checkpoints. Accompaniment datasets must use verified separated stems, not full vocal mixes
labeled as instrumental. Restart an external ACE service after training before switching
adapters or returning to the base model. See [training](../../docs/TRAINING.md).
