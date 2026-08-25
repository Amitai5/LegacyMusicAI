# Model storage

This directory holds shared foundation-model weights and caches at runtime. Its contents are ignored by Git.

```text
models/
|-- shared/
|   |-- ace-step/
|   |-- soulx/
|   `-- seed-vc/
`-- cache/
```

Artist-specific adapters and voice checkpoints belong under the corresponding local artist profile, never in the shared model directory.
