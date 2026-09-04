# Model storage

This directory holds shared foundation-model weights and caches at runtime. Only this README is committed; every downloaded or generated payload below `models/` is ignored by Git.

```text
models/
|-- shared/
|   |-- ace-step/
|   |-- soulx/
|   `-- seed-vc/
`-- cache/
```

Git does not preserve empty directories, so this diagram is the versioned structure and the installer creates the runtime directories as needed. Common checkpoint and trained-model formats are also ignored outside `models/` as a defense-in-depth safeguard against misplaced artifacts.

Artist-specific adapters, preprocessing tensors, and voice checkpoints belong under the corresponding ignored artist profile, never in the shared model directory. Do not force-add model payloads; model identifiers, revisions, licenses, and checksums belong in configuration or manifests instead.

The core installer downloads ACE-Step and SoulX SVC/separator/F0 weights. Clean-vocal
preparation additionally requires the dereverberation assets documented in the
[SoulX setup](../environments/soulx/README.md). Seed-VC uses `models/cache/seed-vc/` for its
supporting model caches and may download them on first use; installing its environment does
not prefetch every weight. `shared/seed-vc/` is a reserved layout, not a required populated directory.

Source commits and selected asset digests are recorded in `config/upstreams.yaml`. This is
not a claim that every upstream weight download is revision-pinned or checksum-verified by
the installer. Review upstream model licenses and only load trusted checkpoints; optional
runtime dependencies and model serialization have a larger security surface than the CLI.
