# Generation

The implemented generation strategy creates one complete ACE-Step draft, separates its vocal and accompaniment, converts only the vocal identity, then aligns and remixes the result. Independent accompaniment and vocal generation is intentionally deferred to preserve synchronization.

Every request records the artist ID, prompt, lyrics file and hash, adapter, duration, tempo, key, seed, voice engine, voice reference, output format, and intermediate-retention choice. Variants receive independent seeds and immutable run directories.

`legacy-music generate` supports deterministic stock or selected-LoRA generation and optional authorized voice conversion. It owns a short-lived ACE process by default so 8 GB GPUs release ACE memory before SoulX starts. `--external-ace` reuses an operator-managed service for stock or high-memory workflows.

`runs list` and `runs show` inspect durable history. Resume, reproduction, variants, and preset selection remain backlog items. No generation command publishes an output, and every provenance record begins with `release_approved: false`.
