# Generation

The implemented generation strategy creates one complete ACE-Step draft, separates its vocal and accompaniment, converts only the vocal identity, then aligns and remixes the result. Independent accompaniment and vocal generation is intentionally deferred to preserve synchronization.

Every request records the artist ID, prompt, lyrics file and hash, adapter, duration, tempo,
key, seed, voice engine, voice reference, output format, and intermediate-retention choice.
Each CLI invocation creates a new immutable run; batch variants are not yet implemented.

`legacy-music generate` supports fixed-seed stock or selected-LoRA generation and optional authorized voice conversion. Use `--voice-engine seed-vc` to apply the artist's selected fine-tuned Seed-VC checkpoint directly to the separated guide vocal; the default remains `soulx` for compatibility. When supplying lyrics, set `--vocal-language` to their ISO 639 language code (for example, `fa` for Persian); otherwise ACE-Step defaults to English vocal conditioning. Requests and lineage are reproducible, but GPU kernels and fresh model processes are not guaranteed to produce bit-identical WAV hashes. It owns a short-lived ACE process by default so 8 GB GPUs release ACE memory before the selected voice engine starts. `--external-ace` reuses an operator-managed service for stock or high-memory workflows.

Quality-first repair is a separate, opt-in child-run path:
`legacy-music voice remix-run <run-id> --reference auto`. It reuses the completed parent draft and stems byte-for-byte,
changes only vocal conversion/mixing/mastering, and records the parent artifact hashes for a
controlled A/B. Passing objective QC does not promote this path into `generate --voice`;
promotion requires explicit human approval of the comparison.

For identity-focused Seed-VC repair, run:

```powershell
uv run legacy-music voice remix-run <run-id> --engine seed-vc --reference auto --identity-gated
```

The process loads Seed-VC once, selects an approved
reference per 6–8-second phrase, renders five deterministic candidates by default, scores
overlapping four-second active-vocal windows, and retries difficult phrases up to 15 candidates.
Local identity scores must meet the bank-calibrated window floor; where present, this gate
supersedes silence-biased whole-phrase scoring, which remains recorded for audit. Pitch,
clipping, dropout, and spectral gates also apply. The runtime repairs only recorded low-level
holes of 120 ms or less with a capped same-candidate envelope, then joins retained phrases
with equal-power crossfades. Scores are not an identity-accuracy percentage.

Without `--identity-gated`, Seed-VC uses whole-song conversion. In `generate`, `--reference auto`
uses the artist default; per-phrase bank selection applies to SoulX quality remixing and
identity-gated Seed-VC remixing. Explicit reference IDs remain supported.

Use `--duration 30` for a prefix-only smoke-test child and listen before the full-song run.
The CLI does not automatically enforce completion of that preliminary review. Seeded
candidates make decisions traceable; GPU inference is not guaranteed bit-identical across
processes, devices, or runtime changes.

Final-vocal treatment is independent of clean training-vocal preparation. The mixer restores
configurable reverb, applies presence EQ and light compression, measures active vocal balance,
and attenuates only the louder derived bus toward the configured target without boosting either
bus into clipping. `--vocal-balance-target-db` controls the active-vocal balance target.
`--vocal-reverb-wet` is available on generation and remixing; `0` disables restored reverb.
Remixing also exposes `--vocal-reverb-pre-delay-ms` and `--vocal-reverb-decay`. Every automatic
correction is retained in mastering and QC manifests. These settings never change clean
training vocals. Review the full mix as well as isolated stems; objective balance does not
guarantee perceptual intelligibility or naturalness.

`runs list` and `runs show` inspect durable history. Resume, reproduction, variants, and preset selection remain backlog items. No generation command publishes an output, and every provenance record begins with `release_approved: false`.
