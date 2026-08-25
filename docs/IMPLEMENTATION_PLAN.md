# Implementation plan

This backlog translates the multi-artist implementation specification into dependency-ordered, independently reviewable tasks. Stable task IDs are mirrored in GitHub Issues. The MVP follows the specification's ten milestones; explicitly later features remain in a separate post-MVP milestone.

Live tracking: [all implementation issues](https://github.com/Amitai5/LegacyMusicAI/issues?q=is%3Aissue%20is%3Aopen%20label%3Aimplementation) and [milestone progress](https://github.com/Amitai5/LegacyMusicAI/milestones).

## Delivery rules

Every task must:

- branch from and merge back to `production` through a focused pull request;
- preserve artist isolation, immutable originals, and immutable run history;
- enforce the relevant rights capability before protected work begins;
- add deterministic unit or fake-engine integration coverage;
- avoid real network, GPU, model, or artist-data dependencies in general CI;
- update operator documentation and persisted contract versions when behavior changes; and
- record any upstream repository revision, model identifier, license, and checksum used.

Real model acceptance tests are local and opt-in. No task may add a real artist recording, private authorization evidence, model weight, generated song, credential, or personal data to Git.

## Milestone 1 - Hardware and workstation readiness

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-001 | Expand `legacy-music doctor` into a structured hardware and tool diagnostic. | Scaffold | Reports CPU, RAM, GPU, driver, CUDA, VRAM, storage, Python, Git, uv, Conda, FFmpeg, FFprobe, model environments, and artist readiness; supports human and JSON output. |
| LM-002 | Build a repeatable Linux workstation bootstrap and storage preflight. | LM-001 | Idempotent scripts verify rather than silently replace drivers; document the 12 GB VRAM preset and warn below configured free-space thresholds. |

## Milestone 2 - Foundation models

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-003 | Create a pinned, license-aware vendor installation framework. | LM-001, LM-002 | Each checkout uses an explicit reviewed revision and writes an environment manifest containing source URL, revision, license, Python, CUDA/PyTorch, model IDs, and install timestamp. |
| LM-004 | Install and smoke-test ACE-Step 1.5 in its isolated environment. | LM-003 | A stock prompt generates a WAV; no training occurs; adapter code invokes the reviewed executable without importing its dependency graph into the app. |
| LM-005 | Install and smoke-test SoulX-Singer SVC in its isolated environment. | LM-003 | A synthetic or licensed target/reference fixture completes zero-shot conversion and records a sanitized subprocess manifest. |
| LM-006 | Add the optional Seed-VC environment and stock SVC smoke test. | LM-003 | Installation remains disabled by default, uses a pinned upstream revision, and proves the generic voice-engine contract without artist data. |

## Milestone 3 - Artist management and authorization

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-007 | Implement an atomic filesystem `ArtistRepository` and profile initializer. | Scaffold | Creates the full private directory layout from validated templates, rejects duplicates, and rolls back incomplete profile creation. |
| LM-008 | Implement `artist create`, `artist list`, and `artist status`. | LM-007 | Interactive and noninteractive paths agree; status distinguishes dataset, music, voice, and authorization readiness without exposing private paths. |
| LM-009 | Implement versioned rights-manifest validation and action-specific authorization gates. | LM-007 | Pending, expired, revoked, mismatched, missing, and insufficient permissions fail closed; tests cover UTC boundaries and separate voice permission. |
| LM-010 | Enforce artist path isolation and immutable-original policies at repository boundaries. | LM-007, LM-009 | Traversal, symlink escape, sibling reads, destructive writes, and cross-artist adapter/reference selection are rejected and regression-tested. |

## Milestone 4 - First dataset and catalog

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-011 | Define versioned song/catalog manifests and crash-safe catalog persistence. | LM-007, LM-010 | Catalog updates use atomic replace and locking, remain artist-scoped, and retain source and derived hashes plus processing status. |
| LM-012 | Implement recursive ingest discovery, SHA-256 deduplication, FFprobe validation, and immutable copy. | LM-009, LM-010, LM-011 | Only authorized hashes and supported files import; originals are copied byte-for-byte; duplicates and unsupported files receive explicit results. |
| LM-013 | Implement the resumable `legacy-music ingest` command and import report. | LM-012 | Interactive and noninteractive modes produce stable song IDs, do not repeat completed copies, and survive interruption without corrupting the catalog. |

## Milestone 5 - Preparation and vocal extraction

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-014 | Implement non-destructive audio validation and normalization. | LM-013 | Derived WAVs match configured format, preserve originals and dynamics, record FFmpeg commands/hashes, and skip unchanged valid outputs. |
| LM-015 | Add a replaceable stem separator with vocal, accompaniment, F0, and separation manifests. | LM-004, LM-014 | Fake-engine tests prove path and resume behavior; local model tests verify timing and minimum required outputs. |
| LM-016 | Add lyrics transcription and assistive musical analysis. | LM-014 | Stores reviewable lyrics, BPM, key, meter, instrumentation, style, production, and vocal metadata with engine confidence and manual-review state. |
| LM-017 | Implement manual vocal-reference curation and rejection metadata. | LM-009, LM-015 | References remain artist-scoped, hashed, tagged, quality-reviewed, and selectable without committing audio. |
| LM-018 | Implement `legacy-music prepare` as a cached, resumable stage pipeline. | LM-014, LM-015, LM-016, LM-017 | `--resume` verifies prior output hashes, restarts at the first invalid stage, and never repeats an expensive valid stage. |
| LM-019 | Build deterministic ACE-Step datasets, splits, tensor caches, and manifests per artist. | LM-004, LM-011, LM-018 | Evaluation songs are excluded from initial fitting, cached preprocessing is reusable, and no path can reference another artist. |

## Milestone 6 - Voice experiment

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-020 | Implement the SoulX-Singer SVC `VoiceEngine` adapter. | LM-005, LM-009, LM-017 | Converts only after voice authorization, validates target/reference audio, captures subprocess evidence, and writes no source file in place. |
| LM-021 | Implement voice-reference selection, `voice references`, `voice test`, and the evaluation matrix. | LM-020 | `auto` deterministically falls back to the configured default; fixed tests score similarity, pronunciation, pitch stability, naturalness, and artifacts. |
| LM-022 | Record the zero-shot voice-quality decision gate. | LM-021 | A signed local evaluation declares SoulX adequate or documents why optional fine-tuning is justified; no automatic threshold enables training. |

## Milestone 7 - Quick music training

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-023 | Implement the ACE-Step LoRA training adapter and ten-epoch `--test-run`. | LM-004, LM-019 | Verifies CUDA, 12 GB memory-conscious settings, tensor compatibility, checkpoint saving/loading, and draft generation without judging artistic quality. |
| LM-024 | Add immutable training requests, state, checkpoints, resume, and subprocess observability. | LM-023 | Interrupted and failed runs remain inspectable and resume only after validating configuration, inputs, engine revision, and checkpoint hashes. |

## Milestone 8 - Full music training and evaluation

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-025 | Implement full LoRA training with configurable resource controls. | LM-023, LM-024 | The initial rank-32/100-epoch preset checkpoints every ten epochs, avoids accidental CPU-only training, and records GPU/memory telemetry. |
| LM-026 | Implement fixed-seed checkpoint benchmarks, human ratings, and atomic adapter promotion. | LM-025 | Compares style, instrumentation, composition, production, musical quality, originality, and memorization; selection never assumes the final epoch is best. |
| LM-027 | Add optional Seed-VC fine-tuning behind the authorization and quality decision gates. | LM-006, LM-009, LM-022 | Cannot run without explicit voice-training permission and an inadequate zero-shot decision; generated checkpoints remain artist-scoped. |

## Milestone 9 - End-to-end generation

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-028 | Implement immutable generation requests, run repositories, and durable state transitions. | LM-009, LM-011 | Every variant receives a unique run ID and complete request; invalid transitions, overwrite attempts, and artist mismatches fail. |
| LM-029 | Implement the ACE-Step generation adapter with artist-specific LoRA selection. | LM-026, LM-028 | Loads only the selected artist adapter, honors prompt/lyrics/duration/BPM/key/seed, and validates the complete draft output. |
| LM-030 | Orchestrate vocal separation, authorized voice conversion, timing alignment, and final mixing. | LM-015, LM-020, LM-028, LM-029 | Preserves accompaniment timing, gain-matches vocals, protects against clipping, exports lossless WAV, and retains configured intermediates. |
| LM-031 | Write complete provenance, disclosure, lineage, and export manifests. | LM-028, LM-030 | Records synthetic status, artist profile, model and adapter revisions, voice reference ID/hash, prompt, lyrics hash, seed, parents, timestamps, and output hashes. |
| LM-032 | Implement similarity, memorization, unapproved-sample, lyric, and release-approval gates. | LM-009, LM-019, LM-030, LM-031 | A failed or unreviewed gate blocks release export; automated scores remain evidence for a human decision rather than legal conclusions. |
| LM-033 | Implement interactive/noninteractive `generate`, variants, and artist presets. | LM-029, LM-030, LM-031, LM-032 | Requires explicit artist selection, produces unique seeds for variants, validates supplied lyrics, and never distributes output automatically. |
| LM-034 | Implement `runs list/show/resume/reproduce` and structured failure records. | LM-024, LM-028, LM-033 | Resume starts from the last verified stage; reproduce creates a child run; logs sanitize secrets and retain commands, exit codes, timing, and outputs. |

## Milestone 10 - Multi-artist MVP acceptance

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-035 | Execute a rights-cleared second-artist end-to-end isolation pilot. | LM-025, LM-033, LM-034 | Two profiles ingest, prepare, train, evaluate, and generate independently; deliberate cross-profile probes fail; no application code changes are needed. |
| LM-036 | Automate the full MVP definition-of-done acceptance suite. | LM-035 | Covers every checklist item from the specification with synthetic/fake CI tests and separately documented local model evidence. |
| LM-037 | Complete operations, backup, storage, retention, cleanup, and disaster-recovery guidance. | LM-034, LM-036 | Protects immutable originals and manifests, estimates capacity, safely prunes reproducible caches, and restores catalogs/runs without mixing artists. |

## Post-MVP extensions

| Task | Outcome | Depends on | Acceptance summary |
| --- | --- | --- | --- |
| LM-038 | Add score-controlled SoulX singing synthesis with MIDI melody and lyric timing. | LM-020, LM-030, LM-036 | Exact pitch/syllable timing is reproducible, remains authorization-gated, and does not alter the default synchronized MVP path. |
| LM-039 | Add a local FastAPI service over stable application use cases. | LM-033, LM-034, LM-036 | API authentication, local-only defaults, job isolation, cancellation, and manifest parity are tested; engine code remains outside request handlers. |
| LM-040 | Add the multi-artist React/Next.js review and generation interface. | LM-039 | Artist selection changes both authorized identities, long jobs are observable, disclosures are visible, and release approval remains human-controlled. |
| LM-041 | Separate reusable `Person`/voice identity from musical project profiles. | LM-035, LM-036 | Sharing is explicit and authorized; solo, band-era, and acoustic profiles retain independent music data and adapters without implicit linkage. |

## MVP completion gate

Milestone 10 is complete only when multiple authorized artist profiles coexist without data leakage; originals remain unchanged; ingestion, preparation, zero-shot voice conversion, LoRA training/evaluation, complete-song generation, voice remixing, provenance, failure resume, and second-artist onboarding all have verified evidence. The web interface and person/profile split are not MVP blockers.
