# Legacy Music AI

Legacy Music AI is an authorization-first local music-generation and singing-voice research pipeline. It combines ACE-Step 1.5 song generation and music-style LoRA training with SoulX-Singer preprocessing, optional artist-fine-tuned Seed-VC conversion, and quality-controlled vocal mixing. Each artist has isolated data, models, permissions, and provenance.

> [!IMPORTANT]
> Only train or generate with written permission covering every applicable recording, composition, performance, name, image, likeness, and voice right. The repository contains no artist recordings, identities, consent documents, model weights, or generated songs. Technical operation is not legal clearance, and the application never publishes an output automatically.

## Working status

The Python CLI implements the core local workflow; model installation and authorized input data are required. Music-style training and voice-identity training are separate:

| Stage | Responsibility |
| --- | --- |
| ACE-Step + artist LoRA | Generate the arrangement, lyrics performance, and guide melody; not a dedicated voice-identity model |
| SoulX-Singer / optional Seed-VC | Convert the guide singing voice; Seed-VC supports clean-vocal fine-tuning and identity-gated phrase selection |
| Training-vocal preparation | Preserve isolated vocals, create separate cleaned vocals, and screen lead-only excerpts |
| Final mixing | Adjust vocal presence, EQ, compression, balance, and restored reverb independently of training cleanup |

Implemented capabilities include:

- create isolated artist profiles and machine-readable authorization gates;
- inventory files by SHA-256 before import and reject every unlisted source;
- preserve originals byte-for-byte and create normalized lossless derivatives;
- enforce independent permanent-source, music-style, and lead-vocal dataset policies;
- prepare labeled ACE-Step datasets from authorized recordings or verified accompaniment stems;
- preprocess, train, export, integrity-check, and explicitly select artist LoRAs;
- generate fixed-seed ACE-Step WAV drafts with either the base model or a selected adapter;
- preserve `vocals.wav` and create a separate `clean_vocals.wav` for reference banks and voice training;
- curate exact-hash-authorized references and five-role register/energy banks;
- fine-tune and explicitly select an artist-specific Seed-VC singing checkpoint;
- remix immutable child runs with overlapping phrases, seeded candidates, pitch/dropout checks, and optional local identity scoring;
- gain-match, restore configurable reverb, balance the lead vocal, and two-pass EBU R128 master at 48 kHz stereo/24-bit;
- persist durable run transitions, hashes, engine metadata, disclosure, and lineage; and
- list and inspect local artists, references, engines, datasets, and generation runs.

Local GPU evaluations on an NVIDIA RTX 4070 Laptop GPU with 8 GB VRAM have exercised:

- ACE-Step 1.5 turbo generated a validated 48 kHz WAV;
- a two-song synthetic authorized dataset completed ACE-Step preprocessing, one-epoch LoRA training, export, integrity selection, and adapter-backed generation;
- SoulX-Singer separated a real mix, extracted F0, and converted its vocal;
- the `legacy-music generate --voice` command completed the full pipeline;
- the managed ACE process released VRAM before voice conversion started;
- a quality-first 180-second child remix passed loudness, peak, pitch, dropout, join,
  duration, format, provenance, and parent-immutability gates; and
- a 190-second fine-tuned, identity-gated Seed-VC child remix passed all 14 automated QC
  checks, including local identity scoring and vocal balance.

These are local evaluations, not a guarantee for every song or GPU. Voice similarity scores
are screening signals, not percentages of identity accuracy. Timbre drift, weak guide notes,
pronunciation errors, and residual separation effects can still require listening review.
No pipeline guarantees a perfect clone from a few songs.

SoulX remains the default engine. Seed-VC and identity-gated remixing are opt-in; neither a
selected checkpoint nor passing QC automatically promotes a new default or approves release.
The React review UI, resume/reproduce commands, variants, and preset selection remain backlog work.

## Prerequisites

- Windows 11, PowerShell 7, and an NVIDIA CUDA-capable GPU; 8 GB VRAM is the tested configuration, not a guarantee for every setting;
- Python 3.11, `uv`, Git, and Conda; and
- enough disk for isolated environments and model weights (allow at least 20 GB, plus additional space for datasets, checkpoints, and retained candidates).

`imageio-ffmpeg` supplies the locked FFmpeg binary used for high-quality resampling,
compression, mixing, limiting, and two-pass mastering. An explicitly configured FFmpeg
executable takes priority, followed by a system binary and then the bundled binary. FFprobe
is optional; WAV validation falls back to SoundFile. The bundled binary adds download size
and a native dependency to maintain, but avoids requiring a separate system FFmpeg install.
The application dependency versions and hashes are recorded in `uv.lock`; model environments
have their own source revisions and dependency pins.

## Install

From a new checkout, install the application plus pinned ACE-Step and SoulX-Singer
environments and the core weights:

```powershell
git clone --branch production https://github.com/Amitai5/LegacyMusicAI.git
cd LegacyMusicAI
pwsh -File scripts/install-models.ps1
uv run legacy-music doctor --strict --root .
```

The installer checks out reviewed upstream commits recorded in [`config/upstreams.yaml`](config/upstreams.yaml), applies the repository-owned Windows training compatibility patch, keeps incompatible PyTorch stacks isolated, downloads weights into Git-ignored `models/`, and creates the ACE-Step shared-checkpoint junction. Re-running it is safe when upstream checkouts contain only that registered patch; any other upstream modification fails closed.

The core installer does not download the additional dereverberation checkpoint used by
`voice clean-vocals`. Follow the [SoulX environment setup](environments/soulx/README.md)
before preparing clean vocals. The strict doctor checks the core ACE/SoulX installation;
it does not certify every optional runtime or perceptual output quality.

Upstreams: [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5),
[SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), and optional
[Seed-VC](https://github.com/Plachtaa/seed-vc). See [environment isolation](environments/README.md)
for dependency, download, and runtime boundaries.

## Configure one authorized artist

Create a private profile:

```powershell
uv run legacy-music artist create example-artist --name "Example Artist" --singing-voice
```

Inventory candidate recordings without importing them:

```powershell
uv run legacy-music ingest example-artist C:\rights-cleared\songs --inventory-only
```

Review `artists/example-artist/rights.yaml`. An authorized operator must set the status, validity window, separate permissions, approval evidence references, and the exact SHA-256 of every approved source. The CLI cannot grant rights to itself.

After approval, import the songs:

```powershell
uv run legacy-music ingest example-artist C:\rights-cleared\songs
```

For vocal training songs, place one reviewed UTF-8 lyric file per source in a private directory. Its filename must match the source filename without the audio extension.

## Train a music-style adapter

Prepare a vocal dataset from the imported catalog:

```powershell
uv run legacy-music train prepare example-artist `
  --tag exampleartiststyle `
  --caption "warm soul arrangement, live rhythm section, expressive lead vocal" `
  --vocal `
  --lyrics-dir C:\rights-cleared\lyrics
```

For accompaniment-only style training, run `voice prepare-stems example-artist` first, then
use `train prepare` with `--instrumental` instead of `--vocal` and omit `--lyrics-dir`.
Instrumental preparation fails closed when a stem or its source lineage is missing or
modified; it never silently relabels a full vocal mix as instrumental. In a separate
terminal, start the pinned local ACE-Step service:

```powershell
pwsh -File scripts/start-ace-step.ps1
```

Run the low-VRAM training preset and explicitly select the result after it completes:

```powershell
uv run legacy-music train run example-artist <dataset-id> `
  --epochs 10 `
  --rank 16 `
  --wait `
  --select
```

Training output, checkpoints, tensors, and the selected adapter stay under that artist's ignored private root. `train status`, `train finalize`, and `train select` support a background or manually reviewed workflow. Adapter selection records a deterministic directory digest and generation refuses a modified selection.

## Add an authorized singing reference

First list the reference recording's exact hash in the artist rights manifest with evidence and enable the separate `singing_voice` permission. Then run:

```powershell
uv run legacy-music voice add-reference example-artist neutral `
  C:\rights-cleared\references\neutral.wav `
  --tags neutral,mid-register
```

The command validates authorization before copying, preserves the source and lineage hash, creates a 24 kHz mono derivative, and extracts SoulX-compatible F0. References cannot be supplied as arbitrary generation-time paths.

Build a quality-screened `neutral`, `soft`, `strong`, `low`, and `high` bank from existing
authorized studio stems:

```powershell
uv run legacy-music voice prepare-stems example-artist
uv run legacy-music voice clean-vocals example-artist
uv run legacy-music voice build-bank example-artist
```

Each approved prompt is 12–18 seconds with voiced-content, level, pitch-range, clipping,
spectral-quality, source-lineage, and review metadata. `vocals.wav` remains the original
separated stem; `clean_vocals.wav` is the dereverberated derivative used by the bank and voice
training. The default cleanup blends 90% dereverberated audio with 10% original stem audio;
this is a processing blend, not a measured percentage of reverb removed. Lead-vocal screening
uses voiced density, continuity, level, spectral, and stereo-center metrics alongside reviewed
exclusion ranges. These heuristics do not reliably identify every backing vocal or ad-lib:
audition retained excerpts and exclude unsafe ranges or entire songs in the private
`data/dataset-policy.yaml`. Live/concert labels are also screened. See [dataset policy](docs/DATASET.md).

## Fine-tune the singing voice (optional)

After a documented zero-shot failure and explicit voice-training authorization, install the
optional pinned Seed-VC runtime and fine-tune only on clean stems:

```powershell
pwsh -File scripts/install-models.ps1 -SkipAceStep -SkipSoulX -IncludeSeedVC
uv run legacy-music voice prepare-training example-artist --segments-per-song 5
uv run legacy-music voice train-model example-artist <voice-dataset-id> `
  --steps 1000 `
  --save-every 250 `
  --select
```

The pinned Seed-VC upstream is recorded as archived and GPL-3.0-only in the upstream manifest;
it stays optional and isolated. Its dataset, checkpoints, logs, selection, and hashes remain
private under the artist profile. Initial training/conversion may download supporting weights
into local caches. The zero-shot comparison is an operator review step, while current training
and singing-voice permissions are enforced by the CLI. See the [Seed-VC guide](environments/seed-vc/README.md).

## Generate music

Create an instrumental with the stock model:

```powershell
uv run legacy-music generate example-artist `
  --prompt "original cinematic soul instrumental, brass accents, steady pocket" `
  --lyrics examples/instrumental.txt `
  --duration 30 `
  --adapter base
```

Create a song with the selected music adapter and authorized singing reference:

```powershell
uv run legacy-music generate example-artist `
  --prompt "original uptempo soul song, live horns, syncopated rhythm section" `
  --lyrics C:\private\new-original-lyrics.txt `
  --duration 60 `
  --vocal-language en `
  --adapter selected `
  --voice `
  --reference neutral
```

Set `--vocal-language` to the lyrics' language, for example `fa` for Persian. To use the
selected voice fine-tune during generation, add `--voice-engine seed-vc`. This whole-song path
does not enable identity-gated phrase selection; use the child-remix workflow below for that.

Generation manages a short-lived ACE process by default: ACE exits after the draft, then the
voice engine receives the freed VRAM. For repeated stock generations against a service started
with `scripts/start-ace-step.ps1`, pass `--external-ace`. In `generate`, `--reference auto`
resolves the artist's default reference; per-phrase bank selection belongs to quality remixing.

## Quality-remix an existing run

Repair only the vocal path of a completed run while preserving its exact draft, lyrics,
arrangement, accompaniment, timing, and parent hashes:

```powershell
uv run legacy-music voice remix-run <run-id> --reference auto
```

For SoulX remixing, `auto` chooses an approved bank reference by phrase register and energy. The runtime
fills only accidental F0 gaps up to 120 ms, preserves longer rests, disables global pitch
shifting, renders two deterministic candidates for each 12–20-second phrase, rejects candidate
dropouts/pitch failures, and joins phrases with one-second equal-power crossfades. SoulX loads
once and runs sequentially in FP16 for the tested 8 GB GPU.

First run with `--duration 30` and review that local GPU smoke test before a full remix; this
is an operator workflow, not an automatically enforced prerequisite. A completed child
must be exactly timed, unclipped, at −14 LUFS ±0.5 LU and at or below −1 dBTP, with no sustained
voiced dropout, no continuously voiced join jump above 6 dB, median F0 error at or below 50
cents, and gross pitch error below 5%. All phrase candidates, leveled vocals, premaster,
mastering data, QC, and versioned provenance remain in the child run. The established
`generate --voice` path remains unchanged until a human approves an A/B comparison.

To evaluate the selected clean-data Seed-VC fine-tune with identity-focused phrase selection:

```powershell
uv run legacy-music voice remix-run <run-id> `
  --reference auto `
  --engine seed-vc `
  --identity-gated
```

Identity-gated mode uses 6–8-second phrases with one-second overlaps, five initial seeded
candidates, and adaptive retries capped at 15 per phrase. CAMPPlus scores overlapping
four-second active-vocal windows against the approved bank; the pipeline retains a candidate
only when its identity and audio-quality gates pass. Short low-level holes may receive a
recorded, same-candidate repair up to 120 ms; true silence or longer gaps still fail.
This mode takes substantially more inference time and disk space than whole-song conversion.
Without `--identity-gated`, Seed-VC uses whole-song conversion rather than per-phrase selection.

Tune the final mix independently of training cleanup with `--vocal-balance-target-db`,
`--vocal-presence-db`, `--instrumental-gain-db`, `--vocal-reverb-wet`,
`--vocal-reverb-pre-delay-ms`, and `--vocal-reverb-decay`. A wet value of `0` disables restored
reverb. Defaults in `config/app.yaml` target +1.5 dB active vocal-to-instrumental balance and
a 0.08 reverb wet mix; use ignored `config/local.yaml` for machine-local overrides.
These are starting points for listening review, not guarantees of a natural balance.

Every result is written beneath a new ignored `runs/run-.../` directory. Inspect it with:

```powershell
uv run legacy-music runs list --artist example-artist
uv run legacy-music runs show <run-id>
```

The final output is lossless and its `provenance.json` remains `release_approved: false`. Distribution always requires a separate human decision outside this application.

## Architecture

```text
          Exact-hash-authorized catalog
                 |                  |
                 v                  v
      Music-style dataset      Clean lead vocals
                 |                  |
                 v                  v
          ACE-Step + LoRA       Reference bank +
                 |          optional Seed-VC fine-tune
                 v                  |
            WAV draft               |
                 |                  |
                 v                  |
       Separate stems + F0          |
                 |                  |
                 +---------+--------+
                           |
                           v
             SoulX / Seed-VC conversion + QC
                           |
                           v
             Vocal EQ / compression / reverb / balance
                           |
                           v
             EBU R128 master + QC + provenance
```

Each artist owns an independent catalog, dataset root, LoRA, voice-reference library, presets, and evaluations. Filesystem confinement, exact source hashes, adapter serialization, and fail-closed authorization prevent accidental cross-artist reuse.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/RIGHTS_AND_SAFETY.md`](docs/RIGHTS_AND_SAFETY.md), and [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md). The implementation backlog is tracked against the `production` branch.

## Repository storage policy

Git contains the orchestration workflow, configuration contracts, documentation, scripts, source, and synthetic tests. Under `artists/`, only the explanatory README and `_template/` contract are committed; every real artist profile is ignored recursively, including its manifests, recordings, derived audio, datasets, references, checkpoints, adapters, evaluations, and logs.

Under `models/`, only the README that documents the expected runtime layout is committed. Downloaded foundation models, caches, and trained artifacts stay local. `runs/` and installed `vendor/` checkouts are also local runtime state, and common checkpoint/model formats are ignored anywhere in the working tree as a defense-in-depth safeguard. See [`artists/README.md`](artists/README.md), [`models/README.md`](models/README.md), and [`runs/README.md`](runs/README.md).

## Development

```powershell
uv sync --locked --group dev
uv run ruff check .
uv run pytest
```

The current synthetic unit/integration suite covers authorization, dataset policy, source
integrity, training contracts, F0 repair, candidate selection, crossfades, gain matching,
mastering, and immutable parent handling. GitHub CI runs lint and tests on `production` pushes
and pull requests without artist data or GPU models. Real-artist A/B listening and GPU
acceptance remain separate, private operator checks.

Do not force-add copyrighted audio, artist profiles, consent documents, private lyrics, credentials, weights, or generated media. Ignore rules reduce accidental staging but do not replace operator review. Upstream code and weights retain their own licenses and restrictions; software licensing does not grant music, training-data, voice, likeness, or output rights.

## License

No open-source license has been granted for this repository. See [`LICENSE`](LICENSE).
