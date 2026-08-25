# Legacy Music AI

Legacy Music AI is a working, authorization-first local pipeline for creating original music with an approved legacy artist profile. It combines ACE-Step 1.5 complete-song generation and artist-specific LoRA training with SoulX-Singer vocal separation, F0 extraction, singing-voice conversion, lossless remixing, and immutable provenance.

> [!IMPORTANT]
> Only train or generate with written permission covering every applicable recording, composition, performance, name, image, likeness, and voice right. The repository contains no artist recordings, identities, consent documents, model weights, or generated songs. Technical operation is not legal clearance, and the application never publishes an output automatically.

## Working status

The local CLI now implements the complete core workflow:

- create isolated artist profiles and machine-readable authorization gates;
- inventory files by SHA-256 before import and reject every unlisted source;
- preserve originals byte-for-byte and create normalized lossless derivatives;
- prepare labeled ACE-Step datasets from a few authorized songs;
- preprocess, train, export, integrity-check, and explicitly select artist LoRAs;
- generate fixed-seed ACE-Step WAV drafts with either the base model or a selected adapter;
- curate exact-hash-authorized singing references and precompute RMVPE F0;
- separate vocals, convert them with SoulX-Singer, and remix at 48 kHz stereo/24-bit;
- persist durable run transitions, hashes, engine metadata, disclosure, and lineage; and
- list and inspect local artists, references, engines, datasets, and generation runs.

Real GPU acceptance was completed on an NVIDIA RTX 4070 Laptop GPU with 8 GB VRAM:

- ACE-Step 1.5 turbo generated a validated 48 kHz WAV;
- a two-song synthetic authorized dataset completed ACE-Step preprocessing, one-epoch LoRA training, export, integrity selection, and adapter-backed generation;
- SoulX-Singer separated a real mix, extracted F0, and converted its vocal;
- the public `legacy-music generate --voice` command completed the full pipeline; and
- the managed ACE process released VRAM before SoulX started.

Seed-VC remains an optional, disabled alternative. The React review UI and other post-MVP features remain in the GitHub backlog; they are not required for local generation.

## Prerequisites

- Windows 11 and an NVIDIA CUDA-capable GPU; 8 GB VRAM is the tested minimum;
- Python 3.11, `uv`, Git, and Conda; and
- enough disk for isolated environments and model weights (allow at least 20 GB).

FFmpeg and FFprobe are optional accelerators. The application has a tested NumPy/SoundFile fallback for validation, lossless conversion, resampling, and mixing.

## Install

Install the application plus pinned ACE-Step and SoulX-Singer environments and weights:

```powershell
pwsh -File scripts/install-models.ps1
uv run legacy-music doctor --strict --root .
```

The installer checks out reviewed upstream commits recorded in [`config/upstreams.yaml`](config/upstreams.yaml), applies the repository-owned Windows training compatibility patch, keeps incompatible PyTorch stacks isolated, downloads weights into Git-ignored `models/`, and creates the ACE-Step shared-checkpoint junction. Re-running it is safe when upstream checkouts contain only that registered patch; any other upstream modification fails closed.

The official upstreams are [ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) and [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer).

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

Use `--instrumental` instead when every source is instrumental. In a separate terminal, start the pinned local ACE-Step service:

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
  --adapter selected `
  --voice `
  --reference neutral
```

Generation manages a short-lived ACE process by default. That makes a single command work on an 8 GB GPU: ACE exits after the draft, then SoulX receives the freed VRAM. For repeated stock generations against a service started with `scripts/start-ace-step.ps1`, pass `--external-ace`.

Every result is written beneath a new ignored `runs/run-.../` directory. Inspect it with:

```powershell
uv run legacy-music runs list --artist example-artist
uv run legacy-music runs show <run-id>
```

The final output is lossless and its `provenance.json` remains `release_approved: false`. Distribution always requires a separate human decision outside this application.

## Architecture

```text
Exact-hash authorization
          |
          +-----------------------+
          |                       |
          v                       v
 Immutable song catalog     Curated voice reference
          |                       |
          v                       v
 ACE-Step dataset/LoRA       RMVPE reference F0
          |                       |
          +-----------+-----------+
                      |
             ACE-Step WAV draft
                      |
                      v
       vocal separation + target F0
                      |
                      v
             SoulX vocal conversion
                      |
                      v
          timing-preserving lossless mix
                      |
                      v
       WAV + provenance + durable run log
```

Each artist owns an independent catalog, dataset root, LoRA, voice-reference library, presets, and evaluations. Filesystem confinement, exact source hashes, adapter serialization, and fail-closed authorization prevent accidental cross-artist reuse.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/RIGHTS_AND_SAFETY.md`](docs/RIGHTS_AND_SAFETY.md), and [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md). The implementation backlog is tracked against the `production` branch.

## Development

```powershell
uv sync --locked --group dev
uv run ruff check .
uv run pytest
```

Do not commit copyrighted audio, artist profiles, consent documents, private lyrics, credentials, weights, or generated media. Upstream code and weights retain their own licenses and restrictions; software licensing does not grant music, training-data, voice, likeness, or output rights.

## License

No open-source license has been granted for this repository. See [`LICENSE`](LICENSE).
