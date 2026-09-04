# Seed-VC environment

Seed-VC is the optional artist-specific singing fine-tuning and conversion runtime. Evaluate
it after documenting an inadequate zero-shot comparison and obtaining voice-training
authorization. SoulX remains the default. The pinned upstream is recorded as archived and
GPL-3.0-only in `config/upstreams.yaml`, and it loads model checkpoints through PyTorch
serialization. Keep it isolated and load only trusted checkpoints.

On Windows, install the reviewed commit and CUDA runtime with:

```powershell
pwsh -File scripts/install-models.ps1 -SkipAceStep -SkipSoulX -IncludeSeedVC
```

The installer uses Python 3.10, PyTorch 2.4.0 with CUDA 12.1, and the direct dependency pins in `requirements-runtime.txt`. Training uses batch size 1 and `num_workers=0` for the tested RTX 4070 Laptop GPU with 8 GB VRAM. Model caches remain project-local; artist datasets, checkpoints, logs, and selections remain under the ignored artist profile.

The installer creates the runtime, not a trained artist voice. Supporting pretrained weights
may download on first training/conversion into `models/cache/seed-vc/`; this environment is
not automatically forced offline. SoulX is still required for source separation and F0 QC.
Set `SEED_VC_PYTHON` or pass `--seed-vc-python` for a nonstandard interpreter location.

After stem preparation, cleanup, and listening review:

```powershell
uv run legacy-music voice prepare-training example-artist --segments-per-song 5
uv run legacy-music voice train-model example-artist <voice-dataset-id> --steps 1000 --save-every 250 --select
uv run legacy-music voice remix-run <run-id> --engine seed-vc --reference auto --identity-gated --duration 30
```

The dataset uses hash-verified `clean_vocals.wav` excerpts and keeps validation excerpts out
of training. Training requires both current `training` and `singing_voice` permissions.
The zero-shot assessment itself is a human review step. `--select` explicitly chooses the
completed checkpoint but does not change the default engine or approve a release.

Identity-gated mode loads Seed-VC, RMVPE, and CAMPPlus once, processes candidates sequentially,
and uses overlapping local identity windows alongside pitch and dropout gates. Five initial
candidates may expand to 15 for difficult phrases. This increases inference time and retained
audio substantially; review the short comparison before removing `--duration 30` for a full run.
Scores are not a percentage of voice accuracy, and all outputs still require listening review.
