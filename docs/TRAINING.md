# Training

ACE-Step training is artist-specific. The shared base checkpoint is reused, while every LoRA dataset, run, checkpoint, evaluation, and selected adapter remains under one artist root.

The implemented 8 GB starting preset uses LoRA rank 16, batch size 1, gradient accumulation 4, gradient checkpointing, CPU offloading, ten epochs, and five-epoch checkpoints. Operators can raise rank and epochs explicitly; these are starting values, not guaranteed optimal settings.

The pinned ACE-Step revision is installed with [`config/patches/ace-step-windows-training.patch`](../config/patches/ace-step-windows-training.patch). It disables multiprocessing data-loader workers on Windows, avoiding a spawn deadlock, and reports every optimizer step so long low-VRAM runs remain observable. The installer verifies the exact pinned revision before applying the patch and refuses unrelated changes in the upstream checkout.

`train prepare` builds labeled input from the verified catalog. `train run` drives the official ACE-Step scan, label validation, tensor preprocessing, and LoRA endpoint. It can wait and export or continue in the background for `train status` and `train finalize`.

The final epoch is never silently selected. `--select` or `train select` is an explicit operator action that hashes the complete adapter directory. Generation verifies that digest, serializes LoRA state changes through a local lock, unloads any prior artist adapter, and rejects arbitrary adapter paths.

Restart the external ACE-Step service after a training session and before generation. The upstream training lifecycle can retain LoRA bookkeeping without the base-decoder backup required to switch adapters; a fresh service restores a known model state. Default managed generation already starts from a fresh process.

A ten-epoch test run with rights-cleared material should prove preprocessing compatibility, memory fit, checkpoint persistence, adapter loading, and fixed-prompt draft quality before a longer run. The repository cannot perform that material-dependent acceptance without operator-provided authorized songs.

## Optional singing-voice fine-tuning

Seed-VC fine-tuning is implemented separately from ACE-Step style training. It remains opt-in
after a documented inadequate zero-shot evaluation. The CLI checks current `training` and
`singing_voice` permissions; the zero-shot assessment is an operator review, not an automated
model-quality decision.

Run `voice prepare-stems`, `voice clean-vocals`, and `voice prepare-training` for the artist
before `voice train-model <artist-id> <voice-dataset-id> --select`. Only verified
`clean_vocals.wav` excerpts passing the current lead-vocal policy and quality checks enter the
prepared voice dataset. Held-out validation excerpts, where available, are not fed to training.
Changes to the music-style dataset do not automatically retrain or select a new voice model.

The optional runtime uses the 44.1 kHz F0-conditioned singing preset, batch size 1, and
`num_workers=0`. Step counts are experimental settings, not promises of identity accuracy.
Selecting a checkpoint does not change the default voice engine or authorize distribution.
See [Seed-VC setup](../environments/seed-vc/README.md) and [voice evaluation](VOICE.md).
