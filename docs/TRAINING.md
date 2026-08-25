# Training

ACE-Step training is artist-specific. The shared base checkpoint is reused, while every LoRA dataset, run, checkpoint, evaluation, and selected adapter remains under one artist root.

The implemented 8 GB starting preset uses LoRA rank 16, batch size 1, gradient accumulation 4, gradient checkpointing, CPU offloading, ten epochs, and five-epoch checkpoints. Operators can raise rank and epochs explicitly; these are starting values, not guaranteed optimal settings.

`train prepare` builds labeled input from the verified catalog. `train run` drives the official ACE-Step scan, label validation, tensor preprocessing, and LoRA endpoint. It can wait and export or continue in the background for `train status` and `train finalize`.

The final epoch is never silently selected. `--select` or `train select` is an explicit operator action that hashes the complete adapter directory. Generation verifies that digest, serializes LoRA state changes through a local lock, unloads any prior artist adapter, and rejects arbitrary adapter paths.

A ten-epoch test run with rights-cleared material should prove preprocessing compatibility, memory fit, checkpoint persistence, adapter loading, and fixed-prompt draft quality before a longer run. The repository cannot perform that material-dependent acceptance without operator-provided authorized songs.

Custom voice fine-tuning is outside the default MVP path. It requires explicit voice-training authorization and a documented finding that zero-shot SoulX conversion is inadequate.
