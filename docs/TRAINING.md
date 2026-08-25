# Training

ACE-Step training is artist-specific. The shared base checkpoint is reused, while every LoRA dataset, run, checkpoint, evaluation, and selected adapter remains under one artist root.

The initial 12 GB VRAM preset from the specification uses LoRA rank 32, batch size 1, 8-bit AdamW, encoder offloading, gradient checkpointing, and checkpoint intervals of ten epochs. These are starting values, not guaranteed optimal settings.

A ten-epoch test run must prove CUDA execution, preprocessing compatibility, memory fit, checkpoint persistence, adapter loading, and draft generation before a full run. Full training compares checkpoints with fixed prompts and seeds; the final epoch is never selected automatically.

Custom voice fine-tuning is outside the default MVP path. It requires explicit voice-training authorization and a documented finding that zero-shot SoulX conversion is inadequate.
