# Environment isolation

The lightweight orchestration application uses Python 3.11 and `uv`. Large model engines must remain in separate environments to avoid incompatible CUDA, PyTorch, and Python dependencies.

| Environment | Manager | Purpose |
| --- | --- | --- |
| Application | `uv` | CLI, manifests, orchestration, bundled FFmpeg, and synthetic tests |
| ACE-Step | `uv` | Music generation, preprocessing, and LoRA training |
| SoulX-Singer | Conda, Python 3.10 | Stem separation, F0, clean-vocal preparation, and default zero-shot conversion |
| Seed-VC | Conda, Python 3.10 | Optional clean-vocal fine-tuning and identity-gated conversion |

The application invokes ACE-Step through its loopback HTTP API and owns a short-lived service
by default during generation. SoulX and Seed-VC run through structured subprocess contracts.
Artist data selection and authorization remain in the application layer; upstream engines
receive only the paths selected for that operation.

On the tested 8 GB GPU, generation releases ACE memory before conversion; phrase conversion
loads its model once and runs candidates sequentially. Do not run training or an external
ACE service concurrently when the GPU cannot hold both workloads.

The application uses `uv.lock`. The model environments use pinned upstream commits plus
their documented dependency pins; the Conda/pip installations are not complete transitive
hash lockfiles. See [ACE-Step](ace-step/README.md), [SoulX](soulx/README.md), and
[Seed-VC](seed-vc/README.md) for setup and optional downloads.
