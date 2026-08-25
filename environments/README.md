# Environment isolation

The lightweight orchestration application uses Python 3.11 and `uv`. Large model engines must remain in separate environments to avoid incompatible CUDA, PyTorch, and Python dependencies.

| Environment | Manager | Purpose |
| --- | --- | --- |
| Application | `uv` | CLI, configuration, manifests, orchestration, and tests |
| ACE-Step | `uv` | Music generation, preprocessing, and LoRA training |
| SoulX-Singer | Conda | Primary zero-shot singing voice conversion |
| Seed-VC | Conda or isolated virtual environment | Optional voice conversion and fine-tuning |

The application will invoke engine adapters through structured subprocess contracts before considering any in-process integration.
