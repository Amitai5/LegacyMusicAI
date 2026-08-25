# SoulX-Singer environment

`scripts/install-models.ps1` creates a dedicated Python 3.10 Conda environment named `soulxsinger`, installs the reviewed CUDA/PyTorch and runtime pins from `requirements-runtime.txt`, and downloads only the SVC, separator, RMVPE, and Whisper assets required by the implemented path.

The application locates the standard Conda environment automatically. Set `SOULX_PYTHON` to an explicit interpreter when using a nonstandard location. Hugging Face, Numba, Matplotlib, and Weights & Biases state is redirected to ignored project-local caches; inference runs offline after installation.
