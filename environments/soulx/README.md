# SoulX-Singer environment

`scripts/install-models.ps1` creates a dedicated Python 3.10 Conda environment named `soulxsinger`, installs the reviewed CUDA/PyTorch and runtime pins from `requirements-runtime.txt`, and downloads only the SVC, separator, RMVPE, and Whisper assets required by the implemented path.

The application locates the standard Conda environment automatically. Set `SOULX_PYTHON` to an explicit interpreter when using a nonstandard location. Hugging Face, Numba, Matplotlib, and Weights & Biases state is redirected to ignored project-local caches; inference runs offline after installation.

## Additional clean-vocal weights

The core installer does not fetch the optional Mel-Band RoFormer dereverberation checkpoint.
Before `voice clean-vocals`, download the two assets named by `SoulXInstallation` from the
repository recorded in `config/upstreams.yaml`. From the project root:

```powershell
conda run -n soulxsinger hf download anvuew/dereverb_mel_band_roformer `
  dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt `
  dereverb_mel_band_roformer_anvuew.yaml `
  --local-dir models/shared/soulx/SoulX-Singer-Preprocess/dereverb_mel_band_roformer
```

Before loading these files, compare their SHA-256 values with
`soulx_singer.dereverb_checkpoint_sha256` and `soulx_singer.dereverb_config_sha256` in
`config/upstreams.yaml`:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
  models/shared/soulx/SoulX-Singer-Preprocess/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt, `
  models/shared/soulx/SoulX-Singer-Preprocess/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml
```

Stop on a mismatch; do not replace the recorded expected hashes to accept a changed download.
The cleanup pipeline records the hashes it actually uses, but its existence check does not
perform this expected-digest comparison for the operator. The core strict doctor also does
not check these optional assets.

`voice prepare-stems` preserves source-linked `vocals.wav`, accompaniment, and F0 artifacts.
`voice clean-vocals` writes a separate `clean_vocals.wav`; it does not overwrite `vocals.wav`.
The default `training_vocals.dereverb_strength: 0.90` blends back 10% of the original stem.
After changing cleanup settings, use `voice clean-vocals example-artist --rebuild`, review
the new stems, then rebuild reference/training datasets. Final-mix reverb is independently
configured under `final_vocal_mix` and never written back into training samples.
