# Mechanical fixes

These changes do not alter the locked procedure, thresholds, models, drift formula, or sampling rules.

## 2026-10-14 — `src/embedder.py`

**What was wrong.** `HFEmbedder` chose only CUDA or CPU. This machine is Apple Silicon, so every model would have run on CPU.

**What changed.** If no device is passed, the embedder now uses CUDA, then Apple MPS (`torch.backends.mps.is_available()`), then CPU. MPS uses float32. CUDA still uses float16.

**Why results are unchanged.** The vectors are still the mean of the focal word’s subword states, then L2-normalized and centered. MPS is only the device that runs the same model. Float32 on MPS matches the CPU dtype already used for non-CUDA runs.
