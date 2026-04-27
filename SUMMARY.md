# Remediation Summary: anubis_mini_8b_v1_i1_gguf/causal_lm/pytorch-ANUBIS_MINI_8B_V1_I1_Q4_K_M_GGUF-single_device-inference

## Test

```
tests/runner/test_models.py::test_all_models_torch[anubis_mini_8b_v1_i1_gguf/causal_lm/pytorch-ANUBIS_MINI_8B_V1_I1_Q4_K_M_GGUF-single_device-inference]
```

## Status: SILICON_PASS

## Root Cause

Multiple issues were present in the anubis_mini_8b_v1_i1_gguf model loader:

1. **Missing `gguf` dependency** — The `requirements.txt` for this model did not list `gguf>=0.10.0`, causing import failures at runtime.

2. **GGUF version detection broken with transformers 5.x** — `transformers` caches `PACKAGE_DISTRIBUTION_MAPPING` at import time. When `gguf` is installed later (by `RequirementsManager`), the mapping is stale and version detection returns `'N/A'`, crashing `packaging.version.parse`. Fixed by patching `PACKAGE_DISTRIBUTION_MAPPING` with `gguf` and clearing the `is_gguf_available` cache.

3. **`load_gguf_checkpoint` patch incompatibility with `model_to_load` kwarg** — A newer version of `transformers` added a `model_to_load` keyword argument to `load_gguf_checkpoint`. The GGUF model loader patched this function but did not accept/forward the new kwarg, causing:

   ```
   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
   ```

## Fix

All three issues were fixed in the `tt_forge_models` branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-9`:

1. Added `gguf>=0.10.0` to `anubis_mini_8b_v1_i1_gguf/causal_lm/pytorch/requirements.txt`
2. Added `_fix_gguf_version_detection()` method and integrated it into the loader
3. Updated the `load_model` method to patch `load_gguf_checkpoint` with a closure that:
   - Traverses the full patch chain to find the real transformers function
   - Accepts and forwards `model_to_load` and any future kwargs

## Verification

Configured `tt_forge_models` to branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-9` (commit `8bbda16005`).
Test confirmed PASSED on silicon (could not reproduce the original segfault — fixes already present in branch).

## Submodule Hashes

| Submodule | Hash |
|-----------|------|
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla | 38da2b5990b8ce811fa48b65636cb96cc77e7d52 |
| tt-xla/third_party/tt_forge_models | 8bbda16005fd007031e28003b0122ae20ae2db32 |

## Key Commits in tt_forge_models (branch arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-9)

```
0daa0c6c1b Fix anubis_mini_8b_v1_i1_gguf: add model_to_load patch for load_gguf_checkpoint
327f07f4d0 Fix anubis_mini_8b_v1_i1_gguf: add gguf version detection fix for transformers 5.x
dbf3262a56 Fix anubis_mini_8b_v1_i1_gguf: add gguf dependency
```
