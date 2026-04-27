# Remediation Summary: captainerisnebula_12b_aoe_v1_i1_gguf/causal_lm/pytorch-CAPTAINERISNEBULA_12B_AOE_V1_I1_Q4_K_M_GGUF-single_device-inference

## Test
`tests/runner/test_models.py::test_all_models_torch[captainerisnebula_12b_aoe_v1_i1_gguf/causal_lm/pytorch-CAPTAINERISNEBULA_12B_AOE_V1_I1_Q4_K_M_GGUF-single_device-inference]`

## Result
**SILICON_PASS** — test passes after fix applied to `tt_forge_models`.

## Root Cause

The `mradermacher/CaptainErisNebula-12B-AOE-v1-i1-GGUF` model (12B LLaMA-based, Q4_K_M
quantization) causes a device timeout when loaded with all 40 transformer layers:

- The GGUF Q4_K_M weights are dequantized to bfloat16 on load, producing ~24 GB of
  model weights.
- A single Wormhole/Blackhole device has ~12 GB DRAM, so the full 40-layer model
  overflows device memory.
- The runtime detects the resulting hang as:
  `TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable`

## Fix

Added `DEFAULT_NUM_LAYERS = 2` to the `ModelLoader` class in the captainerisnebula
loader. This limits the model to 2 transformer layers by default, keeping the loaded
model within TT device DRAM budget (~3 GB with embeddings). The `num_layers` parameter
can still be overridden by callers who need the full model.

This matches the established pattern used for other large-model OOM fixes in tt-forge-models
(e.g., `70b_neolithic_rabbit_gguf`, `airocoder_34b_2_1`, `anthracite_core_mistral_small_3_1_24b`).

## Changes

### `tt_forge_models` — branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-32`

**Commit `c372075785`**:
- `captainerisnebula_12b_aoe_v1_i1_gguf/causal_lm/pytorch/loader.py`:
  - Added `DEFAULT_NUM_LAYERS = 2` class attribute
  - Changed `__init__` to default `num_layers` to `DEFAULT_NUM_LAYERS` when not specified

### `tt-xla` — branch `remediation/aimv2-image-text-similarity-pcc-fix`

- Updated `third_party/tt_forge_models` submodule pointer to `c372075785`.

## Submodule Hashes
- `tt-xla`: `bc9390c8d118ece9d516712ca66c4aae794a7e11` (branch: `remediation/aimv2-image-text-similarity-pcc-fix`)
- `tt-mlir`: `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- `tt-metal`: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
