# Remediation Summary: cadrille/pytorch-2B-single_device-inference

## Test
`tests/runner/test_models.py::test_all_models_torch[cadrille/pytorch-2B-single_device-inference]`

## Result
**SILICON_PASS** — test passes after two fixes applied to `tt_forge_models`.

## Root Cause

The `maksimko123/cadrille` fine-tune of Qwen2-VL-2B has two issues:

1. **Missing `preprocessor_config.json`**: The cadrille checkpoint does not ship a
   `preprocessor_config.json`, so `AutoProcessor.from_pretrained('maksimko123/cadrille')`
   fails with an `OSError`. Fix: load the processor from the base
   `Qwen/Qwen2-VL-2B-Instruct` model instead.

2. **Visual encoder exceeds TT L1 memory**: The Qwen2-VL-2B visual encoder
   (embed_dim=1280, depth=32 blocks) cannot be compiled for TT hardware. When running
   on TT device the visual encoder's `rot_pos_emb` calls `torch.arange(h)` with a TT
   device tensor scalar, causing `RuntimeError: Bad StatusOr access: INTERNAL: Error
   code: 13`. Fix: keep the visual encoder on CPU using an `_apply()` override and a
   `@torch._dynamo.disable` graph break, matching the pattern used for `adavar/pytorch`
   (Qwen2.5-VL-7B).

## Changes

### `tt_forge_models` — branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-46`

**Commit `91bdb0214f`** (pre-existing on branch):
- `cadrille/pytorch/loader.py`: Load processor from `Qwen/Qwen2-VL-2B-Instruct` base
  model (`_BASE_PROCESSOR_NAME`) instead of `maksimko123/cadrille`.

**Commit `ec9f38a832`** (new):
- `cadrille/pytorch/loader.py`: Add `use_fast=False` to suppress
  `Qwen2VLImageProcessor` fast-processor breaking-change warning.
- `cadrille/pytorch/src/model.py`: Rewrite `Wrapper` to keep visual encoder on CPU:
  - `_apply()` override pops `visual` from the module tree during device transfers
  - `@torch._dynamo.disable` on `_precompute_embeddings` creates a graph break
  - Visual encoder runs eagerly on CPU; `inputs_embeds` and `position_ids` are
    pre-computed on CPU, then moved to TT device for the language model forward pass

### `tt-xla` — branch `remediation/aimv2-image-text-similarity-pcc-fix`

- Updated `third_party/tt_forge_models` submodule pointer to `ec9f38a832`.

## Submodule Hashes
- `tt-xla`: `2f6259589d57770e518dc0418352b72c9ed3d41d` (branch: `remediation/aimv2-image-text-similarity-pcc-fix`)
- `tt-mlir`: `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- `tt-metal`: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
