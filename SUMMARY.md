# Remediation Summary: gpt_oss_20b_sombliterated_gguf-causal_lm-pytorch-20B_SOMbliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_sombliterated_gguf/causal_lm/pytorch-20B_SOMbliterated_GGUF-single_device-inference]

## Result
XFAIL — Hardware capacity: 20B Qwen3MoE BF16 ~40 GB exceeds all single-device DRAM (n150 12 GB, p150b 32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

Raised in `AutoModelForCausalLM.from_pretrained` when `transformers 5.2.0` passes
`model_to_load=dummy_model` to `load_gguf_checkpoint`. The function was replaced at
import time by another GGUF loader's `_patched_load_gguf_checkpoint` with the narrow
signature `(gguf_path, return_tensors=False)`.

## Root cause
Two issues:

1. **Cross-loader clobbering (TypeError)**: 26 GGUF loaders monkey-patch
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import
   time with a narrow-signature wrapper `(gguf_path, return_tensors=False)`. When
   `gpt_oss_20b_sombliterated_gguf` calls `AutoModelForCausalLM.from_pretrained`, the
   stale narrow-sig wrapper (installed by one of the 26 other loaders during pytest
   collection) is still active. `transformers 5.2.0` now passes
   `model_to_load=dummy_model`, which the narrow wrapper rejects with `TypeError`.

2. **Broken load_shard_spec (AttributeError, latent)**: The loader's `load_shard_spec`
   method accesses `layer.mlp.up_proj` on every layer. The GGUF file stores the model
   with `general.architecture = qwen3moe`, so transformers loads it as
   `Qwen3MoeForCausalLM` whose MoE layers are `Qwen3MoeSparseMoeBlock` — no `up_proj`.
   This would cause `AttributeError` when `_safely_put_workload_on_device` calls
   `shard_spec_fn(workload.model)` during device placement.

3. **Hardware capacity (XFAIL)**: After dequantization by transformers GGUF loading,
   the 20B Qwen3 MoE model is stored at BF16 precision. 20B params × 2 bytes/param
   ≈ 40 GB, which exceeds all single-device DRAM (n150: 12 GB, p150b: 32 GB).

## Fix
**tt_forge_models — narrow-sig fix** (`de008cffd4`):
- Updated `_patched_load_gguf_checkpoint` in 26 GGUF loaders from
  `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` with pass-through to
  `_orig_load_gguf_checkpoint(*args, **kwargs)`.
  (Cherry-picked from commit `09e6229219` on heretic remediation branch.)

**tt_forge_models — load_shard_spec fix** (`27d6f1eb3c`):
- Removed the broken `load_shard_spec` and `get_mesh_config` methods from
  `gpt_oss_20b_sombliterated_gguf/causal_lm/pytorch/loader.py`. The methods assumed
  LLaMA-style MLP attributes; the model is Qwen3 MoE. Neither method is needed for
  single-device inference.

**tt-xla — test config** (`4b358612d`):
- Added `KNOWN_FAILURE_XFAIL` entry for
  `gpt_oss_20b_sombliterated_gguf/causal_lm/pytorch-20B_SOMbliterated_GGUF-single_device-inference`
  in `tests/runner/test_config/torch/test_config_inference_single_device.yaml` with
  hardware capacity reason.

## Verification
- pytest exit: not-run (hardware-class XFAIL; model dequantizes to ~40 GB BF16)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gpt_oss_20b_sombliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/<26 GGUF loaders>/causal_lm/pytorch/loader.py` (narrow-sig fix)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 062a524b6e3377652ca4c90b23ff565f7a0cedb4 |
| tt-forge-models | 27d6f1eb3c6bf96567e5ff3f8e8d02eae3b3a0f4 |
