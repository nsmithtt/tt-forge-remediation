# Remediation Summary: grm_coder_14b_i1_gguf-causal_lm-pytorch-Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[grm_coder_14b_i1_gguf/causal_lm/pytorch-Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

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
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(surface output: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`)

Second failure after switching to the target branch:
```
NameError: name 'gguf_path' is not defined
```
in `granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py:58`

## Root cause
Two layered loader bugs:

1. **Cross-loader contamination (TypeError)**: The original submodule checkout (`0f7b734348`) had 26 GGUF loaders with narrow-signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Transformers 5.2.0 now calls `load_gguf_checkpoint` with a `model_to_load` kwarg, so when any of these wrappers was installed at pytest collection time, the `grm_coder_14b_i1_gguf` test hit the TypeError. The target branch (`worktree-aknezevic+hf-bringup_1023-2`) already fixes all 26 loaders.

2. **Incomplete body update (NameError)**: The target branch commit `fde82c3752` changed the signature of `granite_3_1_2b_instruct_gguf`'s `_patched_load_gguf_checkpoint` from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` but forgot to update the function body — the call on line 58 still referenced `gguf_path` and `return_tensors` as explicit variables, causing a `NameError` at runtime.

## Fix
Fixed in `tt_forge_models` on branch `remediation/grm_coder_14b_i1_gguf-causal_lm-pytorch-granite_4_0_h_1b_Q4_K_M_GGUF-single_device-inference`:

- **File**: `granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py` (line 58)
- **Change**: Updated the call in the `_patched_load_gguf_checkpoint` body from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)` to match the already-updated signature.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    536.17s (0:08:56)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f12100e8e830d9365a0c64d0092ef8ace2897e16 |
| tt-forge-models | 5231b938beb1df3a8c71fec7b22c29d2e23d2a99 |
