# Remediation Summary: gemma_translate_v3_12b_i1_gguf-causal_lm-pytorch-GEMMA_TRANSLATE_V3_12B_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_translate_v3_12b_i1_gguf/causal_lm/pytorch-GEMMA_TRANSLATE_V3_12B_I1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Value out of range (expected to be in range of [-28, 27], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_51, 2, -1023, 9223372036854775807), kwargs = {})

Original traceback: transformers/cache_utils.py line 214 in SlidingWindowCache.update:
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause

Two bugs were present:

**Bug 1 (loader)**: When pytest collects all test modules, other GGUF loaders (28 of them, e.g. qwen3.5 variants) globally patch `load_gguf_checkpoint` at import time with a narrow `(gguf_path, return_tensors=False)` signature. Transformers 5.x now calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which triggers `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` even on a single-test run.

**Bug 2 (tt-xla)**: Gemma3 uses sliding-window attention via `SlidingWindowCache`. The cache update computes `start = -sliding_window + 1 = -1023` when `sliding_window=1024`. With `max_length=128`, the tensor in dim 2 has only 128 elements, so XLA's lazy backend rejects a start of `-1023` (valid range is `[-128, 127]`). PyTorch CPU silently clamps out-of-range slice indices; XLA does not.

## Fix

**Fix 1 (loader, tt_forge_models remediation branch)**: Changed the signature of all 28 `_patched_load_gguf_checkpoint` wrappers from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` so that the transformers 5.x `model_to_load` kwarg passes through transparently.

File changed: 28 loader files across various model directories (bartowski_coniccat_qwen3_5_27b_writer_gguf, daniloreddy_qwen3_5_0_8b_gguf, dmind_3_mini_i1_gguf, gpt_oss_swallow_*, mradermacher_qwen3_5_*, tvall43_qwen3_5_*, qwen_3_5_*, and others).

**Fix 2 (tt-xla, TorchFunctionOverride)**: Added a guard in `TorchFunctionOverride.__torch_function__` that clamps `aten.slice.Tensor` start and end arguments to `[-size, size]` when their absolute value exceeds the tensor's dimension size. This matches PyTorch CPU semantics and prevents the XLA "Value out of range" error for sliding-window attention patterns.

File changed: `python_package/tt_torch/torch_overrides.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    696.81s (0:11:36)
- Tier A attempts: 1

## Files changed
- `python_package/tt_torch/torch_overrides.py` (tt-xla)
- 28 GGUF loader files in `tt_forge_models` (loader)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3fdd3c25920b87b58eae4da1c7f3dd1de9afdebe |
| tt-forge-models | 8b2c67148cf6a0638037325c7d6a44d865d0021d |
