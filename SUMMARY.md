# Remediation Summary: bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-tensor_parallel-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf/causal_lm/pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-tensor_parallel-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-oob-negative-start-xla

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_51, 2, -1023, 9223372036854775807), kwargs = {})

Original traceback: transformers/cache_utils.py SlidingWindowCache.update → `self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]`

## Root cause

Two bugs were present in sequence (identical to the single-device variant):

**Bug 1 (loader):** Multiple GGUF loaders (Qwen3.5 and GPT-OSS models) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.x calls `load_gguf_checkpoint` with an additional `model_to_load=dummy_model` keyword argument during `AutoModelForCausalLM.from_pretrained` for GGUF checkpoints. When those loaders are imported at test-collection time, the monkey-patch replaces the real function, breaking any GGUF model loaded afterward (including this Gemma GGUF). Error: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

**Bug 2 (tt-xla):** After fixing Bug 1, Gemma3's `SlidingWindowCache.update` performs `full_value_states[:, :, -self.sliding_window + 1 :, :]` where `sliding_window = 1024`. On the first token, the KV cache sequence dimension has only 23 elements, so the start index becomes `-1023`. PyTorch silently clamps out-of-bounds negative indices (treating -1023 as 0 for a size-23 dimension). XLA's eager slice lowering performs a strict range check `[-dim_size, dim_size-1]` and raises `RuntimeError: Value out of range (expected [-23, 22], got -1023)`.

## Fix

Both fixes are identical to those applied in the single-device remediation (branch `remediation/bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-single_device-inference`) and were cherry-picked directly.

**Fix 1 (loader, tt_forge_models):** Widens `_patched_load_gguf_checkpoint` signature from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` across 26 affected GGUF loaders. Commit `2c6b0fa4e624601439af2f63adda9d3e70132675` on the existing `remediation/bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-single_device-inference` branch in `tt_forge_models`.

**Fix 2 (tt-xla, Tier A):** In `python_package/tt_torch/torch_overrides.py`, added handling for `aten.slice.Tensor` inside `TorchFunctionOverride.__torch_function__`. When the input tensor is on an XLA device and the `start` index is more negative than `-dim_size`, it is clamped to `-dim_size` before the call is forwarded to XLA. This matches PyTorch's clamping semantics. Cherry-picked as commit `1bd15c9c1` on the new tensor_parallel remediation branch.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    242.19s (0:04:02)
- Tier A attempts: 1

## Files changed
- `tt_forge_models` (27 files): `_patched_load_gguf_checkpoint` signature widened in 26 Qwen3.5/GPT-OSS loaders (existing commit reused from single-device remediation)
- `tt-xla/python_package/tt_torch/torch_overrides.py`: Added `aten.slice.Tensor` guard in `TorchFunctionOverride.__torch_function__` to clamp OOB negative start indices on XLA tensors

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5d57728f1629267ae129029cdcccc4d44d579eb1 |
| tt-forge-models | 2c6b0fa4e624601439af2f63adda9d3e70132675 |
