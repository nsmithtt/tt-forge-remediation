# Remediation Summary: bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf/causal_lm/pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-single_device-inference]

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

Original traceback path: transformers/cache_utils.py SlidingWindowCache.update → `self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]`

## Root cause

Two bugs were present in sequence:

**Bug 1 (loader):** Before the originally reported error could be hit, a pre-existing loader pollution bug surfaced. Twenty-six GGUF loaders for Qwen3.5 / GPT-OSS models monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)`. A newer version of transformers added a `model_to_load=` keyword argument that is passed during `from_pretrained` for GGUF checkpoints. When those 26 loaders are imported at test-collection time, the monkey-patch replaces the real function, breaking any GGUF model loaded afterward (including this Gemma GGUF). Error: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

**Bug 2 (tt-xla):** After the loader issue was fixed, the originally reported error appeared. Gemma3's `SlidingWindowCache.update` performs `full_value_states[:, :, -self.sliding_window + 1 :, :]` where `sliding_window = 1024`. On the first token, the key/value cache has only 23 elements in the sequence dimension, so the start index becomes `-1023`. PyTorch silently clamps out-of-bounds negative indices (treating -1023 as 0 for a size-23 dimension). XLA's eager slice lowering (called during torch_xla's `partition_fx_graph_for_cpu_fallback` probing phase) does not clamp — it performs a strict range check `[-dim_size, dim_size-1]` and raises `RuntimeError: Value out of range (expected [-23, 22], got -1023)`.

## Fix

**Fix 1 (loader, tt_forge_models):** Cherry-picked commit `deba21f33e` (authored by Nicholas Smith, 2026-04-28) from an existing remediation branch. That commit widens the `_patched_load_gguf_checkpoint` signature from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` (or equivalently adds `model_to_load=None`) across all 26 affected GGUF loaders in `tt_forge_models`. The remediation branch in `tt_forge_models` is `remediation/bartowski_mlabonne_gemma_3_12b_it_abliterated_gguf-causal_lm-pytorch-12B_IT_Abliterated_Q4_K_M_GGUF-single_device-inference`.

**Fix 2 (tt-xla, Tier A):** In `python_package/tt_torch/torch_overrides.py`, added handling for `aten.slice.Tensor` inside the existing `TorchFunctionOverride.__torch_function__` method. When the input tensor is on an XLA device and the `start` index is an integer more negative than `-dim_size`, it is clamped to `-dim_size` before the call is forwarded to XLA. This matches PyTorch's clamping semantics and keeps the slice within the range XLA's eager lowering accepts. The fix is scoped to XLA tensors only (guarded by `tensor.device.type == "xla"`).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    250.23s (0:04:10)
- Tier A attempts: 1

## Files changed
- `tt_forge_models` (27 files): `_patched_load_gguf_checkpoint` signature widened in 26 Qwen3.5/GPT-OSS loaders; new `abhiray_qwen3_5_9b_abliterated` loader added with correct `(*args, **kwargs)` patch
- `tt-xla/python_package/tt_torch/torch_overrides.py`: Added `aten.slice.Tensor` guard in `TorchFunctionOverride.__torch_function__` to clamp OOB negative start indices on XLA tensors

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 68fb3feb244a2c7f41b6942c845d256c01d5a27f |
| tt-forge-models | 2c6b0fa4e6f396e49893a80c5d26374cd7e4cd10 |
