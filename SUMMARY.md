# Remediation Summary: gemmax2_28_9b_v0_1-causal_lm-pytorch-GemmaX2-28-9B-v0.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemmax2_28_9b_v0_1/causal_lm/pytorch-GemmaX2-28-9B-v0.1-single_device-inference]

## Result
SILICON_PASS — clamp_out_of_range_slice_starts FX pass fixes negative start index on sliding-window KV cache slice

## Stack layer
tt-xla

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
E   RuntimeError: Value out of range (expected to be in range of [-256, 255], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

Traceback from transformers/cache_utils.py:214:
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
GemmaX2-28-9B-v0.1 uses Gemma2 architecture with `sliding_window=4096`. The `SlidingWindowCache.update()` in transformers slices the KV cache with start index `-self.sliding_window + 1 = -4095`. PyTorch semantics allow `start < -dim_size` (treated as 0), but the TT/XLA backend validates strictly and rejects values outside `[-dim_size, dim_size]`. With `max_length=256`, the sequence dimension is 256, so the start `-4095` is far outside `[-256, 256]`. The error fires during `partition_fx_graph_for_cpu_fallback` when the FX interpreter tries to execute the slice op for shape analysis.

## Fix
Added `clamp_out_of_range_slice_starts` FX pass in `tt-xla`:
- `python_package/tt_torch/backend/passes.py`: new pass that walks all `aten.slice.Tensor` nodes, detects static negative starts where `start < -dim_size`, and clamps them to `-dim_size` (semantically equivalent under PyTorch slice semantics).
- `python_package/tt_torch/backend/backend.py`: imports and calls the new pass after `bypass_assert_tensor_metadata` in `torch_pass_pipeline`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    186.66s (0:03:06)
- Tier A attempts: 1

## Files changed
- tt-xla: `python_package/tt_torch/backend/passes.py`
- tt-xla: `python_package/tt_torch/backend/backend.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 253f5551dc52be8a66a71900d0abee8ab38dc895 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
