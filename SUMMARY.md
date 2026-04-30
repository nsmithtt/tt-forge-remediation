# Remediation Summary: gemma_sea_lion_v3_9b_it-causal_lm-pytorch-9B_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_sea_lion_v3_9b_it/causal_lm/pytorch-9B_Instruct-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-xla

## Tier
N/A

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
Original traceback:
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Gemma SEA-LION v3 9B uses Gemma 2 architecture with `sliding_window=4096`.
`SlidingWindowCache.update()` emits `full_value_states[:, :, -sliding_window+1:, :]`
which produces start=-4095. With `max_length=256`, the seq_len dimension is 256
and the start -4095 is outside the XLA-permitted range `[-256, 255]`. PyTorch CPU
silently clamps out-of-bounds negative indices; the XLA lazy backend raises
`RuntimeError: Value out of range` instead.

The error fires inside `partition_fx_graph_for_cpu_fallback` in the torch_xla
dynamo bridge, during graph replay on XLA tensors — before tt-mlir compilation.
The fix belongs in the tt-xla layer: clamp start/end in `TorchFunctionOverride`
before the XLA tensor sees the index.

## Fix
Added out-of-bounds negative index clamping to `TorchFunctionOverride.__torch_function__`
in `tt-xla/python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor`,
the start and end indices are clamped to `max(index, -size)` for statically-known
integer dimension sizes. This matches Python/PyTorch eager semantics without
changing the computation on valid inputs.

Remediation branch: `remediation/gemma_sea_lion_v3_9b_it-causal_lm-pytorch-9B_Instruct-single_device-inference`
in `tenstorrent/tt-xla`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    201.89s (0:03:21)
- Tier A attempts: N/A

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ca98e2777c1743b853a89b1f04878eb240744fed |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
