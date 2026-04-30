# Remediation Summary: gemma_2_9b_it_sppo_iter3-causal_lm-pytorch-gemma_2_9b_it_sppo_iter3-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_9b_it_sppo_iter3/causal_lm/pytorch-gemma_2_9b_it_sppo_iter3-single_device-inference]

## Result
SILICON_PASS

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
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

Original traceback: transformers/cache_utils.py:214 in SlidingWindowCache.update:
  self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Gemma-2 uses a sliding window KV cache (sliding_window=4096). When
seq_len < sliding_window (e.g. seq_len=128 during prefill), the cache
update slices with start = -sliding_window + 1 = -4095. PyTorch
semantics clamp negative-out-of-bounds indices to the beginning of the
dimension, but the XLA/TT backend validates that the start index is in
[-dim_size, dim_size-1] and raises RuntimeError for -4095 on a 128-element
tensor (valid range [-128, 127]).

The error surfaces inside partition_fx_graph_for_cpu_fallback when the
UnsupportedNodesCollector runs the FX graph on XLA tensors to identify
unsupported ops, before the TT compilation pass has run.

The fix is in tt-xla (compiler frontend): a new FX pass
`clamp_out_of_range_slice_starts` in torch_pass_pipeline clamps literal
negative slice starts to max(start, -dim_size) using shape metadata from
the exported graph. This produces semantically equivalent output (both
yield all elements when start is more negative than the dimension size)
and puts the value in bounds that XLA accepts.

## Fix
Added `clamp_out_of_range_slice_starts` FX pass in
`python_package/tt_torch/backend/passes.py` and invoked it in
`python_package/tt_torch/backend/backend.py::torch_pass_pipeline`.

The pass iterates over `aten.slice.Tensor` nodes with literal negative
integer start values. When shape metadata (`meta["val"]`) is available
and the dim size is a concrete int, it clamps `start = max(start, -dim_size)`.
This is applied to the post-export compiled graph before it is passed to
XLAExecutor, so the re-exported module seen by `bridge.extract_compiled_graph`
and `partition_fx_graph_for_cpu_fallback` has the in-bounds start value.

Commits on remediation branch in tt-xla:
- 259d4b1fd: add clamp_out_of_range_slice_starts FX pass

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    189.63s (0:03:09)
- Tier A attempts: 1

## Files changed
- python_package/tt_torch/backend/passes.py (new pass: clamp_out_of_range_slice_starts)
- python_package/tt_torch/backend/backend.py (import and invoke the new pass)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550 |
| tt-mlir         | 553c0632b |
| tt-xla          | 259d4b1fd |
| tt-forge-models | 0f7b734348 |
