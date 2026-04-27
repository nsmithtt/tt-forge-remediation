# Remediation Summary: c4ai_command_r7b_12_2024_abliterated_gguf/causal_lm/pytorch-c4ai_command_r7b_12_2024_abliterated_GGUF-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[c4ai_command_r7b_12_2024_abliterated_gguf/causal_lm/pytorch-c4ai_command_r7b_12_2024_abliterated_GGUF-single_device-inference]

## Result
FAIL — compiler-frontend bug: `aten.slice.Tensor` with out-of-bounds negative start index is rejected by XLA compilation even though PyTorch allows it (clamping to 0)

## Failure
Original reported failure (2026-04-25):
```
2026-04-25 03:08:42.044 | critical |          Always | TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology. Inter-mesh mapping failed after 2 attempt(s). Logical meshes being mapped: [0] (1 total). Physical meshes available: [0] (1 total). Failed mesh pair configurations tried: 1 out of 1 possible combinations. Inter-mesh validation mode: STRICT. Solver error: Mapping validation failed: 1 target node(s) are not mapped to any global node: 0. Failed mesh pairs from previous attempts: [(logical=0, physical=0)].. Either relax pinnings or modify the MGD. If this is unexpected, run ./build/test/tt_metal/tt_fabric/test_system_health to check connectivity. (assert.hpp:104)
```

Current failure (2026-04-27, blocking reproduction of original):
```
RuntimeError: Value out of range (expected to be in range of [-74, 73], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_3, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  transformers/cache_utils.py:214, in DynamicSlidingWindowLayer.update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

## Root cause

Two distinct issues are present:

**Issue 1 (loader, already fixed on hf-bringup-25 branch):** The GGUF file uses architecture `cohere2`, which is present in gguf-py but absent from transformers 5.x's `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`. This caused `ValueError: GGUF model with architecture cohere2 is not supported yet.` before the cohere2 patch (`0dc7044e47` in tt-forge-models `ip-172-31-23-5-tt-xla-dev/ubuntu/hf-bringup-25`) was applied.

**Issue 2 (compiler frontend, unfixed):** After the GGUF loader fix, XLA graph compilation fails because `transformers/cache_utils.py:214` (in `DynamicSlidingWindowLayer.update`) produces a `slice.Tensor` with start index `-self.sliding_window + 1 = -4095` on a tensor dimension of 74 (the tokenized input length). PyTorch semantics allow out-of-bounds negative slice starts by clamping to 0, but the XLA/torch_xla compilation pipeline rejects them, raising RuntimeError.

The Cohere2 model has `sliding_window=4096` in its config. When the input sequence is shorter than the sliding window (128 tokens max_length vs 4096 sliding window), the `DynamicSlidingWindowCache` generates `full_value_states[:, :, -4095:, :]`, which is valid PyTorch but illegal in XLA.

This is the same class of bug as SDPA chunk-size limits — a compiler constraint that model code legally exercises with PyTorch but that the XLA lowering does not handle.

## Fix

**Issue 1 fix** (already committed, not on main): Monkey-patch `GGUF_CONFIG_MAPPING`, `GGUF_SUPPORTED_ARCHITECTURES`, and `GGUF_TO_FAST_CONVERTERS` at loader import time to register cohere2 architecture — see `0dc7044e47` in tt-forge-models.

**Issue 2 proposed fix (compiler frontend, in tt-xla or torch_xla):** When lowering `aten.slice.Tensor` to XLA's slice op, clamp the start index to the valid range `[-dim_size, dim_size-1]` to match PyTorch semantics. The relevant lowering is in `torch_xla/_dynamo/dynamo_bridge.py` or the XLA lowering layer.

Alternatively, the transformers `DynamicSlidingWindowLayer.update()` could be patched upstream to use `max(-self.sliding_window + 1, -seq_len)` as the slice start, making the graph XLA-friendly.

## Verification
Test FAILED with RuntimeError (slice bounds) at compilation stage. Did not reach device execution. Wall clock: ~7m28s. Hardware: n150.

## Files changed
None — Issue 2 is a compiler-stack bug that cannot be fixed in the loader without a forbidden workaround.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | eda47fdf3855d357e3750a0d2cee509a7e23673f |
