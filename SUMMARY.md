# Remediation Summary: param_1_2_9b_instruct/causal_lm/pytorch-param_1_2_9b_instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[param_1_2_9b_instruct/causal_lm/pytorch-param_1_2_9b_instruct-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
stablehlo-gather-collapsed-dim-concat-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Do you wish to run the custom code? [y/N] 2026-04-23 22:44:16.639 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)

The TT_FATAL eth-core message is a known transient device init warning (see memory entry `eth_core_remote_mmio_transient`); the real failures were:

1. **Compiler (Tier A)**: `ttir.concat` dimension mismatch during compilation — `Output tensor dimension 0 does not match the sum of input tensor dimensions: 1 vs. 128`.
2. **Loader**: After the compiler fix, `PCC comparison failed. Calculated: pcc=nan` — golden CPU output was NaN due to uninitialized `cos_cached`/`sin_cached` RoPE buffers.

## Root cause

### Bug 1 — tt-mlir (Tier A)

`StableHLOGatherToSliceRepeatConcatPattern` matched the `cos[position_ids]` RoPE gather (shape `[1, 128, 64]` result, source shape `[2048, 64]`).  The pattern uses `indexedDim = startIndexMap[0] = 0` (source-relative) as the ConcatOp output dimension and as the slice axis.  But dimension 0 is listed in `collapsed_slice_dims`, meaning it is **absent** from the gather output type.  `outputType.getDimSize(0)` then returns the batch dim size (1) while the concatenated slices sum to the full source size (2048), causing the `ttir.concat` verifier to fail.

### Bug 2 — loader

`ParamBharatGenRotaryEmbedding` stores `cos_cached` and `sin_cached` as non-persistent buffers, computed once in `__init__` via `_set_cos_sin_cache(dtype=torch.get_default_dtype())`.  When `from_pretrained` is called with `torch_dtype=bfloat16`, `init_empty_weights` creates the module on the meta device; the non-persistent buffers are computed on meta and are never materialized after loading.  They remain as uninitialized (NaN) tensors.  When the CPU forward pass runs first (to produce the golden reference), `cos` is NaN, so all of `q_rot`, `k_rot`, attention weights, and logits are NaN, yielding `pcc=nan`.

## Fix

### Fix 1 — tt-mlir (`StableHLOToTTIRPatterns.cpp`)

Added a guard at the top of `StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite`:

```cpp
auto collapsedSliceDims = dimensionNumbers.getCollapsedSliceDims();
int64_t indexedDim = startIndexMap[0];
if (llvm::is_contained(collapsedSliceDims, indexedDim)) {
  return rewriter.notifyMatchFailure(
      srcOp,
      "Indexed dimension is collapsed; input dim does not map to output "
      "dim at the same index");
}
```

This causes the RoPE gather to fall through to `StableHLOGatherToEmbeddingPattern`, which handles collapsed-dim gathers correctly.

Branch: `remediation/param-1-2-9b-instruct-causal_lm-pytorch-param_1_2_9b_instruct-single_device-inference` in tt-mlir.

### Fix 2 — loader (`param_1_2_9b_instruct/causal_lm/pytorch/loader.py`)

After `from_pretrained`, iterate over all modules and call `_set_cos_sin_cache` to force recomputation of the RoPE buffers with the correct dtype:

```python
target_dtype = dtype_override if dtype_override is not None else torch.get_default_dtype()
for module in model.modules():
    if (
        hasattr(module, "_set_cos_sin_cache")
        and hasattr(module, "inv_freq")
        and hasattr(module, "max_seq_len_cached")
    ):
        module._set_cos_sin_cache(
            seq_len=module.max_seq_len_cached,
            device=module.inv_freq.device,
            dtype=target_dtype,
        )
```

Branch: `remediation/param-1-2-9b-instruct-causal_lm-pytorch-param_1_2_9b_instruct-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    87.84s (0:01:27)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — guard for collapsed indexed dim in SliceRepeatConcat gather pattern
- `tt-xla/third_party/tt_forge_models/param_1_2_9b_instruct/causal_lm/pytorch/loader.py` — reinit cos/sin RoPE cache after BF16 load

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | cac620910e1647c506ddf6cf0db318086333124a |
| tt-xla          | c9d7c88daa85730aecf67b59b9f8331f4947d08e |
| tt-forge-models | 62d66d30ecf5f706e1cac6f636f48b381a18e1eb |
