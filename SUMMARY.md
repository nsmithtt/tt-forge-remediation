# Remediation Summary: informer-pytorch-tourism_monthly-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[informer/pytorch-tourism_monthly-single_device-inference]

## Result
FAIL — loader fix applied (attention mask trimming); residual PCC=-0.785 is a Tier B compiler bug in TT's handling of InformerProbSparseAttention

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
informer-probsparse-attention-advanced-index-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):
```
ValueError: Attention mask should be of size (1, 1, 12, 12), but is torch.Size([1, 1, 24, 24])
```

Post-fix failure (on TT silicon):
```
PCC comparison failed. Calculated: pcc=-0.7852107433462095. Required: pcc=0.99
```

## Root cause

**Loader bug (fixed):** In transformers 5.x, `create_bidirectional_mask` returns `None` in eager mode (the `_ignore_bidirectional_mask_sdpa` path short-circuits when not tracing), but returns a real 4D tensor `(batch, 1, seq_len, seq_len)` during `torch.compile` tracing (because `is_tracing()` blocks the short-circuit). The `InformerEncoder.forward` loop creates this mask once for the full sequence length (24), then passes it unchanged to all encoder layers. The Informer `distil=True` default has conv layers between encoder layers that halve the sequence each time (24→12→6→3 for 4 encoder layers). The second and subsequent encoder layers then fail with a shape mismatch: `(1,1,24,24)` mask vs `(1,1,12,12)` expected.

Fix: Patch `InformerEncoder.forward` to trim `attention_mask[:, :, :new_len, :new_len]` after each conv layer reduces the sequence length.

**Compiler bug (unfixed, Tier B):** After the loader fix, the model runs to completion but produces PCC=-0.785 on TT silicon. The `InformerProbSparseAttention.forward` uses advanced-indexing scatter writes: `context[dim_for_slice, top_u_sparsity_measurement, :] = attn_output` to update context tensors based on the top-U sparsity measurement indices. The anti-correlation in the output (PCC=-0.785, near -1.0) indicates either sign inversion or index addressing errors in the TT lowering of these scatter operations. CPU eager and CPU torch.compile both produce PCC > 0.9999 relative to each other, confirming the computation is correct on CPU; the failure is TT-hardware-specific.

## Fix

**Loader fix (applied):** `tt-xla/third_party/tt_forge_models/informer/pytorch/loader.py`

Added `_patch_informer_encoder(encoder)` function that monkey-patches `InformerEncoder.forward` to trim the 4D attention mask after each distil conv layer. Required new imports: `types`, `torch.nn as nn`, `BaseModelOutput` from `transformers.modeling_outputs`, `create_bidirectional_mask` from `transformers.masking_utils`. Patch is applied in `ModelLoader.load_model()` after `model.eval()`.

Remediation branch: `remediation/informer-pytorch-tourism_monthly-single_device-inference` in `tenstorrent/tt-forge-models`.

**Compiler bug fix (proposed, Tier B):** The advanced-indexing scatter in `InformerProbSparseAttention.forward` (lines 503-504 of `modeling_informer.py`):
```python
context[dim_for_slice, top_u_sparsity_measurement, :] = attn_output
```
This pattern requires correct lowering of scatter writes with index tensors derived from `torch.topk` applied to a query-key matrix product. The fix would live in tt-mlir's StableHLO→TTIR lowering for `stablehlo.scatter` or equivalent, or in tt-metal's runtime kernel for scatter updates.

## Tier B justification

**Indicator: cross-cutting**

The PCC=-0.785 failure involves `InformerProbSparseAttention.forward`, which uses `torch.randint` for random key sampling, `torch.topk` for top-U sparsity selection, and advanced-indexing scatter writes to update a context tensor. The anti-correlation pattern (PCC near -1 rather than a small positive number) suggests either sign inversion in the attention score computation or a systematic index permutation in the scatter. Diagnosing and fixing this requires silicon-level investigation of how XLA→StableHLO→TTIR lowers this specific multi-step operation combination; it is not a single missing lowering pattern.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    81.02s (second run, with loader fix)
- Tier A attempts: 0

## Files changed
- `tt-xla/third_party/tt_forge_models/informer/pytorch/loader.py` — added `_patch_informer_encoder`, new imports, patched encoder in `load_model`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | db2e5a9f66b10ea47fd9e3c789e38e90bfec785c |
