# Remediation Summary: longformer-token_classification-pytorch-OpenMed-PII-French-ClinicalLongformer-Base-149M-v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longformer/token_classification/pytorch-OpenMed-PII-French-ClinicalLongformer-Base-149M-v1-single_device-inference]

## Result
FAIL — Tier A gather fix applied and committed; blocked by Tier B embedding CB L1 overflow (same class as Longformer_Zh and Longformer Large reports)

## Stack layer
tt-mlir, tt-metal

## Tier
B

## Bug fingerprint
embedding-rm-weight-row-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before Tier A fix):
```
ValueError: Error code: 13
```
at `torch_xla._XLAC._xla_warm_up_cache` — compilation failure

Root cause: `StableHLOGatherToSliceRepeatConcatPattern` mishandled the
`token_type_embeddings` gather (table shape `[1, 768]`, `maxIndex=0`),
producing a concat with `2*N-1 = 1023` elements instead of N=512.

Residual failure after Tier A fix:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=1)] grow to 3263488 B which is beyond max L1 size of 1572864 B
```
at `EmbeddingsDeviceOperation` during the first on-device inference call.

## Root cause

**Bug 1 (Tier A — fixed):**
`StableHLOGatherToSliceRepeatConcatPattern` in tt-mlir
(`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`).

The `token_type_embeddings` weight table has shape `[1, 768]`.
`maxIndex = inputShape[indexedDim] - sliceSizes[indexedDim] = 1 - 1 = 0`.
With `maxIndex == 0`, every gather index simultaneously satisfies
`(index == 0)` AND `(index == maxIndex)`, so the `starts` and `ends`
counters both increment for every one of the 512 indices, yielding
`starts = 512`, `ends = 512`, then after the mandated `starts--`/`ends--`:
`starts = 511`, `ends = 511`. The resulting concat would have
`511 + 1 + 511 = 1023` elements instead of the expected 512, causing:

```
loc("gather.50"): error: 'ttir.concat' op Output tensor dimension 0 does
not match the sum of input tensor dimensions: 512 vs. 1023.
```

Fix: added `if (maxIndex == 0) return notifyMatchFailure(...)` before the
counting loop so that `StableHLOGatherToEmbeddingPattern` (benefit=1)
handles the singleton-dimension case correctly.

**Bug 2 (Tier B — not fixed):**
After the gather fix, the model compiles but fails at runtime with an
embedding CB L1 overflow identical to the one documented in
`longformer_zh-feature_extraction-pytorch-Longformer_Zh-single_device-inference`
and `longformer_masked_lm_pytorch_large_4096_single_device_inference`:

```
TT_THROW: Statically allocated circular buffers on core range
[(x=0,y=0) - (x=0,y=1)] grow to 3263488 B which is beyond max L1
size of 1572864 B
```

Callstack: `ttnn::prim::embedding → EmbeddingsDeviceOperation →
tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region`.

The Longformer sliding-window attention (with input padded to 512 by the
model's `_pad_to_window_size`) generates an embedding lookup into a table
with row width `12 heads × 256 seq_chunks × 513 window_span = 1,575,936`
elements × 2 bytes = 3,151,872 bytes. After alignment: 3,263,488 bytes —
exactly 2× the Wormhole L1 limit of 1,572,864 bytes.

`EmbeddingsRMProgramFactory` allocates one CB of that full row width,
which exceeds L1 even at the minimum buffering factor of 1.

## Fix
**Bug 1 fix:**
In `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`,
inside `StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite`, after
computing `maxIndex`, added:
```cpp
if (maxIndex == 0) {
  return rewriter.notifyMatchFailure(
      srcOp, "maxIndex is 0 (single-row operand in indexed dim); "
             "use StableHLOGatherToEmbeddingPattern instead");
}
```
Branch: `remediation/openmed_pii_clinical_longformer_base_149m_v1_token_classification_pytorch_single_device_inference` in tt-mlir.

**Bug 2 proposed fix (not implemented):**
Would live in `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp`
and its reader/writer kernels. Requires implementing horizontal sub-chunking:
split the embedding row into blocks of at most `L1_size/2` bytes, process
each block independently, and concatenate in DRAM. This is new device kernel
infrastructure.

## Tier B justification
**Indicator:** `new-infrastructure`

The minimum kernel work unit for the RM embedding operation is one full row.
At 3.15 MB per row and L1 = 1.5 MB, no existing blocking/slicing parameter
can make this fit. The fix requires a streaming gather kernel that reads and
writes in sub-row chunks — this does not exist anywhere in the three
tt-metal embedding program factories.

## Verification
- pytest exit: FAIL
- Hardware:    n150 (wormhole)
- Duration:    239.07s (0:03:59)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — guard maxIndex==0 in StableHLOGatherToSliceRepeatConcatPattern

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | bd01777f2cc2317bcb10cd898c6f3a621662741b |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
