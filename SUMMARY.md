# Remediation Summary: longformer_masked_lm_pytorch_large_4096_single_device_inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longformer/masked_lm/pytorch-Large_4096-single_device-inference]

## Result
FAIL — after the Tier A gather fix, blocked by Tier B embedding CB L1 overflow (same class as Longformer_Zh report)

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
Original failure (before fix):
```
loc("gather.50"): error: 'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 512 vs. 1023.
module_builder.cc:889 ERR| Failed to convert from SHLO to TTIR module
E   ValueError: Error code: 13
```

Residual failure after Tier A fix (new blocker):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=1)] grow to 4314112 B which is beyond max L1 size of 1572864 B
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

**Bug 1 (Tier A — fixed):** `StableHLOGatherToSliceRepeatConcatPattern` in tt-mlir
(`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`) mishandles gathers
where `sliceSizes[indexedDim] == inputShape[indexedDim]`, making `maxIndex = 0`.

In that case every index simultaneously satisfies `(index == 0)` AND
`(index == maxIndex)`, so the starts/ends counting loop double-counts all N
indices: `starts = N`, `ends = N`, after decrement `starts = N-1`, `ends = N-1`.
The resulting concat contains `(N-1) + 1 + (N-1) = 2N-1` elements instead of the
N expected by the gather output type.

Longformer Large's sliding-window attention (`_sliding_chunks_matmul_qk`) produces
a gather with `inputShape[0] = sliceSizes[0] = 512` (gathering the entire dimension
at each index), triggering the 2N-1 bug: output 512 vs. 1023.

Fix: guard `if (maxIndex == 0) return notifyMatchFailure` so the lower-priority
`StableHLOGatherToEmbeddingPattern` handles the op instead. This fix is
cherry-picked from commit `0c1bf08ec` (authored for the jina-clip-v2 GLuCoSE
model) into `remediation/longformer_masked_lm_pytorch_large_4096_single_device_inference`
in tt-mlir.

**Bug 2 (Tier B — not fixed):** After the gather fix, compilation succeeds but
execution fails with:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=1)]
          grow to 4314112 B which is beyond max L1 size of 1572864 B
```
in `ttnn::prim::embedding` (callstack: `tt::runtime::ttnn::operations::embedding::run
→ ttnn::embedding → ttnn::prim::embedding → EmbeddingsDeviceOperation`).

The Longformer Large sliding-window attention generates an embedding lookup into a
table whose row width is 4,314,112 bytes (≈ 4.1 MB). The `EmbeddingsRMProgramFactory`
allocates a CB of one full row, which is 2.7× the Wormhole L1 limit of 1,572,864 bytes.
This is the same root cause documented in the `longformer_zh-feature_extraction-
pytorch-Longformer_Zh-single_device-inference` report (fingerprint
`embedding-rm-weight-row-exceeds-l1`), where the row was 3.15 MB.
For Longformer Large the row is larger (4.1 MB) because `hidden_size=1024` (vs 768).

## Fix

**Bug 1 (applied):** Cherry-picked commit `0c1bf08ec` to
`remediation/longformer_masked_lm_pytorch_large_4096_single_device_inference`
in tt-mlir. Change: 12 lines in
`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` —
add `if (maxIndex == 0) return notifyMatchFailure(...)` guard after computing
`maxIndex`.

**Bug 2 (proposed, not applied):** The fix would live in
`tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp`
and requires sub-row streaming: split each embedding row into blocks of at most
`L1_size / 2` bytes, process blocks independently, and write each block to the
corresponding slice of the output tensor in DRAM. This is the same new-infrastructure
fix described in the Longformer_Zh SUMMARY.md.

## Tier B justification
**Indicator:** `new-infrastructure`

The `EmbeddingsRMProgramFactory` allocates a CB of one full embedding row.
With row width 4,314,112 bytes and L1 limit 1,572,864 bytes, no existing
blocking parameter reduces this below L1. The fix requires a sub-row streaming
kernel that splits the row into L1-sized chunks — new device kernel infrastructure
not present in tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole (n150)
- Duration:    354.97s (0:05:54)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
  (cherry-picked `0c1bf08ec`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 77020630b4 |
| tt-xla          | 94362e631 |
| tt-forge-models | 0f7b734348 |
