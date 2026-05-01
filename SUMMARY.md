# Remediation Summary: jina_reranker_v2-passage_ranking-pytorch-base-multilingual-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jina_reranker_v2/passage_ranking/pytorch-base-multilingual-single_device-inference]

## Result
SILICON_PASS — added maxIndex==0 guard in StableHLOGatherToSliceRepeatConcatPattern; test passes with PCC ≥ threshold

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
gather-slice-repeat-concat-maxindex-zero

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: Error code: 13
(PJRT INTERNAL error from TT-MLIR compiler during _xla_warm_up_cache)

Compiler log detail:
```
loc("gather.418"): error: 'ttir.concat' op Output tensor dimension 0 does not match
the sum of input tensor dimensions: 35 vs. 69.
module_builder.cc:889 ERR| Failed to convert from SHLO to TTIR module
```

## Root cause
In `StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite`
(`tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`), the
pattern lowers `stablehlo.gather` ops with constant, consecutively-ordered
indices into slice + repeat + concat.

The bug manifests when `maxIndex = inputShape[indexedDim] - sliceSizes[indexedDim] == 0`,
i.e. when the lookup table has exactly one position in the indexed dimension
(e.g. the token-type embedding table `tensor<1x768xbf16>`).  In that case every
index simultaneously satisfies `(index == 0)` AND `(index == maxIndex)`, so the
`starts` and `ends` counters both reach N (the number of indices).  After
decrement: `starts = N-1 = 34`, `ends = N-1 = 34`.  The pattern then builds a
concat with `34 + 1 + 34 = 69` elements (2N-1) in the indexed dimension, while
the declared output type is `tensor<35x768xbf16>` (N=35).  MLIR's verifier
catches the mismatch and the PJRT plugin returns Error code 13 (INTERNAL).

The surface point is `jina-reranker-v2-base-multilingual`'s XLM-RoBERTa model,
which has a `token_type_embeddings` table with shape `[1, 768]` (only one type).

## Fix
Added a guard before the `starts`/`ends` counting loop in
`StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite`:

```cpp
if (maxIndex == 0) {
  return rewriter.notifyMatchFailure(
      srcOp, "maxIndex is 0 (single-row operand in indexed dim); "
             "use StableHLOGatherToEmbeddingPattern instead");
}
```

This causes the op to fall through to `StableHLOGatherToEmbeddingPattern`
(benefit=1), which correctly handles singleton indexed dimensions.

Commit: `0c1bf08ec StableHLOToTTIR: guard GatherToSliceRepeatConcat against maxIndex==0`
Branch: `remediation/jina_reranker_v2-passage_ranking-pytorch-base-multilingual-single_device-inference`
in `tenstorrent/tt-mlir`.

File changed:
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (+12 lines)

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    56.26s
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0c1bf08ec0b10242aab5bdda82d9e96c268b2b83 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
