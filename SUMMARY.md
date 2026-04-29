# Remediation Summary: deberta/sentiment_analysis/pytorch-v3_ft_financial_news_sentiment-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta/sentiment_analysis/pytorch-v3_ft_financial_news_sentiment-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
shared-lhs-matmul-fusion-output-rank-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
(exit code 134 / SIGABRT)

## Root cause
`SharedLHSMatmulFusion<LinearOp>` in `TTIRFusing.cpp` fuses multiple `LinearOp`s that share the same LHS operand. `collectCandidates` checked that candidate RHS ranks matched the root op's RHS rank, but did NOT check that candidate output ranks matched the root op's output rank.

DeBERTa V2's attention mechanism applies `query_proj`, `key_proj`, and `value_proj` to the same `hidden_states` tensor, producing three `LinearOp`s with shared LHS. The `MatmulWithBiasFusionPattern` fuses each `matmul + bias_add` pair into a `LinearOp` using `broadcastShape` for the output type. If bias shapes differ, some resulting `LinearOp`s can have higher output rank than others (e.g. one op has rank 3 output `[1, 128, 768]` while another has rank 2 `[128, 768]`).

`matchAndRewrite` computes `outputFusedDim = rootOutputType.getRank() - 1` using the root op's rank, then passes it to `replaceWithSlices`. `replaceWithSlices` calls `shape[outputFusedDim]` on each candidate's output type. When a candidate has lower rank than the root op, this index is out of bounds, tripping the `llvm::ArrayRef` debug assertion (SIGABRT).

## Fix
Added an output rank consistency check inside `collectCandidates` in `TTIRFusing.cpp`. Candidates whose output rank does not match the root op's output rank are now skipped, preventing the OOB access in `replaceWithSlices`.

File: `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
Branch: `remediation/deberta_sentiment_analysis-pytorch-v3_ft_financial_news_sentiment-single_device-inference` in `tt-mlir`
Commit: `86ea659285cba92aefd38f16d7a8d2cb3d20d45c`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    59.11s
- Tier A attempts: 1

## Files changed
- tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 86ea659285cba92aefd38f16d7a8d2cb3d20d45c |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
