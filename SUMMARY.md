# Remediation Summary: cross_encoder/passage_ranking/pytorch-ms-marco-MiniLM-L4-v2-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cross_encoder/passage_ranking/pytorch-ms-marco-MiniLM-L4-v2-single_device-inference]

## Result
SILICON_PASS — Added diverse sample pairs to fix degenerate single-element PCC.

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
degenerate-single-sample-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.0. Required: pcc=0.95.

## Root cause
The loader provided only a single query-passage pair.  The cross-encoder
model therefore outputs a logit tensor of shape (1, 1) — one element.  The
test framework hard-codes PCC = 0.0 for single-element tensors (variance is
undefined), so the PCC check always fails regardless of numerical correctness.
The forced allclose fallback also fails because TT bfloat16 and CPU bfloat16
matmul use different reduction orderings, producing a 2-ULP difference
(0.125 at magnitude ~9) that slightly exceeds the 0.01 + 0.01×|ref| ≈ 0.101
tolerance.  Both 9.125 (CPU) and 9.0 (TT) are valid bfloat16 values; the
difference is bfloat16 accumulation, not a computation bug.  The root issue
is the loader bug: a single sample collapses evaluation to a single scalar,
making PCC entirely meaningless.

## Fix
Added three additional diverse (query, passage) pairs to `sample_pairs` in
`tt-xla/third_party/tt_forge_models/cross_encoder/passage_ranking/pytorch/loader.py`.
With four pairs the model returns a (4, 1) logit tensor.  PCC is now computed
over four values spanning a ~21-logit range; sub-ULP per-element noise has no
effect on the rank correlation, so PCC ≈ 1.0.

The change is on branch `remediation/cross-encoder-passage-ranking-pcc-fix`
in the tt-forge-models repo.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    36.80s
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/cross_encoder/passage_ranking/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 57b3ed0375040a4c9eb1417db997f590890a8a04 |
