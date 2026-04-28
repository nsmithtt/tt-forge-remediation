# Remediation Summary: cross_encoder-qnli-pytorch-qnli-distilroberta-base-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[cross_encoder/qnli/pytorch-qnli-distilroberta-base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
degenerate-single-element-pcc-single-sample-pair

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
The loader's `sample_pairs` contained only one query-passage pair, producing a model output of shape `(1, 1)` — a single-element tensor. The test framework's evaluator always returns PCC=0.0 for single-element tensors that fail the allclose short-circuit (the evaluator has no meaningful variance to correlate against). The allclose check was also failing because the bfloat16 accumulation difference between CPU and TT hardware (measured atol=0.0625 ≈ 2 ULPs at magnitude ~5) exceeded the default allclose tolerance of 0.01. The root cause was therefore in the loader: a single sample pair makes numerical verification impossible regardless of whether the compiler is producing correct results.

This is identical in structure to the `cross_encoder/passage_ranking/pytorch-ms-marco-MiniLM-L4-v2` fix (remediation/cross-encoder-passage-ranking-pcc-fix).

## Fix
Added three more diverse query-passage pairs to `cross_encoder/qnli/pytorch/loader.py` in `tt_forge_models`. This makes the model output a `(4, 1)` tensor instead of `(1, 1)`. With diverse relevance scores spanning many logit units, sub-ULP per-element bfloat16 noise produces PCC ≈ 1.0, allowing the test to pass.

File changed: `cross_encoder/qnli/pytorch/loader.py`
Branch: `remediation/cross-encoder-qnli-pcc-fix` in `tt-forge-models`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    46.85s
- Tier A attempts: N/A

## Files changed
- `cross_encoder/qnli/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d1c9b7b1a570b417c6f43cd94276200fd8e94766 |
| tt-forge-models | ff37def6d2dabf3bf28f3825bdaca73062ac121a |
