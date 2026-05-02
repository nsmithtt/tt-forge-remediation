# Remediation Summary: ibert-masked_lm-pytorch-Ibert_Roberta_Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ibert/masked_lm/pytorch-Ibert_Roberta_Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
ibert-dtype-override-bfloat16-nan, gather-single-row-concat-shape-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

The actual runtime error (with TTXLA_LOGGER_LEVEL=DEBUG) was:
  PJRT Error code: 13 (INTERNAL): 'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 9 vs. 17

After fixing the compiler bug, a second failure appeared:
  PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99

## Root cause

Two independent bugs:

**Bug 1 — tt-mlir (Tier A):** `StableHLOGatherToSliceRepeatConcatPattern`
(benefit=2) handles gather-to-pad patterns, but its starts/ends counter
double-counts indices that are simultaneously 0 AND maxIndex. For
token_type_embeddings with shape [1, 768], maxIndex == 0, so every index
is both 0 and maxIndex. The pattern emits a ttir.concat whose output
dimension is 17 instead of 9, failing the verifier.

**Bug 2 — loader:** `TorchDynamicLoader.load_model()` introspects the
loader's `load_model` signature and, when `dtype_override` is present,
calls `loader.load_model(dtype_override=torch.bfloat16)`. The IBERT loader
accepted this parameter and passed `torch_dtype=torch.bfloat16` to
`AutoModelForMaskedLM.from_pretrained()`. IBERT in bfloat16 overflows in
layer norm and GELU on CPU, producing NaN logits in the golden reference
output. PCC of NaN vs. any value is NaN, causing the comparison to fail.

## Fix

**tt-mlir fix** (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`):
Added an early-return guard in `StableHLOGatherToSliceRepeatConcatPattern::matchAndRewrite`:
when `maxIndex == 0`, return `notifyMatchFailure` so the lower-benefit
`StableHLOGatherToEmbeddingPattern` handles single-row embedding tables
correctly.

**Loader fix** (`ibert/masked_lm/pytorch/loader.py` in tt_forge_models):
Removed `dtype_override` parameter from `load_model` signature. The model
now loads in its native float32, producing valid CPU golden output.
TT hardware runs in bfloat16 natively and achieves PCC ≥ 0.99.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    39.60s
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- `tt-xla/third_party/tt_forge_models/ibert/masked_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 47cb8ca616ab0f7df042e8f7c7432501bc40e219 |
| tt-xla          | 426b037ca8891cd67a0537575e1e106c68166d75 |
| tt-forge-models | 24b45e6be2f04ffc10f004688a72218d6d0b438a |
