# Remediation Summary: hubert-feature_extraction-pytorch-Base_ls960-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hubert/feature_extraction/pytorch-Base_ls960-single_device-inference]

## Result
FAIL — dtype loader fix applied; residual PCC=0.981 < 0.99 is ttmlir-bf16-matmul-precision-floor (Tier B)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-matmul-precision-floor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: expected scalar type Float but found BFloat16

(Reported as "RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13" in CI; reproduced locally as the dtype mismatch.)

## Root cause
Two issues:

1. **Loader bug (fixed):** `load_inputs()` in `hubert/feature_extraction/pytorch/loader.py` did not cast processor output tensors to `dtype_override`. The `Wav2Vec2FeatureExtractor` always returns `input_values` as float32, while `load_model()` loads weights in bfloat16. The first `Conv1d` in the feature extractor received float32 input with bfloat16 weights and raised `RuntimeError: expected scalar type Float but found BFloat16`.

2. **Residual compiler bug (Tier B):** After the loader fix, the test compiles and runs on hardware but yields PCC=0.981 vs required 0.99. The CPU BF16 floor for this model is 0.9996 (measured: CPU FP32 vs CPU BF16), so the shortfall is not due to BF16 arithmetic but to reduced precision in the WH BF16 matmul lowering path. The mHuBERT_147 variant (identical architecture, different weights) passes with PCC > 0.99 on the same hardware build, confirming the issue is weight-value dependent — Base_ls960 activations compound the WH BF16 matmul rounding error across 12 transformer layers until it falls below threshold.

## Fix
**Loader fix applied** in `tt_forge_models/hubert/feature_extraction/pytorch/loader.py`: after calling `self._processor(...)` in `load_inputs()`, cast all floating-point tensors in the returned dict to `dtype_override` when it is set. Committed to branch `remediation/hubert-feature_extraction-pytorch-Base_ls960-single_device-inference` in tt-forge-models.

**Residual:** No fix attempted for the Tier B PCC issue.

## Tier B justification
Indicator: **cross-cutting**. The WH BF16 matmul precision floor is an accumulation error inherent to the hardware's BF16 ALUs and affects every matmul lowering. Fixing it would require cross-cutting changes (e.g. F32 intermediate accumulation for all matmuls) touching many files across tt-mlir and tt-metal. This is the same root cause as the known `ttmlir-bf16-matmul-precision-floor` failures seen in Gemma 7B, DocTR PARSeq ViT, and others.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    84.11s (after loader fix)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/hubert/feature_extraction/pytorch/loader.py` — cast inputs to dtype_override in load_inputs

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 2ed0225f652c9eb9a2c18064ab220a27ca142dfe |
