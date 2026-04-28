# Remediation Summary: dinov2/feature_extraction/pytorch-Curia-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dinov2/feature_extraction/pytorch-Curia-single_device-inference]

## Result
FAIL — loader fix applied (use_fast=False for Curia processor) but silicon verification blocked: raidium/curia is a gated HuggingFace repository and the available credentials (aleks-knezevic account) return 403 Forbidden

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-use-fast-default

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `BitImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
`dinov2/feature_extraction/pytorch/loader.py` calls `AutoImageProcessor.from_pretrained` for the Curia variant (`raidium/curia`) without `use_fast=False`. The raidium/curia checkpoint stores a `BitImageProcessor` config. In transformers 5.x the fast variant is now selected by default, producing different preprocessing behavior. This is the same transformers 5.x breaking change that affected the XRay_Base variant (BlipImageProcessor) and was fixed in a prior report.

## Fix
Added `use_fast=False` to the processor kwargs for the `CURIA` variant in `dinov2/feature_extraction/pytorch/loader.py`. The fix is in the tt-forge-models repo at commit `e9c32c80381dd995ec888e424fdced6acae69cf8` on branch `remediation/dinov2-feature_extraction-pytorch-Curia-single_device-inference`.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

Note: The test could not be run on silicon. In the remediation environment, `raidium/curia` returned 403 Forbidden (gated repository, not accessible to the aleks-knezevic account). The original `BitImageProcessor` error could not be reproduced locally. The fix is correct by analogy with the XRay_Base/BlipImageProcessor fix (same transformers 5.x breaking change, same loader pattern, same fix).

## Files changed
- `tt_forge_models/dinov2/feature_extraction/pytorch/loader.py` — `use_fast=False` for Curia variant

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | db766b77b0c652a456da92038252132a9adb7ae9 |
| tt-forge-models | e9c32c80381dd995ec888e424fdced6acae69cf8 |
