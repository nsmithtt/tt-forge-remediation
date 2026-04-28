# Remediation Summary: magicoder_s_ds_6_7b-causal_lm-pytorch-Magicoder-S-DS-6.7B-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[magicoder_s_ds_6_7b/causal_lm/pytorch-Magicoder-S-DS-6.7B-single_device-inference]

## Result
NO_FIX_NEEDED — test passed on the configured branch; original failure could not be reproduced

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 14:29:06.993 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

## Root cause
The original failure (Fabric Router Sync Timeout on Device 2) could not be reproduced on the configured branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-29` running on p150b hardware. The test ran to completion and PASSED with PCC within tolerance in 198.80 seconds. The timeout in the original report may have been a transient hardware or runtime condition.

## Fix
No fix required. Test already passes on the configured branch.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    198.80s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c9b45c4dfe71bf9beed21e9db576f2728db20aeb |
