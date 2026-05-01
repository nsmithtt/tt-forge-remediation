# Remediation Summary: lotus_normal_g_v1_1-pytorch-Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lotus_normal_g_v1_1/pytorch-Base-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; original failure was a transient fabric router sync timeout

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
2026-04-19 03:26:30.091 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

## Root cause
The error is a transient device initialization failure in the Ethernet fabric router sync, not a model-specific bug. The fabric router sync timeout (`TT_THROW: Fabric Router Sync: Timeout after 10000 ms`) is a known transient that occurs during device bring-up on multi-device systems; it is not reproducible on retry. The CI framework already excludes this error class from the `tt_fatal` category. Running the test fresh on the `worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-start65-0` branch produced `1 passed in 247.39s` with no errors.

## Fix
No fix required. The test passes without modification on the configured branch.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    247.39s (0:04:07)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
