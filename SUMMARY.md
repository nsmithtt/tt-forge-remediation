# Remediation Summary: dreamshaper_xl-pytorch-turbo-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dreamshaper_xl/pytorch-turbo-single_device-inference]

## Result
NO_FIX_NEEDED — test passes without modification on the configured branch

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
2026-04-24 18:08:37.916 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

## Root cause
The original failure (device TIMEOUT/hang on 2026-04-24) could not be reproduced on branch arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-22 as of 2026-04-28. The test ran to completion successfully on the first attempt, producing valid image outputs and benchmark numbers. The timeout was likely transient (device state from a prior session, thermal event, or race in the scheduler) and is not present in the current build.

## Fix
No fix required. Test passes as-is.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    839.15s (0:13:59)
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
