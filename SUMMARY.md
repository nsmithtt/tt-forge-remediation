# Remediation Summary: glotmax-causal_lm-pytorch-GlotMAX-101-14B-LST-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glotmax/causal_lm/pytorch-GlotMAX-101-14B-LST-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; original TIMEOUT could not be reproduced

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
2026-04-24 18:16:11.247 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

## Root cause
Could not reproduce. The original failure was recorded at 2026-04-24 18:16:11 UTC on branch
arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-42. Running the same test today
(2026-04-28) against the branch HEAD (840b2df627) with TT_METAL_OPERATION_TIMEOUT_SECONDS=30
on a p150b device, the test completed PASS in 192.54s with two clean inference iterations
(avg 0.12s/iter).

The original timeout is consistent with a transient device hang — hardware can enter a
bad state between tests (e.g. from a preceding test on the same CI worker leaving the
device unresponsive), causing the first inference to exceed the operation timeout. A fresh
run on an idle device passes cleanly.

No GlotMAX-specific fixes exist in any commit on the branch between the failure date
(2026-04-24 18:16 UTC) and the current HEAD (2026-04-25 16:47 UTC). The model (Qwen3
architecture, ~14B params, ~28 GB in bfloat16) runs on the p150b with the existing
compiler stack.

## Fix
None required. Test already passes.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    192.54s (0:03:12)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 840b2df627db5ea656a8714d48e3a305e1ae1351 |
