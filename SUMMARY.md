# Remediation Summary: jetbrains_research_qwen3_8b_am-causal_lm-pytorch-8B_am-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[jetbrains_research_qwen3_8b_am/causal_lm/pytorch-8B-am-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; original timeout was a transient Fabric Router Sync condition

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
2026-04-22 15:34:27.749 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 0: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

## Root cause
The original failure was a transient Fabric Router Sync timeout on the ethernet core, not a deterministic compiler or loader bug. On re-run (2026-04-28) with the same branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-35`, `TT_METAL_OPERATION_TIMEOUT_SECONDS=30`, and no code changes, the test completed successfully in 139.32s with correct PCC.

## Fix
No fix required.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    139.32s (0:02:19)
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
