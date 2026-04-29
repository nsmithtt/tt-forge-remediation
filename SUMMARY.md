# Remediation Summary: cocodr-embedding_generation-pytorch-OpenMatch-cocodr-base-msmarco-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[cocodr/embedding_generation/pytorch-OpenMatch/cocodr-base-msmarco-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on silicon with branch ip-172-31-30-232-tt-xla-dev/ubuntu/hf-bringup-range-1500-500-3

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
No failure reproduced. The test passed on silicon with exit code PASS in 35.59s.

## Root cause
The test was run against tt-xla branch ip-172-31-30-232-tt-xla-dev/ubuntu/hf-bringup-range-1500-500-3 and passed without any modifications. The previously reported failure could not be reproduced on this build.

## Fix
No fix required. The test passes as-is on the configured branch.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    35.59s
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
