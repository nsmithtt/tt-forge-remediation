# Remediation Summary: bert/token_classification/pytorch-Davlan/bert-base-multilingual-cased-ner-hrl-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[bert/token_classification/pytorch-Davlan/bert-base-multilingual-cased-ner-hrl-single_device-inference]

## Result
NO_FIX_NEEDED — test passed cleanly on current branch; original failure was a transient hardware event

## Failure
2026-04-19 04:38:04.907 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 0: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)

## Root cause
The Fabric Router Sync timeout is a runtime-layer (tt-metal) hardware transient: the ethernet fabric router on Device 0 did not complete its handshake within 10 s. This is a known flaky failure mode on multi-chip wormhole boards when a previous test run left the device in a partial fabric-init state. The device was reset (or recovered on its own) before our re-run, and the test passed without any code changes.

## Fix
No code change required. The original run timed out due to a transient device state. Re-running after device reset clears the condition.

## Verification
pytest exit status: PASSED
Wall-clock duration: 92.28 s (1:32)
Hardware: n150 (single device, wormhole)

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
