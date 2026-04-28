# Remediation Summary: 8542-55404-tests-collected-garbled-test-name

## Skill version
6

## Test
8542/55404 tests collected (46862 deselected) in 37.00s

## Result
NO_FIX_NEEDED — test_name is garbled pytest collection-summary output, not a valid pytest node ID; cannot be reproduced

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
```
/bin/bash: -c: line 1: `source venv/activate && python -m pytest 8542/55404 tests collected (46862 deselected) in 37.00s -svv'
```

## Root cause
This is not a real test failure. The `test_name` field in `results_main.yaml` for branch `ip-172-31-23-5-tt-xla-dev/ubuntu/2026-04-23_16-01/hf-bringup-12` was recorded as `8542/55404 tests collected (46862 deselected) in 37.00s`, which is pytest's collection-phase summary line (printed when pytest finishes collecting tests), not a test node ID.

When `silicon_validate.py` ran `run_test()` with this string as `test_name`, it constructed:
```
source venv/activate && python -m pytest 8542/55404 tests collected (46862 deselected) in 37.00s -svv
```
and passed it to `bash -c`. Bash rejected the command due to unbalanced quoting introduced by the literal string, producing the failure logged as `failure_detail`.

The root cause is in the CI data-capture layer: whatever wrote this entry to `results_main.yaml` stored a pytest collection-status line as the test name instead of a test node ID. This has no counterpart in the compiler stack.

The same `8542/55404` pattern appears in three other entries (two `MAX_FAILED_ATTEMPTS`, one `SILICON_FAIL`) across different CI runs on the same host pair, confirming it is a recurring collection-output capture bug.

## Fix
No fix required. There is no model, compiler-stack path, or test configuration to change. The `failure_category: unknown` entry should either be filtered out by the CI results collector or the collector should be fixed to not store collection-summary lines as test names.

## Verification
- pytest exit: FAIL (bash syntax error — test name is not a valid node ID)
- Hardware:    not-run
- Duration:    N/A
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
