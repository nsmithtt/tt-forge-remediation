# Remediation Summary: jamba-causal_lm-pytorch-AI21_Jamba_Reasoning_3B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[jamba/causal_lm/pytorch-AI21_Jamba_Reasoning_3B-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on current branch; reported failure message is a harmless SWIG deprecation warning emitted after test completion, not a test failure

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
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

## Root cause
The reported "failure message" is not a test failure. It is a DeprecationWarning emitted by the SWIG C++ extension teardown code (`swigvarlink` is an internal SWIG type) that appears as the very last line of pytest output, after the `1 passed` summary line. The warning originates from `<frozen importlib._bootstrap>:488` during Python interpreter shutdown. On the current branch (`ip-172-31-22-17-tt-xla-dev/ubuntu/hf-bringup-8`), the test passes end-to-end in 622s with no compilation or PCC errors.

## Fix
No fix needed. The test was already passing.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    622.03s (0:10:22)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 0b720724500733f2b6d0eea4788799afd585a9c2 |
| tt-forge-models | eb4b6b292744eb220657107ae7ef821ef140ba6d |
