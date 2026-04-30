# Remediation Summary: genre_recognizer-audio_classification-pytorch-Finetuned_GTZAN-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[genre_recognizer/audio_classification/pytorch-Finetuned GTZAN-single_device-inference]

## Result
NO_FIX_NEEDED — test passes when the pytest node ID is properly shell-quoted; failure was a CI invocation argument-splitting issue

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
ERROR: file or directory not found: GTZAN-single_device-inference]

## Root cause
The pytest node ID `tests/runner/test_models.py::test_all_models_torch[genre_recognizer/audio_classification/pytorch-Finetuned GTZAN-single_device-inference]` contains a space in the variant name (`Finetuned GTZAN`). When the CI system invoked pytest without properly quoting this argument, the shell split it at the space into two separate arguments. pytest received `GTZAN-single_device-inference]` as a positional path argument, could not find it as a file or directory, and aborted with the listed error. This is a shell invocation issue, not a model-code or compiler-stack bug. When run with proper quoting, the test passes on TT silicon in ~83 s.

## Fix
None required. The genre_recognizer model loader (`tt_forge_models/genre_recognizer/audio_classification/pytorch/loader.py`) is correct. The test passes when the node ID is quoted: `python -m pytest -svv "tests/runner/test_models.py::test_all_models_torch[genre_recognizer/audio_classification/pytorch-Finetuned GTZAN-single_device-inference]"`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    82.79s
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | f29c1ea0108a231f7b2a40ae0f065eadd6a901f5 |
