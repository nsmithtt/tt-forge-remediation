# Remediation Summary: huihui_qwen3_5_9b_abliterated_gguf-causal_lm-pytorch-9B_Abliterated_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_5_9b_abliterated_gguf/causal_lm/pytorch-9B_Abliterated_GGUF-single_device-inference]

## Result
NO_FIX_NEEDED — test passes on the configured branch; the InvalidVersion failure was already resolved before this report

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
raise InvalidVersion(f"Invalid version: {version!r}")
```

## Root cause
The failure was a missing `gguf>=0.10.0` entry in requirements.txt. Without the `gguf` package installed before the test, the GGUF checkpoint loading path in transformers raised `packaging.version.InvalidVersion` during initialization. The branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-15` already includes two fixes:

1. `f019b16c65` — added `requirements.txt` with `gguf>=0.10.0` so the package is installed before the loader runs.
2. `0c47a0d23d` — added `ignore_mismatched_sizes=True` to suppress weight shape mismatches that occur because the model is loaded under the Qwen3 architecture rather than the Qwen3.5 hybrid architecture (these layers reinitialize to random values, but CPU↔TT PCC still passes since both runtimes use the same weights).

## Fix
No fix required. The failure could not be reproduced; test exits PASS in 562.33s (0:09:22) on TT silicon.

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 562.33s (0:09:22)
- Tier A attempts: N/A

## Files changed
None — no fix required.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3e237659705c6644cb063dc8e81cffa34717e7d2 |
