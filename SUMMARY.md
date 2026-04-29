# Remediation Summary: f5_tts_russian-pytorch-v1_base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[f5_tts_russian/pytorch-v1_base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
missing-requirements-txt-torchaudio-cpu-wheel

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ModuleNotFoundError: No module named 'f5_tts'

(The reported failure string "sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute" is the last printed line of the pytest summary, but the actual test exception was the missing f5_tts module.)

## Root cause
Two loader bugs:

1. `f5_tts_russian/pytorch/` had no `requirements.txt`, so the `f5-tts` package was never installed before `load_model()` tried to `from f5_tts.model.backbones.dit import DiT`.

2. Once `requirements.txt` was added with `f5-tts>=1.1.5`, pip resolved `torchaudio` to version 2.11.0 (the latest satisfying `>=2.0.0`). The standard manylinux torchaudio wheel links against `libtorch_cuda.so`, which is absent because the venv contains a CPU-only PyTorch build (`torch==2.9.1+cpu`). The fix is to pin `torchaudio==2.9.1` and source it from `https://download.pytorch.org/whl/cpu` — the same pattern used by `seamless_m4t_v2`.

## Fix
Added `f5_tts_russian/pytorch/requirements.txt` in `tt-forge-models` on branch `remediation/f5_tts_russian-pytorch-v1_base-single_device-inference`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torchaudio==2.9.1
f5-tts>=1.1.5
```

This ensures torchaudio is installed from the CPU-only wheel that does not require `libtorch_cuda.so`, matching the existing torch 2.9.1+cpu environment.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    155.70s (0:02:35)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: f5_tts_russian/pytorch/requirements.txt` (new file)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 13b0b0c48078a37a41a2538975304f386be1ab81 |
