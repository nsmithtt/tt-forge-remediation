# Remediation Summary: gemma3_270m_qat_gguf-causal_lm-pytorch-Q4_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_270m_qat_gguf/causal_lm/pytorch-Q4_0-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-missing-requirements-txt, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Two bugs:

1. **Loader (loader layer)**: `gemma3_270m_qat_gguf` had no `requirements.txt`, so `gguf>=0.10.0` was not installed when this model's test ran in sessions where gguf was not in the global venv. `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` raises `ImportError` when `is_gguf_available()` returns False. Fix: add `requirements.txt` with `gguf>=0.10.0`.

2. **Compiler frontend (tt-xla)**: Gemma3's `SlidingWindowCache.update()` does `full_value_states[:, :, -sliding_window+1:, :]` where `sliding_window=512`. With seq_len=12, `start=-511` is outside `[-12, 11]`. PyTorch allows such out-of-range starts (semantically clamped to 0), but the XLA/TT backend raises `RuntimeError: Value out of range`. Fix: add `clamp_out_of_range_slice_starts()` FX pass in `passes.py` that normalizes negative slice starts to `max(-dim_size, start)`.

The branch `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-18` also had a chat_template fix (commit `a9bfb53a4e`) for the tokenizer not having a chat template; this fix was already present on the branch.

## Fix
**Loader fix** (`tt-forge-models`, branch `remediation/gemma3_270m_qat_gguf-causal_lm-pytorch-Q4_0-single_device-inference`):
- Added `gemma3_270m_qat_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`

**Compiler frontend fix** (`tt-xla`, branch `remediation/gemma3_270m_qat_gguf-causal_lm-pytorch-Q4_0-single_device-inference`):
- `python_package/tt_torch/backend/passes.py`: Added `clamp_out_of_range_slice_starts()` FX pass
- `python_package/tt_torch/backend/backend.py`: Import and call the new pass after `bypass_assert_tensor_metadata`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `gemma3_270m_qat_gguf/causal_lm/pytorch-Q4_0-single_device-inference` as `EXPECTED_PASSING`

## Verification
- pytest exit: PASS
- Hardware: n150
- Duration: 298.80s (0:04:58)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py`
- `tt-xla/python_package/tt_torch/backend/backend.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-forge-models/gemma3_270m_qat_gguf/causal_lm/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 252446167b0d04ea187b13ab3e9a7d962dbf2461 |
| tt-forge-models | 51a44bb154cb1feaa3a9e3d79caa44f9f9c0e3ae |
