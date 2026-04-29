# Remediation Summary: c2s_scale_gemma_2_2b-causal_lm-pytorch-C2S_Scale_Gemma_2_2B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[c2s_scale_gemma_2_2b/causal_lm/pytorch-C2S_Scale_Gemma_2_2B-single_device-inference]

## Result
SILICON_PASS — aten.slice OOB negative start clamped in TorchFunctionOverride

## Stack layer
tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-256, 255], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

Originating from transformers/cache_utils.py line 214:
  self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Gemma 2's `SlidingWindowCache` uses `sliding_window=4096`. When updating the KV cache it slices with `start = -sliding_window + 1 = -4095` on a sequence-length-256 tensor (valid range `[-256, 255]`). PyTorch eager silently clamps out-of-range negative indices, but the XLA lazy backend (`torch/csrc/lazy/core/helpers.cpp`) raises `RuntimeError: Value out of range` instead. The fix belongs in `tt-xla` in `TorchFunctionOverride.__torch_function__`, which is the correct interception point before the XLA lazy backend validates the index.

## Fix
Added a pre-clamp guard in `python_package/tt_torch/torch_overrides.py` (`TorchFunctionOverride.__torch_function__`) in tt-xla. When `func is torch.ops.aten.slice.Tensor`, the handler reads the tensor's static dimension size and clamps `start`/`end` to `[-size, size]` before dispatching, matching PyTorch eager semantics. No other files changed.

Remediation branch: `remediation/c2s_scale_gemma_2_2b-causal_lm-pytorch-C2S_Scale_Gemma_2_2B-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    141.15s (0:02:21)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 69bfe7036f813e476723e5c57e4913ab30a6c07d |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
