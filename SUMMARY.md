# Remediation Summary: neural_monarch-pytorch-7B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[neural_monarch/pytorch-7B-single_device-inference]

## Result
SILICON_PASS — clamp aten.slice OOB start in TorchFunctionOverride

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
RuntimeError: Value out of range (expected to be in range of [-32, 31], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

The error originates from transformers/cache_utils.py line 214:
  self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
where sliding_window=4096, so start = -4095.

## Root cause
NeuralMonarch-7B is a Mistral-architecture model with sliding-window attention (window_size=4096). During KV cache updates, `transformers` slices the cached key/value states with `start = -(sliding_window - 1) = -4095` to retain only the last `sliding_window` tokens. With a short sequence (32 tokens), the XLA lazy backend rejects the slice because `-4095 < -32` (the actual dimension size), raising "Value out of range (expected to be in range of [-32, 31], but got -4095)". PyTorch eager silently clamps out-of-range start indices to `-size`, but XLA validates bounds strictly.

The fix belongs in `tt-xla`'s `TorchFunctionOverride.__torch_function__` where `aten.slice.Tensor` calls can be intercepted before reaching XLA.

## Fix
Added a guard in `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py` that intercepts `torch.ops.aten.slice.Tensor` calls and clamps the `start` and `end` arguments to `[-size, size]` whenever they fall below `-size`, matching PyTorch eager semantics. This is a pure Python change that requires no rebuild.

File changed:
- `tt-xla/python_package/tt_torch/torch_overrides.py`

Remediation branch: `remediation/neural_monarch-pytorch-7B-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    116.14s (0:01:56)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1b06492f0ba45c954614a237e186f3e318d0bbf1 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
