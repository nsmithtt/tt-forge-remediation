# Remediation Summary: gemma3_4b_it_openbookqa_sft_c-pytorch-Gemma3_4B_IT_OpenbookQA_SFT_C-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_4b_it_openbookqa_sft_c/pytorch-Gemma3_4B_IT_OpenbookQA_SFT_C-single_device-inference]

## Result
FAIL — Tier A fix applied (aten.slice OOB clamping in TorchFunctionOverride); silicon verification blocked by gated model (HF account lacks access to google/gemma-3-4b-it)

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
```
E   RuntimeError: Value out of range (expected to be in range of [-32, 31], but got -1023)
```

## Root cause
The XLA lazy backend raises `Value out of range` when `aten.slice.Tensor` is
called with a negative start index whose absolute value exceeds the tensor
dimension size. PyTorch eager silently clamps such indices; the XLA lazy
backend does not.

Gemma3 4B uses `SlidingWindowCache` with `sliding_window=1024`. In
`load_inputs`, `max_length=32` tokens are fed. During the forward pass the
cache update computes `start = position - sliding_window + 1 = 0 - 1024 + 1
= -1023` on a cache dim of size 32. PyTorch eager accepts this silently (the
slice just returns all 32 elements), but the XLA lazy backend rejects it with
`Value out of range (expected [-32, 31], got -1023)`.

The bug is in `tt-xla` — `TorchFunctionOverride.__torch_function__` does not
pre-clamp out-of-range negative slice indices before dispatching to the XLA
lazy backend.

## Fix
**File:** `tt-xla/python_package/tt_torch/torch_overrides.py`

Added an `aten.slice.Tensor` interception block in
`TorchFunctionOverride.__torch_function__`. When the slice start or end is
less than `-size` (where `size` is the tensor's extent along `dim`), it is
clamped to `-size` before forwarding to the XLA backend. This matches
PyTorch eager's silent-clamp semantics.

This is the same fix previously confirmed `SILICON_PASS` on:
- `gemma3_12b_cybersecurity_gguf` (714.48s on n150)
- `gemma3_12b_qat_gguf` (626.46s on wormhole)
- `BgGPT-7B-Instruct-v0.2`
- `c2s_scale_gemma_2_2b`

Silicon verification was attempted on this machine but blocked: the base
model `google/gemma-3-4b-it` is a gated HuggingFace model and this machine's
HF account does not have authorized access. The remediation branch is pushed;
a machine with proper HF credentials can run the test to confirm.

## Verification
- pytest exit: not-run
- Hardware:    not-run
- Duration:    not-run
- Tier A attempts: 1 (fix applied; test could not be executed due to gated model)

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py` — added `aten.slice.Tensor` OOB start clamping in `TorchFunctionOverride`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2a5622e838a0134b477e91ac5974932beef26b25 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
