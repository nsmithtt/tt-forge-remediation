# Remediation Summary: grok_3_reasoning_gemma3_12b_distilled_hf_gguf-causal_lm-pytorch-12B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[grok_3_reasoning_gemma3_12b_distilled_hf_gguf/causal_lm/pytorch-12B_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

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
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_51, 2, -1023, 9223372036854775807), kwargs = {})

## Root cause
Gemma3 12B has `sliding_window=1024`. The SlidingWindowCache slices KV state as
`full_value_states[:, :, -sliding_window + 1 :, :]`. With `max_length=128`, the
seq_len after tokenization is 23. The start index is `-1023`, which is outside
the valid XLA range `[-23, 22]` for a dim of size 23. PyTorch CPU silently clamps
such out-of-bound indices; XLA validates strictly and raises RuntimeError.

The bug is in `TorchFunctionOverride.__torch_function__` in tt-xla, which does not
clamp slice start/end indices before forwarding to XLA, causing the error during
`partition_fx_graph_for_cpu_fallback`.

## Fix
Added a guard in `TorchFunctionOverride.__torch_function__` in
`tt-xla/python_package/tt_torch/torch_overrides.py`: when
`func is torch.ops.aten.slice.Tensor`, clamp both the start (args[2]) and end
(args[3]) indices to `max(-dim_size, idx)` before forwarding. This matches
PyTorch CPU's silent clamping behavior.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    794.03s (0:13:14)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 306634b041d9a54d4871ae321ed46e91c7371eaa |
| tt-forge-models | f12c41f32b1b243a8b1ab0931f59978228027980 |
