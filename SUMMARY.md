# Remediation Summary: mistral_rrc-causal_lm-pytorch-mistral_rrc-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_rrc/causal_lm/pytorch-mistral_rrc-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

## Root cause
Mistral-RRC is a Mistral 7B fine-tune with `sliding_window=4096`. During KV-cache update in `transformers/cache_utils.py`, the model slices the concatenated key/value states as `full_value_states[:, :, -sliding_window + 1:, :]`, producing `start = -4095`. With `max_length=128` the tensor's sequence dimension is 128, so `-4095 < -128`. XLA's `aten.slice.Tensor` kernel validates that start >= -dim_size and raises RuntimeError, while PyTorch CPU silently clamps. The error fires in `TorchFunctionOverride.__torch_function__` at trace time (during `partition_fx_graph_for_cpu_fallback`), before any MLIR compilation.

## Fix
Added a guard in `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`: when `func is torch.ops.aten.slice.Tensor`, iterate over args[2] (start) and args[3] (end) and clamp any integer index that is less than `-dim_size` to `-dim_size`. This matches PyTorch's own clamping semantics and is safe for all callers since indices below `-dim_size` are equivalent to `0` in any slice semantics.

Remediation branch: `remediation/mistral_rrc-causal_lm-pytorch-mistral_rrc-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    137.41s (0:02:17)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/torch_overrides.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1ba830537450b285b7ae5a232c58cd47435511aa |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
