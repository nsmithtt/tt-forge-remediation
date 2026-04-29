# Remediation Summary: biomistral_7b_dare-causal_lm-pytorch-biomistral_7b_dare-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[biomistral_7b_dare/causal_lm/pytorch-biomistral_7b_dare-single_device-inference]

## Result
SILICON_PASS — clamped out-of-range negative slice start in TorchFunctionOverride

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
RuntimeError: Value out of range (expected to be in range of [-128, 127], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})

Origin: transformers/cache_utils.py line 214:
  self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
BioMistral-7B-DARE uses the Mistral architecture with `sliding_window=4096`. During inference with `seq_len=128`, the `StaticSlidingWindowLayer.update()` in transformers' cache_utils does `full_value_states[:, :, -sliding_window + 1 :, :]`, i.e. `[:, :, -4095:, :]`. In standard PyTorch, a start index more negative than `-dim_size` is silently clamped to 0, returning the full tensor. The XLA eager backend used by tt-xla enforces strict bounds: start must be in `[-dim_size, dim_size-1]`. Since `4095 >> 128`, this raises `RuntimeError: Value out of range`. The check fires in `TorchFunctionOverride.__torch_function__` during `partition_fx_graph_for_cpu_fallback` — the XLA dry-run that partitions the FX graph.

The bug lives in the **tt-xla compiler frontend**: the XLA backend doesn't clamp out-of-range negative slice starts the way PyTorch CPU does.

## Fix
Added a guard in `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor` and we are not inside torch.compile (to avoid intercepting the trace itself), the start and end indices are pre-clamped to `[-dim_size, dim_size-1]` before dispatch, matching PyTorch CPU semantics.

File changed: `python_package/tt_torch/torch_overrides.py` (+19 lines)

Remediation branch: `remediation/biomistral_7b_dare-causal_lm-pytorch-biomistral_7b_dare-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    53.08s
- Tier A attempts: 1

## Files changed
- tt-xla: `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a25fac2e689a9d26fae8f2709f30eab4a388bc45 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
