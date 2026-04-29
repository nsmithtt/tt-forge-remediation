# Remediation Summary: bielik-causal_lm-pytorch-7B_Instruct_v0.1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bielik/causal_lm/pytorch-7B_Instruct_v0.1-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-15, 14], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback points to transformers/cache_utils.py:214:
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Bielik-7B-Instruct-v0.1 is Mistral-7B-based and uses sliding-window attention with `sliding_window=4096`. On the first forward pass the KV cache has only 15 tokens, so `SlidingWindowCache.update()` computes `full_value_states[:, :, -4095:, :]`, which becomes `aten.slice.Tensor(..., start=-4095, ...)` on a dimension of size 15 (valid range [-15, 14]). PyTorch eager silently clamps out-of-range negative slice starts to 0; the XLA lazy backend raises "Value out of range" instead. The fix is in tt-xla: add an FX pass that pre-clamps such start indices before the graph is submitted to XLA.

## Fix
Added `clamp_out_of_range_slice_starts(gm)` FX pass in `tt-xla/python_package/tt_torch/backend/passes.py`. The pass iterates `aten.slice.Tensor` nodes, reads `dim_size` from `node.args[0].meta["val"].shape`, and clamps `start` to `max(-dim_size, start)` when `start < -dim_size`. Wired into `torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`. Also fixed a `NameError` in `tests/infra/utilities/torch_multichip_utils.py` where `Mesh` was used as a return type annotation but not defined when `torch_xla` is absent (set `Mesh = None` in the `except ImportError` branch, removed the return type hint).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    99.82s (0:01:39)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/backend/passes.py (added clamp_out_of_range_slice_starts pass)
- tt-xla/python_package/tt_torch/backend/backend.py (import and call clamp_out_of_range_slice_starts after bypass_assert_tensor_metadata)
- tt-xla/tests/infra/utilities/torch_multichip_utils.py (fix NameError when torch_xla not installed)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 56131d5d7874defc3766bb5d2a4e6d0637cae88d |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
