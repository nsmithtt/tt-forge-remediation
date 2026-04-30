# Remediation Summary: gritlm-causal_lm-pytorch-emb_m7_nodes16_fast-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gritlm/causal_lm/pytorch-emb_m7_nodes16_fast-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-8, 7], but got -4095)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_4, 2, -4095, 9223372036854775807), kwargs = {})
Original traceback:
  File ".../transformers/models/mistral/modeling_mistral.py", line 161, in forward
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
  File ".../transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
GritLM/emb_m7_nodes16_fast is a Mistral-based model with sliding_window=4096. The test input ("What is the capital of France?") tokenizes to 8 tokens. In DynamicSlidingWindowLayer.update() (cache_utils.py:214), the slice `full_value_states[:, :, -self.sliding_window + 1:, :]` becomes `[:, :, -4095:, :]` on a dim-2 tensor of size 8. PyTorch eager silently clamps the out-of-range start to 0, but the XLA/TT backend validates strictly and raises RuntimeError.

The fix lives in the tt-xla compiler frontend: an FX graph pass runs after decomposition and clamps any static negative `start` argument in `aten.slice.Tensor` nodes that falls below `-dim_size` up to `-dim_size`, matching PyTorch eager semantics.

## Fix
Added `clamp_out_of_range_slice_starts` FX pass to tt-xla:

- `python_package/tt_torch/backend/passes.py` — new function iterates all `aten.slice.Tensor` nodes in the FX graph, identifies static integer `start` values where `start < -dim_size` (using the node's `meta["val"].shape`), and clamps to `-dim_size`.
- `python_package/tt_torch/backend/backend.py` — imports and calls `clamp_out_of_range_slice_starts(compiled_graph)` in `torch_pass_pipeline`, after `bypass_assert_tensor_metadata`.

Branch: `remediation/gritlm-causal_lm-pytorch-emb_m7_nodes16_fast-single_device-inference` in tt-xla repo.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    91.36s (0:01:31)
- Tier A attempts: 1

## Files changed
- tt-xla: python_package/tt_torch/backend/passes.py
- tt-xla: python_package/tt_torch/backend/backend.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eb0f04bd3c1436f1f35ee594658fd36048e8ae9f |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
