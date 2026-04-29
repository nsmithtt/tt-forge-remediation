# Remediation Summary: bggpt_7b_instruct_v0_2-causal_lm-pytorch-7B-Instruct-v0.2-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bggpt_7b_instruct_v0_2/causal_lm/pytorch-7B-Instruct-v0.2-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-19, 18], but got -4095)

## Root cause
BgGPT-7B-Instruct-v0.2 is a Mistral-based Bulgarian language model with `sliding_window=4096` in its attention configuration. On the first forward pass with `seq_len=19` (chat-template tokenized input), `SlidingWindowCache.update()` in transformers computes `full_value_states[:, :, -sliding_window+1:, :]` = `full_value_states[:, :, -4095:, :]`. This produces an `aten.slice.Tensor` node with `start=-4095` on a dimension of size 19 (valid XLA range: `[-19, 18]`). PyTorch CPU silently clamps out-of-range negative slice indices to 0; the XLA backend raises `RuntimeError: Value out of range` instead. The bug is in tt-xla's FX pass pipeline, which does not normalize such indices before handing the graph to XLA.

## Fix
Added `clamp_out_of_range_slice_starts(gm)` FX pass in `tt-xla/python_package/tt_torch/backend/passes.py`. The pass iterates all `aten.slice.Tensor` nodes, reads the slice dimension size from `node.args[0].meta["val"].shape` (or `tensor_meta.shape`), and clamps any `start` index that is more negative than `-dim_size` to `-dim_size`. Wired into `torch_pass_pipeline` in `tt-xla/python_package/tt_torch/backend/backend.py` after `bypass_assert_tensor_metadata`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    122.45s (0:02:02)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/backend/passes.py (added clamp_out_of_range_slice_starts)
- tt-xla/python_package/tt_torch/backend/backend.py (import + call in torch_pass_pipeline)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b7085d5b496c6284641b5ab81136793eb9045cd3 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
