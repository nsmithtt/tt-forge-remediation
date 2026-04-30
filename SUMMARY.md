# Remediation Summary: embeddinggemma-embedding_generation-pytorch-embeddinggemma-german_qna-checkpoint-500-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[embeddinggemma/embedding_generation/pytorch-embeddinggemma-german_qna-checkpoint-500-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-10, 9], but got -256)

## Root cause
The EmbeddingGemma model uses a Gemma backbone with `SlidingWindowCache`. When the input sequence length (10 tokens for the sample sentence) is less than the sliding window size (256), `SlidingWindowCache.update()` computes a slice start of `-sliding_window + 1 = -256` on a cache dimension of size 10. PyTorch semantically clamps out-of-range negative start indices to `-dim_size`, but the XLA/TT backend validates the index strictly and raises `RuntimeError: Value out of range`. This is in the tt-xla compiler frontend — the FX graph contains `aten.slice.Tensor` with `start=-256` and `dim_size=10`, which is valid by PyTorch semantics but rejected by the XLA backend.

## Fix
Added `clamp_out_of_range_slice_starts(gm)` FX pass to `tt-xla/python_package/tt_torch/backend/passes.py`. The pass iterates `aten.slice.Tensor` nodes and, for each constant negative start index that falls below `-dim_size`, clamps it to `-dim_size`. Called from `torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`. Two files changed in tt-xla:

- `python_package/tt_torch/backend/passes.py` — new `clamp_out_of_range_slice_starts` function
- `python_package/tt_torch/backend/backend.py` — import and call after `bypass_assert_tensor_metadata`

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    113.39s (0:01:53)
- Tier A attempts: 1

## Files changed
- tt-xla/python_package/tt_torch/backend/passes.py
- tt-xla/python_package/tt_torch/backend/backend.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 640cc5c1f914d4e2e901c025daa091bf89422e2e |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
