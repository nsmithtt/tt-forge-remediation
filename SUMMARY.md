# Remediation Summary: gams3_12b_instruct-aten-slice-oob

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gams3_12b_instruct/causal_lm/pytorch-GAMS3_12B_INSTRUCT-single_device-inference]

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
E   RuntimeError: Value out of range (expected to be in range of [-29, 28], but got -1023)

## Root cause
The XLA lazy tensor backend (torch/csrc/lazy/core/helpers.cpp) validates slice start indices strictly: `start` must satisfy `-dim_size <= start < dim_size`. PyTorch eager silently clamps out-of-range negative starts to 0, but XLA raises RuntimeError instead.

GaMS3-12B-Instruct uses GemmaAttention with a sliding-window attention bias tensor whose sequence dimension is 1. The forward pass slices it with `attn_bias[:, :, -query_len:, :]` where `query_len` can be 30 (the tokenized input length), yielding `start = -30` on a dim of size 1 — out of the range `[-1, 0]`.

The fix is in the tt-xla compiler frontend: an FX graph pass (`clamp_out_of_range_slice_starts`) runs after decomposition and clamps any static negative `start` argument that falls below `-dim_size` up to `-dim_size`, matching PyTorch eager semantics.

## Fix
Added `clamp_out_of_range_slice_starts` FX pass to `tt-xla`:

- `python_package/tt_torch/backend/passes.py` — new function that iterates all `aten.slice.Tensor` nodes in the FX graph, identifies static integer `start` values that are out of range (`start < -dim_size`), and clamps them to `-dim_size`.
- `python_package/tt_torch/backend/backend.py` — imports and calls `clamp_out_of_range_slice_starts(compiled_graph)` in `torch_pass_pipeline`, after `bypass_assert_tensor_metadata`.

Branch: `remediation/gams3_12b_instruct-aten-slice-oob` in tt-xla repo.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    320.77s (0:05:20)
- Tier A attempts: 1

## Files changed
- tt-xla: python_package/tt_torch/backend/passes.py
- tt-xla: python_package/tt_torch/backend/backend.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7dcf7b6dde8d41103254e1b706fe83f263ae2904 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
