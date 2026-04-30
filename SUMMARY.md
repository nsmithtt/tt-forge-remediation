# Remediation Summary: docling_layout_heron/pytorch-docling_layout_heron-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[docling_layout_heron/pytorch-docling_layout_heron-single_device-inference]

## Result
FAIL — RTDetrV2 deformable attention produces near-zero PCC (-0.008) on TT device after the loader-layer int64 guard is fixed; numerical divergence in grid_sample decomposition is a Tier B compiler bug.

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
rtdetrv2-grid-sample-deformable-attn-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (before loader fix):

```
transformers/models/rt_detr_v2/modeling_rt_detr_v2.py:185: in forward
    torch_compilable_check(
        (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length, ...)
transformers/utils/import_utils.py:1395: in torch_compilable_check
    torch._check_tensor_all_with(error_type, cond, msg_callable)
torch/__init__.py:1808: in _check_tensor_all_with
    _check_with(error_type, cond._is_all_true().item(), message)
ValueError: Make sure to align the spatial shapes with the sequence length of the encoder hidden states
```

After loader fix applied:

```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=-0.007887676887135925. Required: pcc=0.99.
```

## Root cause
Two bugs were found:

**Bug 1 (loader — fixed):** `RTDetrV2MultiscaleDeformableAttention.forward` performs a sanity check using `(spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == sequence_length`, where `spatial_shapes` is a `torch.long` tensor created on the TT device (because `device = pixel_values.device`). TT hardware does not support `int64` element-wise arithmetic, so the multiplication and `.sum()` produce incorrect values, causing the equality to evaluate to False and raise ValueError. The fix replaces this check with equivalent Python-integer arithmetic over `spatial_shapes_list` (a list of `(height, width)` tuples always passed alongside the tensor that remains on CPU/Python side).

**Bug 2 (tt-mlir — unfixed Tier B):** After the loader fix the model compiles and runs on TT hardware but produces PCC of -0.008 (near-zero correlation) against the CPU reference. `RTDetrV2MultiscaleDeformableAttention` calls `F.grid_sample` (bilinear, `align_corners=False`) to sample multi-scale feature maps inside the deformable cross-attention. This op is decomposed via `aten.grid_sampler_2d` in `tt-xla/python_package/tt_torch/backend/decompositions.py`. The decomposed bilinear grid-sampling implementation uses `floor`, `clamp`, and scatter-like gather operations that produce numerically incorrect results on TT hardware, corrupting all 6 decoder layers and yielding garbage logits.

## Fix
**Loader fix (applied):** `tt_forge_models/docling_layout_heron/pytorch/loader.py`
Added `_patch_rtdetrv2_for_tt_device(model)` which monkey-patches every `RTDetrV2MultiscaleDeformableAttention` instance's `forward` to replace the `torch_compilable_check` on `spatial_shapes` (torch.long TT tensor) with a Python-integer equivalence check over `spatial_shapes_list`. Branch: `remediation/docling-layout-heron-pytorch-single-device-inference` in `tenstorrent/tt-forge-models`.

**Compiler fix (not attempted — Tier B):** A correct native tt-mlir lowering for `aten.grid_sampler_2d` that produces accurate bilinear interpolation results on TT hardware. The fix would require new or corrected lowering patterns in `tt-mlir` and potentially a new tt-metal kernel, touching more than 3 files across multiple repos.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting — Correct bilinear grid sampling requires new or corrected tt-mlir op lowering patterns and potentially a new tt-metal kernel; diagnosis and fix span multiple repos and more than 3 files.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 332.92s (5:32)
- Tier A attempts: 0

## Files changed
- `tt_forge_models/docling_layout_heron/pytorch/loader.py` (loader fix, remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 56b6d7c284046488377bc4f8a1fcca8c5a6e574b |
