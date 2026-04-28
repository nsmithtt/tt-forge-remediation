# Remediation Summary: glm_4_6v-conditional_generation-pytorch-glm_4_6v_flash-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_6v/conditional_generation/pytorch-glm_4_6v_flash-single_device-inference]

## Result
FAIL — PJRT device-to-host transfer fails when extracting integer scalar from `image_grid_thw` TT device tensor for `torch.arange(h)` in the visual encoder's `rot_pos_emb`

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Glm46VImageProcessor` is now loaded as a fast
processor by default, even if the model checkpoint was saved with a slow
processor. This is a breaking change and may produce slightly different
outputs. To continue using the slow processor, instantiate this class with
`use_fast=False`.

(After the loader fix, the actual test failure is:)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

at transformers/models/glm4v/modeling_glm4v.py:734:
    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
via python_package/tt_torch/torch_overrides.py:34 __torch_function__

## Root cause

Two issues:

1. **Loader bug (fixed)**: `AutoProcessor.from_pretrained` was called without
   `use_fast=False`. In transformers 5.x, `Glm46VImageProcessor` now loads
   as a fast image processor by default, raising a FutureWarning about
   potentially different outputs. Added `use_fast=False` to restore the
   slow-processor behaviour.

2. **Compiler-stack bug (Tier B, unfixed)**: The model's `Glm4vVisionModel.rot_pos_emb`
   iterates over the `image_grid_thw` input with a Python for loop:
   ```python
   for t, h, w in grid_thw:
       hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
   ```
   The test runner's `to_device()` moves ALL input tensors—including the
   integer tensor `image_grid_thw`—to the TT device. When the for loop
   extracts a row (`h` becomes a 0-d TT device tensor), `torch.arange(h)`
   requires the scalar value from that device tensor. This triggers a
   device-to-host copy in the PJRT backend (`buffer_instance.cc` →
   `CopyToHostThread` → `move_to_host()`), which fails with `kInternal`
   (Error code: 13). The TT PJRT backend does not support device-to-host
   transfer of integer scalar tensors.

## Fix

### Applied — loader fix in tt_forge_models

**File**: `glm_4_6v/conditional_generation/pytorch/loader.py`
**Branch**: `remediation/glm_4_6v-conditional_generation-pytorch-glm_4_6v_flash-single_device-inference`

Changed `AutoProcessor.from_pretrained(...)` to
`AutoProcessor.from_pretrained(..., use_fast=False, ...)` to suppress the
transformers 5.x `Glm46VImageProcessor` breaking change.

### Not applied — Tier B compiler-stack bug

The `pjrt-device-to-host-transfer` bug requires the TT PJRT backend to
support integer tensor device-to-host copies for scalar extraction.
Proposed fix location: `pjrt_implementation/src/api/buffer_instance.cc`
(specifically the `CopyToHostThread` that handles `kInternal` on failure)
and/or the underlying TT-Metal `move_to_host()` path for integer dtypes.
This is a cross-cutting infrastructure change to the PJRT layer.

## Tier B justification
Tier B indicator: `new-infrastructure`. Fixing the device-to-host scalar
transfer for integer tensors requires changes to the PJRT buffer transfer
infrastructure (`buffer_instance.cc` → `move_to_host()` for int dtype). It
is not a scoped single-function fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    219.32s (0:03:39)
- Tier A attempts: N/A

## Files changed
- `glm_4_6v/conditional_generation/pytorch/loader.py` (tt_forge_models — use_fast=False)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e08afadc497bd7dd1c71b6cf4d5eb93a35265b48 |
| tt-forge-models | 4b335fa6ba68e02c649bf2f30d3ac72b34a90fd4 |
