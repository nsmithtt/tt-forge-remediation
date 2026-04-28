# Remediation Summary: orvion_vl_3b_i1_gguf/image_to_text/pytorch-orvion_vl_3b_i1_gguf-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[orvion_vl_3b_i1_gguf/image_to_text/pytorch-orvion_vl_3b_i1_gguf-single_device-inference]

## Result
FAIL — rot_pos_emb calls grid_thw.tolist() on a TT device tensor; device-to-host transfer fails with INTERNAL: Error code: 13

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
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

(After the loader bug was fixed, the test proceeded to silicon and failed with:
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13)

## Root cause
Two issues were present:

**1. Loader bug (fixed)**: `AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")` loaded
`Qwen2VLImageProcessor` as the fast processor variant by default in transformers 5.x. Fixed by
adding `use_fast=False`.

**2. Silicon bug (Tier B)**: The Qwen2.5-VL visual encoder's `rot_pos_emb` method
(`transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:375`) calls `grid_thw.tolist()` in a
Python `for` loop. When the model runs on TT silicon, `grid_thw` is a TT device tensor. The
`tolist()` call dispatches through `tt_torch/torch_overrides.py:__torch_function__` and fails
because the TT PJRT layer does not support synchronous device-to-host data extraction during
model execution.

Stack trace (abbreviated):
```
transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:375 in rot_pos_emb
  for t, h, w in grid_thw.tolist():
python_package/tt_torch/torch_overrides.py:34 in __torch_function__
  return func(*args, **(kwargs or {}))
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Fix
The loader fix was applied in tt_forge_models:
- Branch: `remediation/orvion_vl_3b_i1_gguf-image_to_text-pytorch-orvion_vl_3b_i1_gguf-single_device-inference`
- File: `orvion_vl_3b_i1_gguf/image_to_text/pytorch/loader.py`
- Change: Added `use_fast=False` to `AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=False)`

The silicon bug is unfixed. The proposed fix would live in `tt-xla`'s PJRT layer to support
synchronous device-to-host tensor value extraction (`.tolist()`, `.item()`) when called on TT
device tensors from Python control flow. Without this, any Qwen2-VL or Qwen2.5-VL model with
image inputs will fail at the `rot_pos_emb` step of the visual encoder.

## Tier B justification
**Indicator**: cross-cutting — supporting `.tolist()` / `.item()` on TT device tensors requires
changes to the PJRT device-to-host transfer path in tt-xla. This affects any model that uses
Python-level control flow conditioned on tensor values mid-execution, not just this one model.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    240.78s
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/orvion_vl_3b_i1_gguf/image_to_text/pytorch/loader.py`: add `use_fast=False`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 87eceefdc4855f893b4404332bf2bb21210d0c86 |
