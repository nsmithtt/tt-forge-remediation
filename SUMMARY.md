# Remediation Summary: nanonets_ocr-doc_ocr-pytorch-nanonets_ocr_s-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nanonets_ocr/doc_ocr/pytorch-nanonets_ocr_s-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer: grid_thw.tolist() on TT tensor in Qwen2.5-VL rot_pos_emb raises INTERNAL Error code: 13

## Stack layer
loader, tt-xla

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

at transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:375 in rot_pos_emb:
    for t, h, w in grid_thw.tolist():

(Original CI failure message: "The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.")

## Root cause
Two issues found:

1. **Loader bug (fixed)**: `AutoProcessor.from_pretrained` was called without `use_fast=False`, triggering a transformers 5.x breaking-change warning that the fast `Qwen2VLImageProcessor` is now the default. Fix: add `use_fast=False` to the `AutoProcessor.from_pretrained` call.

2. **Tier B compiler bug (terminal)**: `Qwen2_5_VLForConditionalGeneration.visual.rot_pos_emb` calls `grid_thw.tolist()` where `grid_thw` is a TT device tensor. `tolist()` triggers a device-to-host transfer via `__torch_function__` in `tt_torch/torch_overrides.py`, which raises INTERNAL: Error code: 13. The PJRT bridge does not support `tolist()`/`item()` on TT tensors. This is the same root cause as confirmed in nanonets_ocr2_aio_gguf's report (reverted tolist patches as Tier B).

## Fix
- **Loader fix committed**: `nanonets_ocr/doc_ocr/pytorch/loader.py` — added `use_fast=False` to `AutoProcessor.from_pretrained()`. Branch: `remediation/nanonets_ocr-doc_ocr-pytorch-nanonets_ocr_s-single_device-inference` in tt_forge_models.
- **Terminal Tier B bug**: No fix attempted. Requires PJRT infrastructure to support `tolist()` / device-to-host transfer on TT tensors. The same bug blocks all Qwen2.5-VL and Qwen3-VL pytorch loaders.

## Tier B justification
**new-infrastructure**: `tensor.tolist()` on a TT device tensor requires a synchronous device-to-host transfer path in the PJRT bridge. The current bridge raises `INTERNAL: Error code: 13` for any such transfer. Adding this capability requires new PJRT transfer infrastructure, not a scoped pattern fix.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 296.81s
- Tier A attempts: N/A

## Files changed
- `nanonets_ocr/doc_ocr/pytorch/loader.py` (in tt_forge_models): add `use_fast=False` to `AutoProcessor.from_pretrained`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a5ee01f69f6bec898ed6191fc8a68a376be3f7f0 |
| tt-forge-models | 5f1b428f8b (remediation/nanonets_ocr-doc_ocr-pytorch-nanonets_ocr_s-single_device-inference) |
