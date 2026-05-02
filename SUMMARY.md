# Remediation Summary: mai_ui_8b-image_to_text-pytorch-mai_ui_8b-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mai_ui_8b/image_to_text/pytorch-mai_ui_8b-single_device-inference]

## Result
FAIL — terminal Tier B compiler bug: `_deepstack_process` boolean-indexed gather `hidden_states[visual_pos_masks, :]` generates a dynamic shape via `stablehlo.set_dimension_size` that hangs Shardy during compilation; device timeout after ~2.5 minutes

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
set-dimension-size-shardy-dynamic-shape

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Original failure: `fast_pos_embed_interpolate` at line 699 calls `grid_thw.tolist()` on a TT-device tensor,
triggering a device sync that fails with INTERNAL Error code: 13.

After loader fixes + Conv3d fix: the test runs for ~2.5 minutes and then hangs in
`torch_dynamo_resume_in__deepstack_process_at_949` with:
```
TT_THROW: TIMEOUT: device timeout in fetch queue wait, potential hang detected
```
The Dynamo trace shows compilation blocked inside `_deepstack_process` → `extract_compiled_graph_helper`
→ `partition_fx_graph_for_cpu_fallback` → `wait_device_ops`.

## Root cause
Three issues were found in sequence:

**Issue 0 (loader, fixed):** `Qwen3VLVisionModel.fast_pos_embed_interpolate` calls
`grid_thw.tolist()` inside the compiled forward; on TT device, `.tolist()` triggers an
eager PJRT sync that fails with INTERNAL:13. Similarly, `rot_pos_emb`, `get_rope_index`,
and `get_image_features` call `.tolist()` on metadata tensors on TT device.
Also: `get_placeholder_mask` calls a runtime check via boolean-indexed gather
(`inputs_embeds[special_image_mask]`) which has a data-dependent output shape that XLA
cannot compile statically.
Also: the original loader had `device_map="auto"` and `dtype="auto"` which interfere
with TT device placement; pixel limits need to be set on the processor.
Fix: patch `fast_pos_embed_interpolate` to reimplement bilinear interpolation entirely
on CPU (`@torch.compiler.disable`) and return via `xm.send_cpu_data_to_device`; move
`rot_pos_emb`/`get_rope_index`/`get_image_features` grid_thw to CPU before calling
originals; reimplement `get_placeholder_mask` without the boolean-mask check.

**Issue 1 (tt-mlir, Tier A, fixed):** `Qwen3VLVisionPatchEmbed` uses
`Conv3d(in=3, out=1152, kernel=[2,16,16])`. With `C_in_block=TILE_WIDTH=32`, the
kernel volume 512 tiles causes:
- `vol2col_tiled` CB = 512 × 1 × 2048 bytes = 1.0 MB
- `weight_tiled` CB = 1 × 512 × 2048 bytes = 1.0 MB
- Total = 2.0 MB > 1.5 MB L1 → INTERNAL error 13 during first sync after Conv3d compile.

Fix in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
`Conv3dOpConversionPattern::matchAndRewrite`: when `kernelVol > 384`, set
`cInBlock = TILE_WIDTH/2 = 16` (minimum that satisfies hardware l1_alignment=16
constraint). Pass an explicit `Conv3dConfigAttr` with `c_in_block=16` and
`c_out_block=TILE_WIDTH=32` so the runtime uses the same blocking as the compiled
weight layout.

**Issue 2 (tt-mlir, Tier B, unfixed):** After the Conv3d fix, the vision encoder
runs successfully but compilation hangs at `Qwen3VLTextModel._deepstack_process`
(line 949). This function performs:
```python
local_this = hidden_states[visual_pos_masks, :]  # boolean-indexed gather
hidden_states[visual_pos_masks, :] = local_this + visual_embeds
```
The boolean-indexed gather `hidden_states[visual_pos_masks, :]` where
`visual_pos_masks.shape = [1, seq_len]` is lowered by XLA via `nonzero` →
`stablehlo.set_dimension_size`, creating a dynamic-shape output tensor.
`tt-mlir` / Shardy cannot handle this static-to-dynamic shape transition and the
compilation hangs (device timeout after TT_METAL_OPERATION_TIMEOUT_SECONDS
elapses). Same root cause as the `[stablehlo.set_dimension_size Shardy static shape]`
Tier B bug reported previously.

## Fix
**Loader fixes (tt_forge_models commit `374fb352da` + `662fc4ad6a`):**
- `mai_ui_8b/image_to_text/pytorch/loader.py`:
  - Remove `device_map="auto"` and `dtype="auto"` from model kwargs
  - Set pixel limits: `min_pixels=56*56, max_pixels=13*28*1280`
  - Patch `Qwen3VLVisionModel.fast_pos_embed_interpolate` with CPU reimplementation
    using pre-captured `pos_embed.weight` and `xm.send_cpu_data_to_device`
  - Patch `rot_pos_emb`, `get_rope_index`, `get_image_features` to move
    grid_thw/image_grid_thw to CPU before calling originals
  - Reimplement `get_placeholder_mask` without `torch_compilable_check`
    (skips boolean-mask validation that has data-dependent output shape)

**Compiler fix (tt-mlir commit `1a9cd2d3f`):**
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` `Conv3dOpConversionPattern`:
  - Compute `cInBlock = (kernelVol > 384) ? 16 : TILE_WIDTH`
  - Pass explicit `Conv3dConfigAttr(c_in_block=cInBlock, c_out_block=TILE_WIDTH)`

**Proposed fix for Issue 2 (not attempted — Tier B):**
The fix requires adding support in tt-mlir for `stablehlo.set_dimension_size` through
Shardy sharding propagation, or adding a pass that converts dynamic-shape boolean
gathers to a static equivalent (e.g., padding to a fixed size using a mask). This
requires new infrastructure across multiple passes in tt-mlir.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

The `_deepstack_process` boolean gather generates `stablehlo.set_dimension_size` which
creates a dynamic-shape tensor that Shardy cannot propagate through statically. Adding
support requires a new pass or fundamental changes to the dynamic-shape handling in
tt-mlir's compilation pipeline. This is the same class as the previously documented
`set-dimension-size-shardy-dynamic-shape` Tier B bug.

## Verification
- pytest exit: FAIL (device timeout / hang)
- Hardware:    n150
- Duration:    ~2.5 min (hung, terminated by timeout)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/mai_ui_8b/image_to_text/pytorch/loader.py` (two loader fix commits)
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (Conv3d L1 fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 1a9cd2d3f420cb1bda56b8cfe75a182899cd7e64 |
| tt-xla          | 56c04e235b81019399cc63e9ccc22eb23ae8d15a |
| tt-forge-models | 662fc4ad6a89da1294007598e5251dadbbe562f3 |
