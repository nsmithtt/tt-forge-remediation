# Remediation Summary: davidau_qwen3_5_9b_claude_4_6_highiq_instruct_heretic/image_to_text/pytorch-9B_Claude_4.6_HighIQ_INSTRUCT_HERETIC_UNCENSORED-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[davidau_qwen3_5_9b_claude_4_6_highiq_instruct_heretic/image_to_text/pytorch-9B_Claude_4.6_HighIQ_INSTRUCT_HERETIC_UNCENSORED-single_device-inference]

## Result
FAIL — second compiler bug in `get_placeholder_mask` (boolean indexing with dynamic output shape on TT device) blocked silicon pass

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
dynamic-shape-boolean-index-embedding-scatter

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Original failure in `Qwen3_5VisionPatchEmbed` Conv3d forward pass (L1 circular buffer overflow). After fixing the Conv3d L1 issue, the test progressed to `get_placeholder_mask` and failed again with the same error code 13, this time due to dynamic-shape boolean indexing.

## Root cause

Two bugs were found in sequence:

**Bug 1 (fixed): Conv3d L1 overflow in `tt-mlir` — `conv3d-patch-embed-l1-overflow`**

`Qwen3_5VisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152, kernel=[2,16,16])`. The `Conv3dOpConversionPattern` in `TTIRToTTNN.cpp` was constructing a `Conv3dConfigAttr` with no blocking parameters, causing `conv3d_program_factory.cpp` to allocate `cb_vol2col_tiled + cb_weight_tiled = 2.1 MB` of circular buffers per core, exceeding the L1 limit (~1.5 MB). The fix computes a safe `c_in_block` (16 for 2×16×16 kernel) and explicitly sets `c_out_block=TILE_WIDTH=32` plus the full device worker grid, reducing each dominant CB to ≤512 KB.

**Bug 2 (not fixed): Dynamic-shape boolean indexing in `get_placeholder_mask`**

After the vision encoder ran successfully, the model's `Qwen3_5Model.get_placeholder_mask` executes:
```python
special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
torch_compilable_check(inputs_embeds[special_image_mask].numel() == image_features.numel(), ...)
```
The expression `inputs_embeds[special_image_mask]` is a boolean-masked gather that produces a tensor whose shape is unknown at compile time (depends on the number of image tokens). TT device compilation requires static shapes; this dynamic-shape indexing triggers Error code: 13. Per skill rules, after the first compiler-stack fix the second bug is filed as FAIL.

## Fix

**Bug 1 fix** — in `tt-mlir` (`remediation/davidau_qwen3_5_9b_claude_4_6_highiq_instruct_heretic-image_to_text-pytorch-9B_Claude_4.6_HighIQ_INSTRUCT_HERETIC_UNCENSORED-single_device-inference`):

`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `Conv3dOpConversionPattern::matchAndRewrite`:
Compute a safe `c_in_block` that keeps `cb_vol2col_tiled` within L1:
```
maxCInBlock = MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k)
```
With `MAX_CB_TILES=256` (512 KB / 2048 B), `kernel=[2,16,16]` → `maxCInBlock=16`.
Set `c_out_block=TILE_WIDTH=32` (keeps `cb_weight_tiled = tile_size * K_t * 1 = 512 KB`).
Set `compute_with_storage_grid_size` to the full device worker grid via `ttcore::lookupDevice(op).getWorkerGrid()` so that the `C_in_blocks <= total_cores` FATAL check passes with `c_in_block < TILE_WIDTH`.

**Bug 1 fix also required (loader)**:

`tt-xla/third_party/tt_forge_models/davidau_qwen3_5_9b_claude_4_6_highiq_instruct_heretic/image_to_text/pytorch/loader.py` — added `_patch_qwen3_5_for_tt_device()` patching `Qwen3_5VisionModel.fast_pos_embed_interpolate`, `Qwen3_5VisionModel.rot_pos_emb`, `Qwen3_5Model.get_rope_index`, `Qwen3_5Model.get_image_features` to move `grid_thw`/`input_ids` to CPU before `.tolist()` calls. Added pixel limits (`min_pixels=56*56, max_pixels=13*28*1280`) to keep patch count within hardware budget.

**Bug 2 proposed fix** — in `tt-xla` or `tt-mlir`:

The root cause is that TorchDynamo/TT compilation cannot handle `tensor[bool_mask]` when the output shape is dynamic. Two possible approaches:
1. In `tt-xla` compilation pipeline: add a graph-break or guard on data-dependent tensor sizes inside `get_placeholder_mask`, allowing the embedding scatter to run on CPU.
2. In the model loader: override `get_placeholder_mask` to replace the `torch_compilable_check` boolean-index with a static-shape alternative (e.g., `torch.where` scatter into a pre-allocated buffer), analogous to the `.tolist()` patch pattern used for `get_rope_index`.

Option 2 is feasible as a loader-layer workaround but the real fix requires TT device support for dynamic-shape indexing (new infrastructure → Tier B).

## Tier B justification
**Indicator: new-infrastructure**

`inputs_embeds[special_image_mask]` is a data-dependent boolean gather that produces an output whose shape cannot be determined statically. Supporting this on TT device requires dynamic-shape tensor infrastructure in the PJRT compilation pipeline — a larger project than a scoped one- or two-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    308.64s
- Tier A attempts: 1

## Files changed
**tt-mlir (remediation branch):**
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — compute safe c_in_block, set c_out_block=32 and full device grid in Conv3dConfigAttr

**tt-xla / tt-forge-models (remediation branch):**
- `third_party/tt_forge_models/davidau_qwen3_5_9b_claude_4_6_highiq_instruct_heretic/image_to_text/pytorch/loader.py` — added `_patch_qwen3_5_for_tt_device()` and pixel limits

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 151e2000ce185c9cd344c4934963b03c838f91a2 |
| tt-xla          | 68b4337d54ac4840e411f87456bff5406d798914 |
| tt-forge-models | bdd42e0e508c5782b81c2e05a0cc5d6c7348fc68 |
