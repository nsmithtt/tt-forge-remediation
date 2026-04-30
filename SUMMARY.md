# Remediation Summary: ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic-image_to_text-pytorch-4b_claude_opus_4_6_distilled_heretic-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic/image_to_text/pytorch-4b_claude_opus_4_6_distilled_heretic-single_device-inference]

## Result
FAIL — second compiler-stack bug: `ttnn::prim::concat` allocates 9 MB CBs on a 3-core grid (6× L1 limit); Tier B

## Stack layer
loader, tt-mlir, tt-metal

## Tier
B

## Bug fingerprint
concat-cb-size-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
Test exceeded configured timeout and was killed
```

## Root cause
Three loader bugs and two compiler-stack bugs were discovered, each unmasked by fixing the previous:

**Loader bug 1 — missing preprocessor_config.json**: The model repo
`ghost-actual/Qwen3.5-4B-Claude-Opus-4.6-Distilled-heretic` does not ship a
`preprocessor_config.json`. `AutoProcessor.from_pretrained` makes a network
request for the file, which either hangs (causing CI timeout) or raises an
OSError. Fix: construct `Qwen3VLProcessor` manually using `Qwen2VLImageProcessor`
(the AutoImageProcessor mapping for `qwen3_5`) + `Qwen3VLVideoProcessor` +
tokenizer, passing `tokenizer.chat_template` explicitly.

**Loader bug 2 — wrong patch_size**: `Qwen2VLImageProcessor` defaults to
`patch_size=14` (Qwen2-VL). The Qwen3.5 vision config specifies `patch_size=16`,
causing a `RuntimeError: shape '[-1, 3, 2, 16, 16]' is invalid for input of size
2591904` during `Qwen3_5VisionPatchEmbed.forward`. Fix: pass `patch_size=16`
to `Qwen2VLImageProcessor`.

**Loader bug 3 — .tolist() on TT device tensors**: `Qwen3_5VisionModel.rot_pos_emb`,
`fast_pos_embed_interpolate`, and `Qwen3_5Model.get_rope_index` /
`get_image_features` call `.tolist()` on device tensors for Python control flow.
TT device does not support eager tensor reads; these sync with Error code: 13.
Fix: patch all four methods to move grid_thw / input_ids metadata to CPU before
`.tolist()`, then move computed outputs back to original device.

**Compiler bug 1 (Tier A, fixed) — Conv3d L1 CB overflow**: `Qwen3_5VisionPatchEmbed`
uses `nn.Conv3d(in_channels=3, out_channels=1024, kernel_size=[2,16,16])`. With
`cInBlock=TILE_WIDTH=32`, `K_t = kernelVol = 2×16×16 = 512`, making
`vol2col_tiled + weight_tiled = 2 × 2048 × 512 = 2.1 MB`, exceeding the 1.5 MB
L1 limit. Fix: use `cInBlock=16` when `kernelVol > 384`, halving K_t to 256 and
keeping CBs within L1.

**Compiler bug 2 (Tier B, unfixed) — concat CB overflow**: After the Conv3d fix,
`ttnn::prim::concat` allocates `9163264 B` of CBs on a 3-core grid (6× the 1.5 MB
L1 limit) when concatenating large vision feature tensors in the Qwen3.5 language
model path. The concat program factory does not bound CB allocation per core, causing
device error state 13.

## Fix
**Loader fixes** committed to
`remediation/ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic-...`
in `tt_forge_models` (3e752acf0b):
- `ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic/image_to_text/pytorch/loader.py`:
  construct `Qwen3VLProcessor` manually with `Qwen2VLImageProcessor(patch_size=16,
  min_pixels=56*56, max_pixels=13*28*1280)` + `Qwen3VLVideoProcessor`; patch
  `Qwen3_5VisionModel.rot_pos_emb`, `fast_pos_embed_interpolate`,
  `Qwen3_5Model.get_rope_index`, and `get_image_features` to move metadata to CPU.

**Conv3d fix (Tier A)** committed to
`remediation/ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic-...`
in `tt-mlir` (ce4cc60d3):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` `Conv3dOpConversionPattern`: use
  `cInBlock = TILE_WIDTH/2 = 16` when `kernelVol > 384`; pass explicit
  `Conv3dConfigAttr` with `c_in_block` and `c_out_block=TILE_WIDTH`.

**Proposed fix for Bug 2** (not attempted): In `tt-metal`, the
`ConcatDeviceOperation` program factory needs to bound CB allocations per core
to fit within L1. For large tensors the factory should either chunk the work
across more cores or use a streaming approach that limits the per-core CB
footprint.

## Tier B justification
Tier B indicator: `cross-cutting` — fixing the concat CB overflow requires changes
to the concat program factory in tt-metal (complex multi-core kernel scheduling
logic). The factory computes CB sizes based on the full tensor dimensions without
an L1 budget cap; adding chunking or streaming requires coordinated changes to the
kernel scheduling and the reader/writer programs (≥3 files, potentially 2 repos).

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    247.30s (0:04:07) to reach second compiler error
- Tier A attempts: 1

## Files changed
- `tt_forge_models`: `ghost_actual_qwen3_5_4b_claude_opus_4_6_distilled_heretic/image_to_text/pytorch/loader.py` (new file, 190 lines)
- `tt-mlir`: `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (+35/-7 lines)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | ce4cc60d31c3b4ffe5ef0dc401d7b1a00f2af80a |
| tt-xla          | 45ef4ebdec84e11ebdbd22056007ea36773dfdea |
| tt-forge-models | 3e752acf0b16b82cc88724f71b2daacc6e4cf13e |
