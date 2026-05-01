# Remediation Summary: medix_r1_8b_gguf/image_to_text/pytorch-8b_q4_k_m-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[medix_r1_8b_gguf/image_to_text/pytorch-8b_q4_k_m-single_device-inference]

## Result
FAIL — `ttnn::prim::concat` allocates 14,594,560 B CBs on a 3-core grid; 9× L1 limit; device enters unrecoverable state

## Stack layer
loader, tt-mlir, tt-metal

## Tier
A

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
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

C++ root (captured with TT_METAL_LOGGER_LEVEL=FATAL):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=2)]
grow to 14594560 B which is beyond max L1 size of 1572864 B
  ttnn::prim::concat
```
After the device enters error state, a subsequent 30-second timeout follows:
```
TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
TT_THROW: Device 0: Timeout (30000 ms) waiting for physical cores to finish
```

## Root cause

Four bugs were found in sequence:

**Bug 1 (fixed — loader): `qwen3vl` GGUF arch not registered**
`MBZUAI/MediX-R1-8B-GGUF` stores `general.architecture = "qwen3vl"`. Transformers does
not have `qwen3vl` in `GGUF_SUPPORTED_ARCHITECTURES`, raising `ValueError: GGUF model
with architecture qwen3vl is not supported yet.`

**Bug 2 (fixed — loader): wrong `vision_config.out_hidden_size`**
The MBZUAI HF repo has no `config.json`. Our GGUF-derived config provides
`text_config.hidden_size = 4096` but no `vision_config`, so the Qwen3VLConfig default
`vision_config.out_hidden_size = 3584` (for the 2B/4B model) is used instead of 4096
(the 8B model value). This causes `Qwen3VLModel.get_placeholder_mask` to raise:
`ValueError: Image features and image tokens do not match, tokens: 2752, features: 2752`
because `inputs_embeds[special_image_mask].numel() (2752 × 4096) ≠ image_features.numel()
(2752 × 3584)`.

**Bug 3 (fixed — loader): `.tolist()` on TT device tensors**
`Qwen3VLVisionModel.fast_pos_embed_interpolate` calls `grid_thw.tolist()` on a TT device
tensor inside a `torch.compile`-ed graph, triggering a PJRT device-to-host transfer that
fails with `INTERNAL: Error code: 13`.

**Bug 4 (fixed — tt-mlir, Tier A): Conv3d CB overflow**
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152,
kernel=[2,16,16])`. `Conv3dOpConversionPattern::matchAndRewrite` in TTIRToTTNN.cpp
hardcoded `cInBlock = TILE_WIDTH = 32`, giving `K_t = 2×16×16 = 512 tiles`. The two main
CBs (`vol2col_tiled + weight_tiled = 2 × 1 MB`) exceed the 1.5 MB L1 limit. Fixed by
computing `kernelVol = 512 > L1_K_TILES = 384` and setting `cInBlock = TILE_WIDTH/2 = 16`,
halving both CBs to fit within L1.

**Bug 5 (unfixed — tt-metal, Tier B): concat CB overflow**
After the Conv3d fix, the visual encoder compiles and runs through all 27 transformer
blocks. Then `ttnn::prim::concat` allocates 14,594,560 B CBs on a 3-core grid
`(x=0,y=0)-(x=0,y=2)`, overflowing 1.5 MB L1 by ~9×. This corresponds to concatenating
the image embedding sequence (merged visual tokens) into the text embedding sequence
before the language model forward pass. The concat program factory allocates CBs for the
full per-core output without bounding to L1 capacity. This is the same bug as in the
`felldude_qwen_3_vl_uncensored_gguf` report (identical error: 14,594,560 B on the same
3-core grid).

## Fix

**Loader fixes** committed to `remediation/medix-r1-8b-gguf-single-device-inference` in tt_forge_models:
- `medix_r1_8b_gguf/image_to_text/pytorch/loader.py`:
  - `_register_qwen3vl_gguf_support()`: adds `qwen3vl` to `GGUF_SUPPORTED_ARCHITECTURES`
    and `GGUF_TO_TRANSFORMERS_MAPPING["config"]`
  - `_qwen3vl_load_gguf` wrapper: translates flat GGUF config → nested `Qwen3VLConfig`
    (`model_type = "qwen3_vl"`, `text_config`, `vision_config.out_hidden_size = hidden_size`)
  - Direct GGUF tensor loading via `_load_qwen3vl_tensors` with hard-coded name map,
    bypassing `get_gguf_hf_weights_map` multi-submodule traversal issue
  - `_patch_qwen3vl_for_tt_device()`: patches `fast_pos_embed_interpolate`, `rot_pos_emb`,
    `get_rope_index`, `get_image_features` to move `.tolist()`-calling tensors to CPU;
    `fast_pos_embed_interpolate` reimplemented on CPU with pre-captured `pos_embed.weight`
  - Pixel limits: `min_pixels=56*56, max_pixels=13*28*1280`

**Tier A fix** committed to `remediation/medix-r1-8b-gguf-single-device-inference` in tt-mlir:
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`:
  `Conv3dOpConversionPattern::matchAndRewrite`: compute `cInBlock = (kernelVol > 384) ?
  TILE_WIDTH/2 : TILE_WIDTH`; pass explicit `Conv3dConfigAttr` so runtime uses same
  block sizes as compiled weight layout.

**Proposed fix for Bug 5 (concat, Tier B):**
`ConcatDeviceOperation` program factory needs to bound per-core CB allocation to the L1
capacity (`1,572,864 B`). When the naive per-core output exceeds L1, the operation should
either shard across more cores, use DRAM-backed CBs, or break the concat into multiple
passes. This requires coordinated changes across the concat program factory and potentially
the TTNN op dispatch layer.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting: The concat CB overflow requires either a new multi-pass concat strategy or
changes to the concat program factory's core-count selection logic — changes that would
affect all consumers of `ttnn::concat`, not just this model.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    563.16s (0:09:23) for last run before reporting (Conv3d fix + concat fail)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/medix_r1_8b_gguf/image_to_text/pytorch/loader.py` (3 commits)
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (1 commit)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | e6a22682613732a83ea87c3ea6b156d8d2c467de |
| tt-xla          | fc78476fddfd1cbff46289f6913daf45dce1a525 |
| tt-forge-models | 6e7e96de9c7c7f7a34f0e7be19028377df48eb2c |
