# Remediation Summary: diffusionvl_qwen2_5_vl-pytorch-3B-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[diffusionvl_qwen2_5_vl/pytorch-3B-single_device-inference]

## Result
FAIL — Conv3d patch_embed L1 overflow: statically allocated circular buffers (1,745,920 B) exceed tt-metal L1 max (1,572,864 B)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-patch-embed-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
The original CI failure (use_fast processor warning) was a loader-layer transformers 5.x breaking change. The existing remediation branch had already fixed four loader bugs (_tied_weights_keys list→dict, tie_weights **kwargs compat, _merge_vision_text IMAGE_TOKEN_INDEX clamp, attention_mask 4D bool). After all loader fixes are applied, the test fails with:

```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 1745920 B which is beyond max L1 size of 1572864 B
```

This comes from `DiffusionVL_Qwen2_5_VL_VisionPatchEmbed`, which uses `nn.Conv3d(in_channels=3, out_channels=1152, kernel_size=[2,14,14], stride=[2,14,14])`. The tt-metal `Conv3dDeviceOperation` statically allocates circular buffers based solely on kernel parameters, independent of batch size or number of patches. For this kernel, the allocation (1,745,920 B) exceeds the Wormhole L1 limit (1,572,864 B). This is the identical mechanism as the Qwen3VL conv3d-patch-embed-l1-overflow bug, with a slightly smaller kernel (patch_size=14 vs 16).

The error surfaced as INTERNAL: Error code: 13 during an XLA graph sync triggered by `grid_thw.cpu()` (a loader-layer fix for Python control-flow on device tensors) or by `torch.arange(h)` with a device-tensor argument in `rot_pos_emb`. In both cases the Conv3d was the first pending operation in the graph and its execution failed.

## Fix
Five loader-layer fixes were applied to `diffusionvl_qwen2_5_vl/pytorch/loader.py` in the `tt-forge-models` remediation branch:

1. **_tied_weights_keys list→dict** (`97deb103c7`): Custom model declares `_tied_weights_keys = ["lm_head.weight"]` (old list format). transformers 5.x requires a dict `{target: source}`. Patched class attribute to `{"lm_head.weight": "model.embed_tokens.weight"}` before `from_pretrained`.

2. **tie_weights **kwargs compat** (`19b0b3da50`): transformers 5.x calls `tie_weights(recompute_mapping=False)` but the custom model's `tie_weights()` does not accept `**kwargs`. Wrapped it with a `**kwargs`-accepting shim.

3. **_merge_vision_text IMAGE_TOKEN_INDEX clamp** (`8d09698daf`): The custom `_merge_vision_text` calls `embed_tokens(input_ids)` where `input_ids` may contain -200 (IMAGE_TOKEN_INDEX), which is out of range. Clamped -200 → 0 before the embedding lookup.

4. **attention_mask 4D bool** (`458e0d39a4`): The processor returns `int64` attention_mask with shape `[B, S]`; SDPA requires `bool` with shape `[B, 1, 1, S]`. Converted in the Wrapper's forward.

5. **rot_pos_emb / get_window_index CPU patches + use_fast=False** (`351a8269c3`): Added `use_fast=False` to `AutoProcessor.from_pretrained` (fixes original CI error). Patched `DiffusionVL_Qwen2_5_VL_VisionTransformer.rot_pos_emb` to iterate with `grid_thw.cpu().tolist()` instead of over device tensors, and wrapped `get_window_index` to move `grid_thw` to CPU and return `window_index` back to the original device.

The proposed fix for the Tier B Conv3d L1 overflow lives in tt-metal's `conv3d_program_factory.cpp` and in tt-mlir's `TTIRToTTNN.cpp` (the same fix described for `conv3d-patch-embed-l1-overflow` in Qwen3VL): reduce the static circular buffer allocation for `cb_vol2col_tiled` and `cb_weight_tiled` so that the combined size stays within 1,572,864 B for all kernel sizes used by vision patch embeddings.

## Tier B justification
Which indicator: **cross-cutting**

The Conv3d L1 overflow fix requires coordinated changes across at least three files in two repos (tt-mlir: `TTIRToTTNN.cpp`; tt-metal: `conv3d.cpp` / `conv3d_program_factory.cpp`, `prepare_conv3d_weights.cpp`) to change how circular buffer sizes are computed and allocated for the Conv3d kernel. This is the same Tier B bug as the Qwen3VL `conv3d-patch-embed-l1-overflow` failure.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~267s (to Conv3d failure)
- Tier A attempts: N/A

## Files changed
- `diffusionvl_qwen2_5_vl/pytorch/loader.py` (in tt-forge-models, remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 351a8269c38761a542a3fd0d9b16a720270fbe80 |
