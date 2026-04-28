# Remediation Summary: gelab_zero_4b_preview_gguf-image_to_text-pytorch-4B_PREVIEW_Q4_K_M_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gelab_zero_4b_preview_gguf/image_to_text/pytorch-4B_PREVIEW_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Conv3d patch embedding OOMs L1 (2.2 MB > 1.5 MB) on every Qwen3VL forward pass; Tier B compiler-stack bug in tt-metal Conv3d kernel

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

Traceback (excerpt):
```
transformers/models/qwen3_vl/modeling_qwen3_vl.py:778: in forward
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
third_party/tt_forge_models/gelab_zero_4b_preview_gguf/image_to_text/pytorch/loader.py:97: in _fast_pos
    def _fast_pos(self, grid_thw):
torch_xla/_dynamo/dynamo_bridge.py:826: in extract_compiled_graph_helper
    torch_xla.sync(reset_scope=False)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=~1152, kernel=[2,16,16], stride=[2,16,16])` to embed visual patches. The `ttnn.experimental.Conv3dDeviceOperation` kernel statically allocates circular buffers based only on kernel/channel parameters, not on the number of input patches:

- `C_in=3` is padded to `TILE_WIDTH=32` in `TTIRToTTNN.cpp`, making `matmul_K_t = (2×16×16×32)/32 = 512 tiles`
- `conv3d_program_factory.cpp` allocates `cb_vol2col_tiled` (1×512×2048B = 1 MB) and `cb_weight_tiled` (512×1×2048B = 1 MB)
- Total: ~2.2 MB, exceeding the L1 maximum of 1,572,864 B (~1.5 MB) on every forward pass
- This is independent of batch size, sequence length, or the number of patches — the overflow is determined entirely by the Conv3d kernel geometry

The error surfaces at `torch_xla.sync()` during graph compilation of `fast_pos_embed_interpolate`, which triggers the full vision encoder forward (including Conv3d) to be compiled and executed on TT silicon.

## Fix
Three loader-layer bugs were fixed in `tt_forge_models` (all committed to remediation branch):

1. **GGUF architecture registration** (`gelab_zero_4b_preview_gguf/image_to_text/pytorch/loader.py`):
   - `qwen3vl` GGUF arch string not in `GGUF_SUPPORTED_ARCHITECTURES` → added via `_patch_qwen3vl_gguf_support()`
   - `get_gguf_hf_weights_map` tried to read `config.num_hidden_layers` directly on `Qwen3VLConfig` (which nests it under `text_config`) → patched to extract from `text_config`
   - Required handling `model_type="qwen3_vl"` passed by upstream monkey-patches (e.g. momix_44 loader)
   - Added `ignore_mismatched_sizes=True` because gguf-py's qwen3vl tensor name map incorrectly routes a [2560] text-model tensor to `model.visual.merger.norm.weight` (expects [1024])

2. **`model_to_load` kwarg rejection** (26 GGUF loader files):
   - transformers 5.2.0 added `model_to_load` kwarg to `load_gguf_checkpoint`; 26 loaders had narrow `(gguf_path, return_tensors=False)` signatures in their monkey-patches, causing `TypeError`
   - Fixed by changing to `(*args, **kwargs)` pass-through

3. **`.tolist()` on TT device tensors** (`gelab_zero_4b_preview_gguf/image_to_text/pytorch/loader.py`):
   - Four Qwen3VL methods call `.tolist()` on `grid_thw` / `input_ids` tensors that may be on the TT device
   - Patched `fast_pos_embed_interpolate`, `rot_pos_emb`, `get_image_features`, and `get_rope_index` to move metadata tensors to CPU before the calls
   - Also set `processor.image_processor.min_pixels = 56*56` and `max_pixels = 13*28*1280` to prevent the 1376×2048 demo image from generating 11K+ visual patches

After all loader fixes, the test advances to Conv3d execution in the vision encoder and fails with Error code: 13. This is the Tier B bug — it requires coordinated changes across `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`, `tt-metal/ttnn/cpp/ttnn/operations/conv/conv3d/conv3d_program_factory.cpp`, and `tt-metal/ttnn/cpp/ttnn/operations/conv/conv3d/prepare_conv3d_weights.cpp` to reduce the static circular buffer allocation below L1 capacity.

Proposed fix (not implemented): In `TTIRToTTNN.cpp`, avoid over-padding `C_in` to `TILE_WIDTH` when the true `matmul_K = C_in × T × H × W` already aligns to a tile; alternatively, in `conv3d_program_factory.cpp`, split the `cb_vol2col_tiled` allocation across fewer cores or use a streaming approach to keep per-core CB usage under L1 limit.

## Tier B justification
`cross-cutting` — The fix requires coordinated changes across at least three files spanning two repos (tt-mlir lowering + tt-metal kernel factory), and the root cause (static CB over-allocation in Conv3d) affects every Conv3d model on TT hardware, not just this test.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    3m 12s (wall clock to Conv3d OOM)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gelab_zero_4b_preview_gguf/image_to_text/pytorch/loader.py` — GGUF arch registration, tolist() patches, pixel limits, ignore_mismatched_sizes
- 26 GGUF loader files (narrow `_patched_load_gguf_checkpoint` signature fixed to `*args, **kwargs`)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3e918925d7517727728fc7b7d5977f8be7564d50 |
| tt-forge-models | 6fee3a6717fc517d3da7c3804439f6ff39937c83 |
