# Remediation Summary: mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf-image_to_text-pytorch-8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf/image_to_text/pytorch-8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Conv3d L1 circular buffer overflow in Qwen3VL vision patch embedding (Tier B)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
conv3d-small-cin-padding-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to 2247168 B which is beyond max L1 size of 1572864 B
...
tt::runtime::ttnn::operations::conv::run(tt::target::ttnn::Conv3dOp const*, tt::runtime::ttnn::ProgramContext&)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Original remote failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

## Root cause
The original `ImportError` was caused by the missing `gguf>=0.10.0` package in `requirements.txt`. Four additional loader bugs were uncovered and fixed:

1. **GGUF architecture not registered**: `qwen3vl` (no underscore) is absent from `GGUF_SUPPORTED_ARCHITECTURES` and `get_gguf_hf_weights_map` uses `model_type="qwen3_vl"` (with underscore) which gguf-py doesn't recognize.

2. **Session contamination**: 26 qwen3_5 GGUF loaders monkey-patch `load_gguf_checkpoint` with a narrow signature `(gguf_path, return_tensors=False)` that drops the `model_to_load` kwarg added in transformers 5.2.0.

3. **Wrong model dimensions**: Loading with a bare `Qwen3VLConfig()` (default dimensions) produces shape mismatches; the correct 8B architecture requires `AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")`.

4. **D2H tolist() on TT device**: `fast_pos_embed_interpolate`, `rot_pos_emb`, `get_rope_index`, and `get_image_features` all call `.tolist()` on tensors resident on the TT XLA device, raising INTERNAL error 13.

After all loader fixes, the terminal failure is in `tt-metal`'s `Conv3dDeviceOperation`. `Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152, kernel=[2,16,16])`. The Conv3d lowering in tt-mlir pads `C_in=3` to `TILE_WIDTH=32`, causing the `cb_vol2col_tiled` + `cb_weight_tiled` circular buffers to total 2,247,168 B — exceeding the 1,572,864 B L1 maximum. This is independent of input size; the overflow is determined by kernel parameters alone.

## Fix
All loader-layer fixes were committed to `remediation/mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf-image_to_text-pytorch-8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF-single_device-inference` in tt-forge-models:

- `mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf/image_to_text/pytorch/requirements.txt` — added `gguf>=0.10.0`
- `mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf/image_to_text/pytorch/loader.py` — `_patch_qwen3vl_support()` to register qwen3vl arch + patch `get_gguf_hf_weights_map`; `_patch_qwen3vl_for_tt_device()` to move grid_thw metadata to CPU for the four tolist-calling methods; `AutoConfig.from_pretrained` for correct 8B dimensions; `ignore_mismatched_sizes=True`; pixel limits `min_pixels=56*56, max_pixels=13*28*1280`
- 26 qwen3_5 GGUF loaders — `_patched_load_gguf_checkpoint(*args, **kwargs)` to fix narrow-sig session contamination

The terminal Conv3d L1 overflow requires a fix in `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` `Conv3dOpConversionPattern::matchAndRewrite`: compute `maxCInBlock = MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k)` (where `MAX_CB_TILES=256`), cap `c_in_block = min(c_in_block, maxCInBlock)`, and set `compute_with_storage_grid_size` to the full worker grid so that the `C_in_blocks <= total_cores` assertion passes when `c_in_block < TILE_WIDTH`. This fix was previously developed on the tt-mlir remediation branch as commit `151e2000ce` but caused a secondary `CB page size < tensor page size` regression in tt-metal that prevented it from shipping.

## Tier B justification
**cross-repo**: Fixing `conv3d-small-cin-padding-l1-overflow` requires coordinated changes in both tt-mlir (`Conv3dOpConversionPattern::matchAndRewrite` to cap `c_in_block`) and tt-metal (`conv3d_op_sharded_program_factory` to keep the reader CB page size consistent with the reduced `c_in_block`). The secondary `conv3d-cb-page-size-vs-tensor-mismatch` bug was confirmed in a prior session and blocks any Tier A attempt on this model.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    271.40s (model loading + compilation; Conv3d L1 crash at runtime)
- Tier A attempts: 0

## Files changed
**tt-forge-models** (`remediation/mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf-image_to_text-pytorch-8B_INSTRUCT_ABLITERATED_Q4_K_M_GGUF-single_device-inference`):
- `mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf/image_to_text/pytorch/requirements.txt` (created)
- `mradermacher_huihui_qwen3_vl_8b_instruct_abliterated_gguf/image_to_text/pytorch/loader.py` (5 fixes)
- 26 × `*/image_to_text/pytorch/loader.py` or `*/causal_lm/pytorch/loader.py` (qwen3_5 narrow-sig fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eab49339a4a058031f778f60ff12fa2cac11d3b4 |
| tt-forge-models | 06c0f7877b40d6ca1f87adb81b281f9c4e5c2cff |
