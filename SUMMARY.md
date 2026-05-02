# Remediation Summary: mradermacher_robobrain_2_5_4b_gguf-image_to_text-pytorch-robobrain_2_5_4b_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_robobrain_2_5_4b_gguf/image_to_text/pytorch-robobrain_2_5_4b_gguf-single_device-inference]

## Result
FAIL — Conv3d VisionPatchEmbed L1 CB overflow in Qwen3VL visual encoder; Tier B compiler bug

## Stack layer
loader, tt-mlir

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
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Two separate issues:

**Loader bug (fixed):** The GGUF file stores `general.architecture = "qwen3vl"` (no underscore). `transformers` 5.x does not register this architecture in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING`, causing `load_gguf_checkpoint` to raise `ValueError: GGUF model with architecture qwen3vl is not supported yet.` Additionally, the session contains ~30 loaders that patch `load_gguf_checkpoint` with narrow signatures incompatible with the `model_to_load` kwarg added in transformers 5.2.0. The config also requires remapping from a flat qwen3vl dict to the nested `Qwen3VLConfig` structure (with `text_config` sub-dict and corrected `vision_config.out_hidden_size`).

**Compiler bug (terminal, Tier B):** `Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=~1152, kernel=[2,16,16], stride=[2,16,16])`. The tt-mlir Conv3d lowering pads C_in=3 to TILE_WIDTH=32, yielding `matmul_K_t = (2×16×16×32)/32 = 512 tiles`. `conv3d_program_factory.cpp` then allocates `cb_vol2col_tiled` (1MB) + `cb_weight_tiled` (1MB) = 2.1MB of circular buffers per core, exceeding the hardware L1 limit of 1.5MB. This causes `INTERNAL: Error code: 13` during graph compilation. A c_in_block capping fix exists (tt-mlir commit `151e2000ce`) but was not applied here because it exposes a secondary CB page-size mismatch (`conv3d-cb-page-size-vs-tensor-mismatch`) in tt-metal's Conv3d kernel reader.

## Fix
**Loader fix** (committed to `tt-forge-models` on `remediation/mradermacher_robobrain_2_5_4b_gguf-image_to_text-pytorch-robobrain_2_5_4b_gguf-single_device-inference`):

`mradermacher_robobrain_2_5_4b_gguf/image_to_text/pytorch/loader.py`:
1. `_register_qwen3vl_gguf_support()`: registers `qwen3vl` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`, and `GGUF_TO_FAST_CONVERTERS` at import time.
2. `_robobrain_load_gguf` wrapper installed around `gguf_utils.load_gguf_checkpoint` in `load_model()`: accepts `*args, **kwargs` (bypasses narrow-sig contamination); on config pass (return_tensors=False) delegates to chain; translates flat `qwen3vl` config dict into nested `Qwen3VLConfig` structure; on tensor pass loads tensors directly via `GGUFReader` + `_build_qwen3vl_gguf_tensor_mapping` bypassing chain entirely.
3. `_patch_qwen3vl_for_tt_device()`: patches four `Qwen3VLVisionModel`/`Qwen3VLModel` methods to move `grid_thw`/`input_ids`/`attention_mask` tensors to CPU before `.tolist()` calls. Reimplements `fast_pos_embed_interpolate` on CPU using a captured weight copy and transfers result back via `xm.send_cpu_data_to_device`.
4. Pixel limits set on processor (`min_pixels=56*56`, `max_pixels=13*28*1280`) to prevent 11K+ patches from oversized images.

**Terminal compiler bug fix** (proposed, not implemented — Tier B):
In `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`, `Conv3dOpConversionPattern::matchAndRewrite`: cap `c_in_block = min(c_in_block, MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k))` where `MAX_CB_TILES=256`. For kernel [2,16,16]: `c_in_block ≤ 16`. Also set compute_with_storage_grid_size to full device grid when c_in_block < TILE_WIDTH. This fix exists as commit `151e2000ce` but triggers a secondary bug (`conv3d-cb-page-size-vs-tensor-mismatch`) in `tt-metal/ttnn/cpp/ttnn/operations/conv/conv3d/conv3d_op_sharded_program_factory_common.cpp` where the CB page size (192 B for c_in_block=16) is less than the tensor page size (392 B = 2×14×14 for kernel shape). Both bugs must be fixed together.

## Tier B justification
`cross-cutting`: The Conv3d L1 fix (tt-mlir) exposes a secondary CB page-size mismatch bug (tt-metal) — a correct fix requires coordinated changes across two files in two repos. The prior attempt (commit `151e2000ce` in tt-mlir) confirmed this cross-repo dependency. Per the one-Tier-A-fix-per-report rule, the fix was not attempted here.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    269.16s (0:04:29)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: `mradermacher_robobrain_2_5_4b_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7c39e8f9590061649df87daa56a72d0f1d0881b5 |
| tt-forge-models | c14faa247f820733fa61221e659631a060eed060 |
