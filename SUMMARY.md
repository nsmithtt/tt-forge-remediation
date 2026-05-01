# Remediation Summary: megamind_v2_vl_med_i1_gguf-image_to_text-pytorch-v2_vl_med_i1_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[megamind_v2_vl_med_i1_gguf/image_to_text/pytorch-v2_vl_med_i1_gguf-single_device-inference]

## Result
FAIL — Conv3d patch embedding L1 CB overflow (Tier B: secondary CB page size mismatch blocks the Tier A fix)

## Stack layer
tt-mlir, tt-metal

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

Stack trace ends at:
    torch_xla.sync(reset_scope=False) inside extract_compiled_graph_helper,
called when compiling the first subgraph of Qwen3VLVisionModel.forward (which
includes the Conv3d patch embedding via the @torch.compiler.disable graph break
on fast_pos_embed_interpolate).

## Root cause
Three loader bugs were fixed in sequence before reaching the final compiler-stack
failure:

1. **qwen3vl GGUF arch not registered** — transformers 5.x has no entry for
   `"qwen3vl"` in `GGUF_SUPPORTED_ARCHITECTURES` or `GGUF_TO_TRANSFORMERS_MAPPING`.
   Fixed by patching both at import time.

2. **model_to_load kwarg incompatibility** — Other GGUF loaders installed
   narrow-signature patches of `load_gguf_checkpoint` that dropped the
   `model_to_load` kwarg added in transformers 5.2.0. These patches intercept
   `Qwen3VLForConditionalGeneration.from_pretrained(gguf_file=...)`.
   Fixed by bypassing `from_pretrained` entirely: instantiate the model directly
   from the base-model config, download the GGUF with `huggingface_hub.hf_hub_download`,
   and load tensors via raw `gguf.GGUFReader`/`dequantize`.

3. **Visual sub-module key stealing in get_gguf_hf_weights_map** —
   `_orig_get_gguf_hf_weights_map` traverses all model children recursively
   including vision encoder sub-modules. The qwen3vl name_map matched
   `merger.norm` → `output_norm`, which claimed the `output_norm.weight` GGUF
   key before the language model norm could map it. Fixed by building the full
   map, stripping visual entries, then re-adding language-model entries with a
   direct call scoped to `hf_model.model.language_model`.

4. **tolist() on TT device** — `Qwen3VLVisionModel.fast_pos_embed_interpolate`,
   `rot_pos_emb`, `Qwen3VLModel.get_rope_index`, and `get_image_features` all
   call `.tolist()` on tensors on the TT device. TT device does not support
   eager readback. Fixed by patching all four methods: move metadata tensors
   (grid_thw, input_ids, attention_mask) to CPU for the control-flow computation;
   reimplement `fast_pos_embed_interpolate` entirely on CPU using a pre-captured
   `pos_embed.weight`; transfer the result back via `xm.send_cpu_data_to_device`.

After all loader bugs are fixed, the test fails during the first XLA graph sync
(inside `extract_compiled_graph`) because `Qwen3VLVisionPatchEmbed` uses
`nn.Conv3d(in_channels=3, out_channels=1152, kernel=[2,16,16], stride=[2,16,16])`.
With `cInBlock = TILE_WIDTH = 32` in tt-mlir's `Conv3dOpConversionPattern`, the
tt-metal `conv3d_program_factory` allocates:
  - `cb_vol2col_tiled` = 512 kernel tiles × 1 c_in_block × 2048 B ≈ 1 MB
  - `cb_weight_tiled`  = 512 kernel tiles × 1 c_in_block × 2048 B ≈ 1 MB
  - Total ≈ 2 MB  >  L1 max (1.5 MB)

This allocation is kernel-parameter-driven and independent of batch size or
patch count (pixel limits do not help). The device returns INTERNAL: Error
code: 13 (OOM).

## Fix
**Loader fixes (tt-forge-models remediation branch):**
- `megamind_v2_vl_med_i1_gguf/image_to_text/pytorch/loader.py`
  - `_patch_qwen3vl_gguf_support()`: register `"qwen3vl"` in
    `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`.
  - `_patched_get_gguf_hf_weights_map()`: remap `model_type "qwen3_vl" →
    "qwen3vl"`, extract `num_hidden_layers` from nested `text_config`, strip
    visual entries from the full weight map, re-add language-model mappings.
  - `_load_tensors_from_gguf()`: direct GGUF loading via `GGUFReader`/
    `dequantize` to bypass the patched `load_gguf_checkpoint` chain.
  - `_patch_qwen3vl_for_tt_device()`: four-method `.tolist()` patch;
    `fast_pos_embed_interpolate` reimplemented on CPU with pre-captured
    `pos_embed.weight` + `xm.send_cpu_data_to_device` transfer.
  - Pixel limits: `min_pixels=56*56`, `max_pixels=13*28*1280` on
    `processor.image_processor` to cap patch count.

**Compiler-stack fix (proposed, not implemented):**
The Conv3d L1 overflow requires capping `c_in_block` in
`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
`Conv3dOpConversionPattern::matchAndRewrite`. Compute:
```
maxCInBlock = MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k)
```
where `MAX_CB_TILES = 256` (512 KB / 2048 B per tile). For kernel=[2,16,16]:
`maxCInBlock = 8192 / 512 = 16`. Use `min(TILE_WIDTH, maxCInBlock)` as
`cInBlock` for weight preparation.

A previous Tier A attempt (tt-mlir commit `151e2000ce`) confirmed this resolves
the L1 overflow for Qwen3VL. However, a subsequent investigation of GLM-OCR
(kernel=[2,14,14]) revealed a secondary tt-metal bug: `TT_FATAL: CB page size
192 should be greater than the config tensor page size 392`. Until this secondary
CB page size mismatch is resolved, the Conv3d path cannot be fixed with a single
scoped Tier A change.

## Tier B justification
cross-cutting — The primary fix (c_in_block capping in `TTIRToTTNN.cpp`) exposed a
secondary `TT_FATAL: CB page size < kernel tensor page size` bug in the tt-metal
Conv3d kernel (`conv3d_op_program_factory_common.cpp`). Resolving the root cause
requires coordinated changes across TTIRToTTNN.cpp, the tt-metal Conv3d program
factory, and `prepare_conv3d_weights.cpp`. The "Remove workarounds for conv3d"
commit (c5a5100dc) removed the c_in_block capping and a subsequent uplift
(080ad1295) set `cInBlock = TILE_WIDTH`, reinstating the overflow. A complete fix
is not scoped to a single file.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    457.48s (0:07:37)
- Tier A attempts: 0

## Files changed
- `tt-xla/third_party/tt_forge_models/megamind_v2_vl_med_i1_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 2f91f992f001eb8ea6a311ff1db0ebd44868732f |
