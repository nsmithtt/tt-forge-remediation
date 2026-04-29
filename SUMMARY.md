# Remediation Summary: chandra_ocr_gguf/image_to_text/pytorch-Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chandra_ocr_gguf/image_to_text/pytorch-Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — `get_placeholder_mask` uses `inputs_embeds[bool_mask].numel()` inside `torch_compilable_check`; Python evaluates the argument before `TRANSFORMERS_DISABLE_TORCH_CHECK` short-circuits the call, compiling a dynamic-shape boolean-index node into the XLA graph that TT rejects with Error code 13. Tier B new-infrastructure bug.

## Stack layer
loader, tt-mlir

## Tier
A (Conv3d L1 overflow), then B (dynamic-shape boolean index after Conv3d fix)

## Bug fingerprint
dynamic-shape-boolean-index

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

Stack root:
```
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1225: in torch_dynamo_resume_in_forward_at_1219
    image_mask, _ = self.get_placeholder_mask(
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1137: in get_placeholder_mask
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
transformers/models/qwen3_vl/modeling_qwen3_vl.py:1137: in torch_dynamo_resume_in_get_placeholder_mask_at_1137
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
torch_xla/_dynamo/dynamo_bridge.py:346: in extract_graph_helper
    torch_xla.sync(reset_scope=False)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Original failure (before fixes): `ValueError: GGUF model with architecture qwen3vl is not supported yet.`

## Root cause
**Four issues found: three loader bugs and one Tier A compiler fix (both applied), and one Tier B compiler bug (unfixed).**

### Loader bug 1: `qwen3vl` GGUF architecture not registered (FIXED)
The Chandra OCR GGUF file stores `general.architecture = "qwen3vl"` (no underscore).
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` does not have `qwen3vl`
in `GGUF_SUPPORTED_ARCHITECTURES`, raising `ValueError: GGUF model with architecture
qwen3vl is not supported yet.`. Additionally, `get_gguf_hf_weights_map` uses
`hf_model.config.model_type = "qwen3_vl"` (with underscore) while gguf-py expects
`"qwen3vl"`, breaking the weight-name mapping.

Fix: add `qwen3vl` to `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`, patch
`get_gguf_hf_weights_map` to remap `model_type="qwen3_vl"` → `"qwen3vl"`, and install
a `load_gguf_checkpoint` wrapper that BFS-walks the monkey-patch chain to find the real
function (bypassing broken fixed-signature wrappers from other loaders that drop the
`model_to_load` kwarg added in transformers 5.2.0).

**Layer: loader (tt-forge-models)**

### Loader bug 2: `.tolist()` on TT device tensors + pixel limits (FIXED)
`Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb` call `.tolist()` on
`grid_thw` for Python control flow. `Qwen3VLModel.get_rope_index` calls `.tolist()` on
`input_ids`, `image_grid_thw`, and `video_grid_thw`. `Qwen3VLModel.get_image_features`
calls `(image_grid_thw.prod(-1)).tolist()`. TT device does not support eager tensor
readback — any `.tolist()` on a TT tensor triggers a device sync that fails with Error
code 13. These four methods are patched with `@torch.compiler.disable` to prevent XLA
from tracing the `.cpu()` moves for these metadata operations into the compiled graph.

Additionally, the processor was initialized without `min_pixels`/`max_pixels`, which
would produce far more patches than the hardware L1 budget allows.

**Layer: loader (tt-forge-models)**

### Loader bug 3: `output_norm` weight mapped to wrong layer (FIXED)
`Qwen3VLModel` registers `visual` before `language_model` in `_modules`. The generic
GGUF weight-name mapping applies `output_norm` → the first module found with `norm`,
which is `visual.merger.norm` (size 1152) instead of `language_model.norm` (size 4096),
causing a size-mismatch error during weight loading. Fix: in the `load_gguf_checkpoint`
compat wrapper, reorder `_modules` so `visual` comes after `language_model` when both
are present, giving `language_model` priority for the `norm`→`output_norm` mapping.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug 1: Conv3d L1 overflow (FIXED — Tier A)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152,
kernel_size=[2,16,16], stride=[2,16,16])`. With `c_in_block = TILE_WIDTH = 32`, the
two main circular buffers `cb_vol2col_tiled` and `cb_weight_tiled` each consume 512
tiles (= 1 MB at 2048 B/tile), totalling 2 MB > L1 max (~1.5 MB per core), causing
`TT_THROW: Statically allocated circular buffers grow to 2247168 B which is beyond max
L1 size of 1572864 B`.

Fix: in `Conv3dOpConversionPattern::matchAndRewrite` in `TTIRToTTNN.cpp`, compute
`c_in_block` dynamically so that `matmul_K_t ≤ MAX_CB_TILES = 256` tiles (512 KB per
CB). The full device worker grid is passed as `compute_with_storage_grid_size` to satisfy
the runtime's `C_in_blocks ≤ num_cores` assertion. An explicit `Conv3dConfigAttr` is
attached so the runtime uses the same `c_in_block` as the pre-blocked weight tensor.

**Layer: tt-mlir**

### Compiler-stack bug 2: dynamic-shape boolean index in `get_placeholder_mask` (UNFIXED — Tier B)
After the Conv3d fix, `Qwen3VLModel.forward` compiles further and reaches
`get_placeholder_mask` at line 1225. Inside that function, `torch_compilable_check` is
called with `inputs_embeds[special_image_mask].numel() == image_features.numel()` as the
condition. Python evaluates the argument **before** calling `torch_compilable_check`, so
the boolean-indexed expression `inputs_embeds[special_image_mask]` is compiled into the
XLA graph regardless of `TRANSFORMERS_DISABLE_TORCH_CHECK=1`. The boolean index produces
a dynamic-shape output tensor (size depends on how many True values are in the mask),
which the TT XLA backend cannot compile; it rejects the graph with Error code 13.

This is the same `dynamic-shape-boolean-index` class of bug observed in Qwen3.5's
`get_placeholder_mask`. There is no Python-level workaround: the only ways to prevent
the boolean indexing from being traced are (a) patching/disabling the check (forbidden
— assertion suppression) or (b) implementing dynamic-shape support in the TT XLA
backend.

**Layer: tt-xla (PJRT / XLA compilation)**

## Fix
### Applied (loader, tt-forge-models — commits `4e5d47823f`, `7f79ecb843`, `06e0996e45`):
1. Register `qwen3vl` GGUF architecture in `GGUF_CONFIG_MAPPING`, `GGUF_SUPPORTED_ARCHITECTURES`, and `get_gguf_hf_weights_map`.
2. Install a compat `load_gguf_checkpoint` wrapper that walks the monkey-patch chain and reorders `_modules` to fix weight mapping.
3. Patch four Qwen3VL methods with `@torch.compiler.disable` to prevent `.tolist()` D2H failures. Add pixel limits.
4. Fix `output_norm` weight mapping by reordering `_modules`.

### Applied (tt-mlir — commit `8d43a8a60`):
In `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`, `Conv3dOpConversionPattern::matchAndRewrite`:
- Compute `maxCInBlock = MAX_CB_TILES * TILE_WIDTH / kernelElements` and halve `cInBlock` until `matmul_K_t ≤ MAX_CB_TILES`.
- Attach `Conv3dConfigAttr` with `c_in_block`, `c_out_block=TILE_WIDTH`, and the full device worker grid.

### Proposed (tt-xla, dynamic-shape boolean index):
Implement support for dynamic-shape boolean-indexed tensors in the TT XLA backend's
StableHLO lowering. The `inputs_embeds[bool_mask]` pattern requires either:
1. A dynamic-gather lowering that pads the output to a statically-known maximum size, OR
2. Moving the entire `get_placeholder_mask` check outside the compiled graph by wrapping
   `torch_compilable_check` in a way that Dynamo treats the condition as lazy (i.e.,
   changing the call site to `torch_compilable_check(lambda: ..., ...)` upstream in
   transformers, or adding a Dynamo-level hook that skips argument evaluation when
   `TRANSFORMERS_DISABLE_TORCH_CHECK=1`).

## Tier B justification
**Indicator: new-infrastructure**

The boolean-indexed expression `inputs_embeds[special_image_mask]` produces a tensor
whose leading dimension equals the number of True values in the mask at runtime. XLA
requires all tensor shapes to be known at compile time; supporting dynamic-shape gathers
from boolean masks requires new infrastructure in the TT XLA lowering (dynamic gather
ops, shape inference for variable-length outputs, or a speculative execution scheme).
This is not a localized bug fix — it requires cross-cutting changes to how the compiler
handles data-dependent shapes.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    323.47s (0:05:23)
- Tier A attempts: 1 (Conv3d c_in_block fix — resolved the patch-embed L1 overflow; unmasked the boolean-index bug)

## Files changed
**tt-mlir** (`remediation/chandra_ocr_gguf-image_to_text-pytorch-Q4_K_M_GGUF-single_device-inference`):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**tt-forge-models** (`remediation/chandra_ocr_gguf-image_to_text-pytorch-Q4_K_M_GGUF-single_device-inference`):
- `chandra_ocr_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 8d43a8a6027a1ac1c860acc30bb9b7e592566d7c |
| tt-xla          | 64395845eb4bb67513e5a132364bf35e4becd83f |
| tt-forge-models | 06e0996e451f068ce28a499f1ba40c88287aa947 |
