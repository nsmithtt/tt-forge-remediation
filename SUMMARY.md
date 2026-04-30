# Remediation Summary: huihui_qwen3_vl_4b_abliterated_gguf/image_to_text/pytorch-4b_instruct_abliterated_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[huihui_qwen3_vl_4b_abliterated_gguf/image_to_text/pytorch-4b_instruct_abliterated_gguf-single_device-inference]

## Result
FAIL — `ttnn.experimental.Conv3dDeviceOperation` statically allocates circular buffers that exceed L1 (2,247,168 B > L1 max 1,572,864 B) regardless of input size. This is a Tier B compiler/runtime bug in tt-metal's Conv3d kernel.

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
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Traceback root (after loader fixes applied):
```
transformers/models/qwen3_vl/modeling_qwen3_vl.py:782: in torch_dynamo_resume_in_forward_at_782
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
torch/_dynamo/eval_frame.py:1044: in _fn
    return fn(*args, **kwargs)
tt_torch/backend/backend.py:225: in __call__
    return self._call_experimental_compile(*args)
torch_xla/_dynamo/dynamo_bridge.py:826: in extract_compiled_graph_helper
    torch_xla.sync(reset_scope=False)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

The Error code: 13 is a tt-metal circular-buffer overflow. In the graph compiled after
the `@torch.compiler.disable` graph break from `_patched_fast_pos`, the continuation
graph contains the VisionModel forward including `patch_embed` (a Conv3d). When that
Conv3d is lowered, `ttnn.experimental.Conv3dDeviceOperation` statically allocates
2,247,168 B of circular buffers on the full 11×10 core grid, exceeding L1 max
(1,572,864 B).

## Root cause
**Two issues found: three loader bugs (fixed) and one compiler-stack bug (unfixed).**

### Loader bug 1: `qwen3vl` GGUF architecture not registered (FIXED)
The noctrex Huihui-Qwen3-VL-4B GGUF file stores `general.architecture = "qwen3vl"` (no
underscore). `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` only knows
`qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe` — `qwen3vl` is absent from
`GGUF_SUPPORTED_ARCHITECTURES`, causing an immediate `ValueError`.

### Loader bug 2: `get_gguf_hf_weights_map` model_type mismatch (FIXED)
`Qwen3VLConfig.model_type = "qwen3_vl"` (with underscore) but gguf-py's
`MODEL_ARCH_NAMES` uses `"qwen3vl"` (no underscore). The weight-map builder fails
to find the architecture, producing wrong or empty weight mappings.

### Loader bug 3: Broken `load_gguf_checkpoint` monkey-patch chain (FIXED)
Other loaders in the test suite wrap `load_gguf_checkpoint` with fixed-signature
wrappers that drop the `model_to_load` kwarg added in transformers 5.2.0. A BFS
walk installs a properly-signed top wrapper that routes `model_to_load` to the
real function, bypassing the broken chain.

### Compiler-stack bug: Conv3d L1 overflow (UNFIXED — Tier B)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152,
kernel=[2,16,16], stride=[2,16,16])`. The tt-mlir TTIRToTTNN lowering pads
`C_in=3` to `TILE_WIDTH=32`, giving `matmul_K_t = (2×16×16×32)/32 = 512` tiles.
`conv3d_program_factory.cpp` then allocates `cb_vol2col_tiled` (1×512×2048 B = 1 MB)
and `cb_weight_tiled` (512×1×2048 B = 1 MB), totalling 2,247,168 B > 1,572,864 B L1.
This allocation is determined by kernel parameters alone, not batch size — even with
pixel limits (max_pixels = 13×28×1280 ≈ 1768 patches) the overflow persists.

## Fix
Loader fixes applied in `tt-forge-models` remediation branch
`remediation/huihui_qwen3_vl_4b_abliterated_gguf-image_to_text-pytorch-4b_instruct_abliterated_gguf-single_device-inference`
(commit `7ab35c1b4e023203f9bc5e888c083c21bf4f72725`):

1. Added `_register_qwen3vl_gguf_architecture()`:
   - Adds `"qwen3vl"` entry to `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`
   - Patches `get_gguf_hf_weights_map` to remap `model_type "qwen3_vl" → "qwen3vl"` and
     extract `num_hidden_layers` from `text_config`
   - BFS-walks the monkey-patch chain to find the real `load_gguf_checkpoint` with
     `model_to_load`, installs a compat wrapper

2. Added `_patch_qwen3vl_for_tt_device(model=model)`:
   - Pre-captures `pos_embed.weight` on CPU before model moves to TT device
   - Reimplements `fast_pos_embed_interpolate` entirely on CPU with `@torch.compiler.disable`
     using `xm.send_cpu_data_to_device` for the result transfer
   - Patches `rot_pos_emb`, `get_rope_index`, `get_image_features` to move metadata tensors
     to CPU for `.tolist()` control flow

3. Model loading changes:
   - Uses `Qwen3VLConfig.from_pretrained(BASE_MODEL)` + `ignore_mismatched_sizes=True`
   - Sets pixel limits: `min_pixels=56*56`, `max_pixels=13*28*1280`
   - Sets `torch_dtype=torch.bfloat16` by default

**File changed:** `huihui_qwen3_vl_4b_abliterated_gguf/image_to_text/pytorch/loader.py`

**Proposed compiler fix** (Tier B — not attempted): In
`tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
`Conv3dOpConversionPattern::matchAndRewrite`, compute
`maxCInBlock = MAX_CB_TILES * TILE_WIDTH / (D_k * H_k * W_k)` and cap
`c_in_block = min(c_in_block, maxCInBlock)`. For kernel `[2,16,16]`: maxCInBlock = 16,
reducing total CB allocation to within 1.5 MB L1. This requires a coordinated fix in
`tt-metal/ttnn/cpp/ttnn/operations/conv/conv3d/conv3d_program_factory.cpp` and
`tt-metal/ttnn/cpp/ttnn/operations/conv/prepare_conv_weights.cpp` to handle the
non-default `c_in_block` path.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
more-than-3-files

The Conv3d L1 overflow fix requires coordinated changes across at least three files:
`TTIRToTTNN.cpp` (set c_in_block cap), `conv3d_program_factory.cpp` (handle reduced
c_in_block in CB allocation), and `prepare_conv3d_weights.cpp` (handle weight layout
for non-TILE_WIDTH c_in_block). A previous attempt (commit `151e2000ce` in tt-mlir)
exposed a secondary tt-metal bug (`conv3d-cb-page-size-vs-tensor-mismatch`: CB page
size 192 < kernel tensor page size 392) that requires a further cross-repo fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    181.62s (0:03:01)
- Tier A attempts: N/A

## Files changed
- `huihui_qwen3_vl_4b_abliterated_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | fbea833c4bb4b00aa4e14e73e7e11f5a82ccc9ef |
| tt-forge-models | 7ab35c1b4e023203f9bc5e888c083c21bf4f72725 |
