# Remediation Summary: egoactor_4b_qwen3vl_i1_gguf/image_to_text/pytorch-4b_qwen3vl_i1_gguf-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[egoactor_4b_qwen3vl_i1_gguf/image_to_text/pytorch-4b_qwen3vl_i1_gguf-single_device-inference]

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
transformers/models/qwen3_vl/modeling_qwen3_vl.py:778: in torch_dynamo_resume_in_forward_at_778
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
torch/_dynamo/eval_frame.py:1044: in _fn
    return fn(*args, **kwargs)
tt_torch/backend/backend.py:225: in __call__
    return self._call_experimental_compile(*args)
torch_xla/_dynamo/dynamo_bridge.py:826: in extract_compiled_graph_helper
    torch_xla.sync(reset_scope=False)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Underlying device error (from prior runs of the same architecture on n150):
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)]
grow to 2247168 B which is beyond max L1 size of 1572864 B
tt::runtime::ttnn::operations::conv::run(tt::target::ttnn::Conv3dOp const*, ...)
```

## Root cause
**Two issues found: three loader bugs (fixed) and one compiler-stack bug (unfixed).**

### Loader bug 1: `qwen3vl` GGUF architecture not registered (FIXED)
The mradermacher EgoActor GGUF file stores `general.architecture = "qwen3vl"` (no
underscore). `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` only
knows `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe` — `qwen3vl` is absent from
`GGUF_SUPPORTED_ARCHITECTURES`, raising `ValueError: GGUF model with architecture
qwen3vl is not supported yet.` Also, `get_gguf_hf_weights_map` uses
`hf_model.config.model_type = "qwen3_vl"` (with underscore) while gguf-py's
`MODEL_ARCH_NAMES` expects `"qwen3vl"`, breaking the weight-name mapping.

**Layer: loader (tt-forge-models)**

### Loader bug 2: `model_to_load` kwarg dropped by broken GGUF wrappers (FIXED)
Several loaders import-time patch `load_gguf_checkpoint` with a narrow fixed
signature `(gguf_path, return_tensors=False)`, dropping the `model_to_load` kwarg
added in transformers 5.2.0. The original egoactor_4b loader called the unpatched
function but the session-global wrapper installed by other loaders would have
dropped the kwarg on the weight-loading path. A BFS walk of the monkey-patch chain
finds the real function and installs a compat wrapper that routes `model_to_load`
to the real function.

**Layer: loader (tt-forge-models)**

### Loader bug 3: `.tolist()` on TT device tensors + pixel limits (FIXED)
`Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb` call
`grid_thw.tolist()` for Python control-flow. `Qwen3VLModel.get_rope_index`
calls `.tolist()` on `input_ids`, `image_grid_thw`, and `video_grid_thw`.
`Qwen3VLModel.get_image_features` calls `(image_grid_thw.prod(-1)).tolist()`.
TT device does not support eager tensor readback — any `.tolist()` on a TT
tensor triggers a device sync that fails with Error code: 13.

Additionally, the processor was initialized without `min_pixels`/`max_pixels`,
so the demo.jpeg (≈1376×2048) produced 11,008 patches instead of ~1,768.
Standard pixel limits of `min_pixels=56*56, max_pixels=13*28*1280` are required.

**Layer: loader (tt-forge-models)**

### Compiler-stack bug: Conv3d L1 overflow (UNFIXED, Tier B)
`Qwen3VLVisionPatchEmbed` uses `nn.Conv3d(in_channels=3, out_channels=1152,
kernel_size=[2,16,16], stride=[2,16,16])`. `ttnn.experimental.Conv3dDeviceOperation`
statically allocates `cb_vol2col_tiled` (1×512×2048B=1MB) and `cb_weight_tiled`
(512×1×2048B=1MB), totaling ~2.24MB > L1 max (1.57MB) on the full 11×10 core
grid. This allocation is determined by kernel parameters (in_channels padded to
TILE_WIDTH=32, matmul_K_t=512), not batch size — 1,768 patches and 11,008 patches
both trigger the same overflow.

The error surfaces when Dynamo flushes the accumulated XLA graph before calling
the `@torch.compiler.disable`-decorated `fast_pos_embed_interpolate` patch at
line 778. The graph being compiled at that point contains the Conv3d patch
embedding from `Qwen3VLVisionPatchEmbed`.

**Layer: tt-metal (runtime)**

## Fix
### Applied (loader, tt-forge-models):

1. **Register `qwen3vl` GGUF architecture:** Add `qwen3vl` key to
   `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`. Patch
   `get_gguf_hf_weights_map` to remap `model_type="qwen3_vl"` → `"qwen3vl"` and
   extract `num_hidden_layers` from `text_config` (where Qwen3VLConfig nests it).
   Install a compat `load_gguf_checkpoint` wrapper that BFS-walks the monkey-patch
   chain to find the real function and routes `model_to_load` to it.

2. **Patch `.tolist()` callers:** Monkey-patch four `Qwen3VL` methods to move
   metadata tensors to CPU before `.tolist()` calls:
   - `Qwen3VLVisionModel.fast_pos_embed_interpolate(grid_thw)` → reimplemented on
     CPU using pre-captured `pos_embed.weight`; result sent back to TT device via
     `xm.send_cpu_data_to_device`
   - `Qwen3VLVisionModel.rot_pos_emb(grid_thw)` → `grid_thw.cpu()`
   - `Qwen3VLModel.get_rope_index(input_ids, image_grid_thw, ...)` → all args
     `.cpu()`, `position_ids`/`rope_deltas` moved back to original device
   - `Qwen3VLModel.get_image_features(pixel_values, image_grid_thw)` →
     `image_grid_thw.cpu()`

3. **Add pixel limits:** Set `processor.image_processor.min_pixels = 56*56` and
   `processor.image_processor.max_pixels = 13*28*1280` after processor load.

4. **Use base model config:** Load `Qwen3VLConfig.from_pretrained(BASE_MODEL)` and
   pass `config=config, ignore_mismatched_sizes=True` to `from_pretrained` so the
   nested `text_config`/`vision_config` structure is preserved and the GGUF weight
   mapping does not misconfigure the model architecture.

### Proposed (compiler-stack, tt-metal):
In `tt-metal`'s `Conv3dDeviceOperation` / `conv3d_program_factory.cpp`:
1. Add a pre-flight check that the static circular buffer allocation fits within
   L1, and fail fast with a descriptive error when it does not.
2. Implement a tiled/streaming strategy for Conv3d when the combined
   `cb_vol2col_tiled + cb_weight_tiled` footprint exceeds L1 capacity. The
   key quantity is `matmul_K_t = ceil(kernel_t * kernel_h * kernel_w * in_channels
   / TILE_WIDTH) * TILE_WIDTH / TILE_WIDTH` — for `in_channels=3` padded to 32,
   this is 512 tiles → 2×512×2048B = 2MB.

## Tier B justification
**Indicator: cross-cutting**

Fixing the Conv3d L1 overflow requires coordinated changes across at least three
files: `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` (where `C_in=3` is
padded to `TILE_WIDTH=32` producing the 512-tile kernel), `tt-metal/ttnn/cpp/ttnn/
operations/transformer/conv/device/conv3d_program_factory.cpp` (where the static
CB allocation is computed), and `prepare_conv3d_weights.cpp`. A correct streaming
or split-kernel strategy would also need coordinated changes in the host-side
prepare path.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    213.70s (0:03:33) before failing at Conv3d compile
- Tier A attempts: N/A

## Files changed
- `egoactor_4b_qwen3vl_i1_gguf/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c34729d4f5a04cbd5c6c75e00c2962b642619997 |
| tt-forge-models | 73a1977da77735e7b356e06c289cb290def4b7a9 |
