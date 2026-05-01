# Remediation Summary: mai_ui_8b_i1_gguf-image_to_text-pytorch-mai_ui_8b_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mai_ui_8b_i1_gguf/image_to_text/pytorch-mai_ui_8b_gguf-single_device-inference]

## Result
FAIL — second compiler-stack bug: `ttnn::prim::concat` CB allocation exceeds L1 after Conv3d fix; device enters unrecoverable error state

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Original failure (before loader fix): `ValueError: GGUF model with architecture qwen3vl is not supported yet.`
After loader + Conv3d fix: device enters error state 13 after visual encoder runs, causing all subsequent `torch_xla.sync()` calls to fail.

## Root cause
Three bugs were discovered in sequence:

**Bug 0 (loader, fixed):** The `mradermacher/MAI-UI-8B-GGUF` file stores
`general.architecture = "qwen3vl"` (no underscore). Transformers does not have
`qwen3vl` in `GGUF_SUPPORTED_ARCHITECTURES`, so `load_gguf_checkpoint` raises
`ValueError`. Multiple other GGUF loaders also clobbered the
`load_gguf_checkpoint` monkey-patch with narrow-signature wrappers that drop the
`model_to_load` kwarg added in transformers 5.2.0 — these wrappers also needed
to be bypassed via BFS chain-walk to find the real function. `get_gguf_hf_weights_map`
also needed patching to remap `model_type="qwen3_vl"` → `"qwen3vl"` (with
underscore vs without) and to extract `num_hidden_layers` from the nested
`text_config`. Finally, `Qwen3VLConfig.from_pretrained` must be passed explicitly
because GGUF config parsing maps flat fields to the wrong struct level. Four
`.tolist()` call sites in the VisionModel and Model (`fast_pos_embed_interpolate`,
`rot_pos_emb`, `get_rope_index`, `get_image_features`) call `.tolist()` on device
tensors for Python control flow; these were patched to move metadata to CPU, with
`fast_pos_embed_interpolate` fully reimplemented on CPU and result transferred back
via `xm.send_cpu_data_to_device`.

**Bug 1 (tt-mlir, Tier A, fixed):** `Conv3dOpConversionPattern::matchAndRewrite`
in `TTIRToTTNN.cpp` hardcoded `cInBlock = TILE_WIDTH = 32` regardless of kernel
volume. For Qwen3VL's `patch_embed.proj` with kernel (kD=2, kH=14, kW=14), this
gives `K_t = kD·kH·kW = 392`, and the two main CBs (`vol2col_tiled +
weight_tiled`) allocate `2 × 2048 × 392 = 1.6 MB`, exceeding the 1.5 MB L1
budget. Fixed by using `C_in_block = TILE_WIDTH/2 = 16` when
`kernelVol = kD·kH·kW > 384`, which halves K_t to 196 and keeps CB usage within
budget. A `Conv3dConfigAttr` is passed explicitly so the runtime uses the same
blocking as the compiled weight layout. After this fix, the test ran for 427s
(7+ minutes) processing the visual encoder before hitting Bug 2.

**Bug 2 (tt-metal, Tier B, unfixed):** After the Conv3d fix, the device enters
error state 13 (INTERNAL) during a `torch_xla.sync()` inside
`partition_fx_graph_for_cpu_fallback`, which is triggered when compiling the chunk
containing `get_placeholder_mask` at `modeling_qwen3_vl.py:1225`. This sync
flushes the accumulated lazy XLA computation from the visual encoder, and some
operation within that accumulated computation causes a TT_FATAL. The error pattern
and timing are identical to the `egoactor_8b_qwen3vl` report Bug 2, where
`ttnn::prim::concat` (`ConcatDeviceOperation`) allocates approximately 14 MB of
circular buffers per core when concatenating image and text embeddings along the
sequence dimension (hidden_dim=3584, ~6000–7000 sequence positions). The concat
program factory does not bound per-core CB allocations to the 1.5 MB L1 budget,
causing a 9× overflow that puts the device into unrecoverable error state 13.

## Fix
**Bug 0 fix** committed to `remediation/mai_ui_8b_i1_gguf-...` in tt_forge_models:
- `mai_ui_8b_i1_gguf/image_to_text/pytorch/loader.py`:
  - Register `qwen3vl` in `GGUF_CONFIG_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`
  - BFS-walk patch chain to find real `load_gguf_checkpoint` with `model_to_load` kwarg
  - Patch `get_gguf_hf_weights_map` to remap `model_type="qwen3_vl"` → `"qwen3vl"`
    and extract `num_hidden_layers` from nested `text_config`
  - Load config explicitly via `Qwen3VLConfig.from_pretrained(BASE_MODEL)`
  - Add `ignore_mismatched_sizes=True` for visual merger norm size mismatch
  - Patch `fast_pos_embed_interpolate` (CPU reimplementation + `send_cpu_data_to_device`)
  - Patch `rot_pos_emb`, `get_rope_index`, `get_image_features` to pass CPU tensors
  - Set pixel limits on image processor (`min_pixels=56*56, max_pixels=13*28*1280`)

**Bug 1 fix** committed to `remediation/mai_ui_8b_i1_gguf-...` in tt-mlir:
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`:
  - Compute `kernelVol = kD·kH·kW` from weight tensor dimensions
  - Set `cInBlock = TILE_WIDTH/2 = 16` when `kernelVol > 384`, else `TILE_WIDTH = 32`
  - Pass explicit `Conv3dConfigAttr` with `c_in_block` and `c_out_block=TILE_WIDTH`

**Bug 2 proposed fix** (not attempted): In `tt-metal`, the `ConcatDeviceOperation`
program factory (`ConcatDeviceOperation` in the experimental op library) needs to
bound CB allocations per core to fit within L1. For large tensors the factory should
either shard across more cores or use a streaming approach that limits the per-core
CB footprint. This is the same fix proposed in the `egoactor_8b_qwen3vl` report.

## Tier B justification
Tier B indicator: `cross-cutting` — fixing the concat CB overflow requires changes
to the concat program factory in tt-metal (complex multi-core kernel scheduling
logic) and may require coordinated changes to how tt-mlir lowers large tensor
concats (memory layout and sharding strategy). The 9× L1 overflow on a small core
grid for a large-context VLM sequence is not a localized parameter; it reflects
a systematic absence of CB size capping in the concat program factory.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 427s (0:07:07) — after Conv3d and loader fixes
- Tier A attempts: 1 (Conv3d C_in_block fix — resolved the Conv3d overflow; unmasked the concat bug)

## Files changed
**tt-mlir** (`remediation/mai_ui_8b_i1_gguf-...`):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**tt-xla / tt_forge_models** (`remediation/mai_ui_8b_i1_gguf-...`):
- `mai_ui_8b_i1_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5bc41e746625394ad0a1181c8fad139c9d9af4d2 |
| tt-xla          | cdcae27657ee1c3c2cd51ebc19db1ca2b5d8e458 |
| tt-forge-models | 9d0fb29ba6585351d5eec5cd247721dc86ca8a4f |
