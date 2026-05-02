# Remediation Summary: next2_air_gguf-image_to_text-pytorch-base_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[next2_air_gguf/image_to_text/pytorch-base_Q4_K_M-single_device-inference]

## Result
FAIL — dynamic-shape boolean-masked gather in get_placeholder_mask is Tier B (new-infrastructure)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
dynamic-shape-boolean-index-embedding-scatter

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

Full traceback terminates at:
```
transformers/models/qwen3_5/modeling_qwen3_5.py:1584: in get_placeholder_mask
    inputs_embeds[special_image_mask].numel() == image_features.numel(),
...
torch_xla/_dynamo/dynamo_bridge.py:346: in extract_graph_helper
    torch_xla.sync(reset_scope=False)
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Two bugs were found; the first was fixed:

**Bug 1 (loader, fixed):** Four Qwen3.5 VL methods (`fast_pos_embed_interpolate`,
`rot_pos_emb`, `get_rope_index`, `get_image_features`) call `.tolist()` on
device tensors for metadata control flow. On TT silicon, any eager read from a
device tensor triggers a PJRT sync that fails with Error code: 13. Fix: patch
the four class methods in the loader to move the metadata arguments to CPU
before the `.tolist()` call.

**Bug 2 (tt-mlir / compiler, Tier A, fixed):** `Qwen3_5VisionModel` uses a
Conv3d patch-embedding layer with kernel `[2, 14, 14]`. With `c_in_block =
TILE_WIDTH = 32`, the volume-to-column and weight circular buffers total ~2 MB,
exceeding the L1 limit of ~1.5 MB. Fix: compute a safe `c_in_block` in
`TTIRToTTNN.cpp` by halving from `TILE_WIDTH` until
`c_in_block * kernelElements ≤ MAX_CB_TILES * TILE_WIDTH`.

**Bug 3 (tt-xla / compiler, Tier B, terminal):** After Bug 1 and Bug 2 are
fixed, the vision encoder compiles and runs. The subsequent text decoder
compilation fails in `get_placeholder_mask`:

```python
special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
torch_compilable_check(inputs_embeds[special_image_mask].numel() == image_features.numel(), ...)
```

`inputs_embeds[special_image_mask]` is a boolean-masked gather whose output
shape depends on the number of image tokens at runtime. TT device compilation
requires fully static shapes. The XLA graph captured for the subgraph
containing this operation cannot be compiled, surfacing as INTERNAL:13 in the
`partition_fx_graph_for_cpu_fallback` path.

## Fix
- **Bug 1 (loader):** `tt-xla/third_party/tt_forge_models/next2_air_gguf/image_to_text/pytorch/loader.py` — added `_patch_qwen3_5_for_tt_device()` patching four class methods, called from `load_model()`. Added pixel limits `min_pixels = 56*56`, `max_pixels = 13*28*1280` applied in `_load_processor()`.
- **Bug 2 (tt-mlir):** `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `Conv3dOpConversionPattern::matchAndRewrite`: replaced hardcoded `c_in_block = TILE_WIDTH` with a safe computation that halves until `c_in_block * kernelElements ≤ MAX_CB_TILES * TILE_WIDTH` and divides `c_in_aligned` evenly; passes explicit `Conv3dConfigAttr` with the computed value.
- **Bug 3 (proposed fix):** Override `Qwen3_5Model.get_placeholder_mask` in the loader to replace `inputs_embeds[special_image_mask]` with a static-shape alternative (e.g., pre-allocate output buffer, use `torch.where` / `masked_select` + reshape with known `n_image_tokens * hidden_size`). However, this fix requires dynamic shape support or a static-shape reformulation in the PJRT compilation pipeline, and has not been validated as a loader-layer fix under the current skill rules.

## Tier B justification
Tier B indicator: **new-infrastructure** — TT device compilation requires fully static shapes throughout the compiled graph. Supporting data-dependent output shapes from boolean-masked gathers (`tensor[bool_mask]`) requires new shape inference infrastructure in the PJRT / StableHLO compilation path that does not currently exist.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    224.38s (0:03:44)
- Tier A attempts: 1 (Conv3d c_in_block fix — applied and confirmed to unblock the vision encoder)

## Files changed
- `tt-xla/third_party/tt_forge_models/next2_air_gguf/image_to_text/pytorch/loader.py`
- `tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | ba94946c4 |
| tt-xla          | 0ac5d9b9d |
| tt-forge-models | d5bb76b507 |
