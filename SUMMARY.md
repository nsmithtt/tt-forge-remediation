# Remediation Summary: felldude_qwen_3_vl_uncensored_gguf-image_to_text-pytorch-8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[felldude_qwen_3_vl_uncensored_gguf/image_to_text/pytorch-8B_INSTRUCT_UNCENSORED_V2_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — `ttnn::concat` allocates 14 MB CBs per core (9× L1 limit); device enters unrecoverable error state

## Stack layer
tt-metal

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
ValueError: GGUF model with architecture qwen3vl is not supported yet.
```
After loader fix, progressed to:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=2)] grow to 14594560 B which is beyond max L1 size of 1572864 B
```
followed by:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Three bugs were discovered in sequence:

**Bug 1 (fixed — loader):** `load_gguf_checkpoint` raises `ValueError: GGUF model with
architecture qwen3vl is not supported yet.` because `qwen3vl` (the `general.architecture`
field in the mradermacher GGUF) is absent from `GGUF_SUPPORTED_ARCHITECTURES`. Additionally,
`get_gguf_hf_weights_map` looks up the model by `model_type = "qwen3_vl"` (underscored) but
gguf-py's `MODEL_ARCH_NAMES` expects `"qwen3vl"` (no underscore). A third complication is
that 26+ loader modules in the test session replace `load_gguf_checkpoint` with broken
fixed-signature wrappers that drop the `model_to_load` kwarg introduced in transformers
5.2.0, causing a `TypeError` when `return_tensors=True` during weight loading.

**Bug 2 (fixed — tt-mlir, Tier A):** `Conv3dOpConversionPattern::matchAndRewrite` in
`TTIRToTTNN.cpp` hardcoded `cInBlock = TILE_WIDTH = 32` regardless of kernel volume.
For Qwen3VL's `patch_embed.proj` with kernel (kD=2, kH=14, kW=14), this gives
`K_t = kD·kH·kW = 392`, and the two main CBs (`vol2col_tiled + weight_tiled`) would
overflow the 1.5 MB L1 budget. Fixed by using `C_in_block = TILE_WIDTH/2 = 16` when
`kernelVol > 384` (cherry-pick of commit from egoactor_8b_qwen3vl_i1_gguf remediation
branch, same architecture).

**Bug 3 (unfixed — tt-metal, Tier B):** After the Conv3d fix, `ttnn::prim::concat`
(`ConcatDeviceOperation`) allocates 14,594,560 B of CBs per core on a 3-core grid
`(x=0,y=0)-(x=0,y=2)`. This corresponds to concatenating roughly 6000–7000 sequence
positions of hidden_dim=3584 (the text decoder dimension). The concat program factory
allocates CBs for the full per-core output without bounding them to L1 capacity,
causing a 9× L1 overflow that puts the TT device into error state 13.

## Fix
**Bug 1 fix** committed to `remediation/felldude_qwen_3_vl_uncensored_gguf-...` in tt_forge_models:
- `felldude_qwen_3_vl_uncensored_gguf/image_to_text/pytorch/loader.py`:
  - `_register_qwen3vl_gguf_architecture()`: adds `qwen3vl` to `GGUF_CONFIG_MAPPING`,
    patches `get_gguf_hf_weights_map` to remap `model_type "qwen3_vl" → "qwen3vl"`,
    and installs a compat `load_gguf_checkpoint` wrapper that walks the broken chain
    to find and call the real function with `model_to_load`
  - `_patch_qwen3vl_for_tt_device()`: patches `fast_pos_embed_interpolate`, `rot_pos_emb`,
    `get_rope_index`, `get_image_features` to move `.tolist()`-calling tensors to CPU
  - `load_model()`: registers GGUF architecture, loads config from base model
    `Qwen/Qwen3-VL-8B-Instruct`, passes `ignore_mismatched_sizes=True` to handle the
    merger.norm size mismatch

**Bug 2 fix** committed to `remediation/felldude_qwen_3_vl_uncensored_gguf-...` in tt-mlir:
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: replace hardcoded `cInBlock = TILE_WIDTH`
  with `cInBlock = (kernelVol > 384) ? TILE_WIDTH/2 : TILE_WIDTH`. Pass an explicit
  `Conv3dConfigAttr` with `c_in_block` and `c_out_block` so the runtime uses the same
  blocking as the compiled weight layout.

**Bug 3 proposed fix** (not attempted): In `tt-metal`, the `ConcatDeviceOperation`
program factory needs to bound CB allocations per core to fit within L1. For large
tensors the factory should either chunk the work across more cores or use a streaming
approach that limits the per-core CB footprint.

## Tier B justification
Tier B indicator: `cross-cutting` — fixing the concat CB overflow requires changes
to the concat program factory in tt-metal (complex multi-core kernel scheduling logic)
and may require coordinated changes to how tt-mlir lowers large tensor concats
(memory layout and sharding strategy). The 9× L1 overflow on a 3-core grid for a
6000+ token sequence is not a localized parameter; it reflects a systematic absence
of CB size capping in the concat program factory for large-context VLMs. This is the
identical bug found in the egoactor_8b_qwen3vl_i1_gguf remediation (same model
architecture, same fingerprint `concat-cb-size-exceeds-l1`).

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 463.27s (0:07:43)
- Tier A attempts: 1 (conv3d C_in_block fix — eliminated Conv3d CB overflow; unmasked the concat bug)

## Files changed
**tt-mlir** (`remediation/felldude_qwen_3_vl_uncensored_gguf-...`):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**tt_forge_models** (`remediation/felldude_qwen_3_vl_uncensored_gguf-...`):
- `felldude_qwen_3_vl_uncensored_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 1007de3e674713b0e40da1eb0bae2b0b80b3cd95 |
| tt-xla          | 51337f4916d0a7f0b514371f4f5e2755e9f0ff6f |
| tt-forge-models | 3684e3a0f7f3cf57d4efff8c281c2bf1dfb1d672 |
