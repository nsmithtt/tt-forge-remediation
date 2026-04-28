# Remediation Summary: egoactor_8b_qwen3vl_i1_gguf-image_to_text-pytorch-8b_qwen3vl_i1_gguf-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[egoactor_8b_qwen3vl_i1_gguf/image_to_text/pytorch-8b_qwen3vl_i1_gguf-single_device-inference]

## Result
FAIL — second compiler-stack bug: `ttnn::concat` allocates 14 MB CBs per core (9× L1 limit); device enters unrecoverable error state

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
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=2)] grow to 14594560 B which is beyond max L1 size of 1572864 B
```
followed by:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
Two compiler-stack bugs were discovered, each unmasked by fixing the previous:

**Bug 1 (fixed — tt-mlir, Tier A):** `Conv3dOpConversionPattern::matchAndRewrite` in
`TTIRToTTNN.cpp` hardcoded `cInBlock = TILE_WIDTH = 32` regardless of kernel volume.
For Qwen3VL's `patch_embed.proj` with kernel (kD=2, kH=14, kW=14), this gives
`K_t = kD·kH·kW = 392`, and the two main CBs (`vol2col_tiled + weight_tiled`) allocate
`2 × 2048 × 392 = 1.6 MB`, exceeding the 1.5 MB L1 budget. Additionally the loop
producing C_in_block values like 8 and 4 failed the hardware's `C_in_block % 16 == 0`
alignment constraint. Fixed by using `C_in_block = TILE_WIDTH/2 = 16` when
`kernelVol > 384`, which halves K_t to 196 and brings CB usage to ~800 KB.

**Bug 2 (unfixed — tt-metal, Tier B):** After the Conv3d fix, `ttnn::prim::concat`
(`ConcatDeviceOperation`) allocates 14,594,560 B of CBs per core on a 3-core grid.
This corresponds to concatenating roughly 6000–7000 sequence positions of
hidden_dim=3584 (the text decoder dimension). The concat program factory allocates
CBs for the full per-core output without bounding them to L1 capacity, causing a
9× L1 overflow. This puts the TT device into error state 13, causing all subsequent
`torch_xla.sync()` calls to fail.

## Fix
**Bug 1 fix** committed to `remediation/egoactor_8b_qwen3vl_i1_gguf-...` in tt-mlir:
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`: replace the unconstrained
  `cInBlock = TILE_WIDTH` with a binary threshold: use `TILE_WIDTH/2 = 16` when
  `kernelVol = kD·kH·kW > 384`, otherwise `TILE_WIDTH = 32`. Also pass an explicit
  `Conv3dConfigAttr` with `c_in_block` and `c_out_block` so the runtime uses the
  same blocking as the compiled weight layout.

**Loader fixes** committed to tt_forge_models remediation branch:
- GGUF architecture registration for Qwen3VL
- Config wrapper chain fix (`Qwen3VLTextConfig` → `Qwen3VLConfig`)
- `fast_pos_embed_interpolate`, `get_rope_index`, `get_rope_cos_sin`, `get_image_features`
  patched to avoid `.tolist()` on TT tensors and to run CPU-side operations with
  `@torch.compiler.disable`

**Bug 2 proposed fix** (not attempted): In `tt-metal`, the `ConcatDeviceOperation`
program factory needs to bound CB allocations per core to fit within L1. For large
tensors the factory should either chunk the work across more cores or use a streaming
approach that limits the per-core CB footprint.

## Tier B justification
Tier B indicator: `cross-cutting` — fixing the concat CB overflow requires changes
to the concat program factory in tt-metal (complex multi-core kernel scheduling logic)
and may require coordinated changes to how tt-mlir lowers large tensor concats
(memory layout and sharding strategy). The 9× L1 overflow on a 3-core grid for a
6000+ token sequence is not a localized parameter; it reflects a systematic absence
of CB size capping in the concat program factory for large-context VLMs.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 450.26s (0:07:30)
- Tier A attempts: 1 (conv3d C_in_block fix — reduced the Conv3d overflow; unmasked the concat bug)

## Files changed
**tt-mlir** (`remediation/egoactor_8b_qwen3vl_i1_gguf-...`):
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

**tt-xla / tt_forge_models** (`remediation/egoactor_8b_qwen3vl_i1_gguf-...`):
- `egoactor_8b_qwen3vl_i1_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 762e48d62918f9b50bbe165f2971010f3b4d1c49 |
| tt-xla          | 46ea998d4afd01f09dfce47d6b2adacbc3eebfba |
| tt-forge-models | 3a93ffcc833abd322443792f675bab435bd04bd1 |
