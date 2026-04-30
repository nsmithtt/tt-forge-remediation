# Remediation Summary: hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_cfg_distilled_Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_1_5_i2v_720p_gguf/pytorch-720p_i2v_cfg_distilled_Q8_0-single_device-inference]

## Result
FAIL — `aten.view.dtype` (uint8→int8 bitcast) in `dequantize_blocks_Q8_0` not supported by TTNN runtime (INTERNAL Error code: 13)

## Stack layer
loader, tt-metal

## Tier
B

## Bug fingerprint
ttnn-bitcast-cross-size-dtype-unsupported

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

While executing %view_33 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_10, torch.int8), kwargs = {})
Original traceback:
  File ".../diffusers/quantizers/gguf/utils.py", line 232, in dequantize_blocks_Q8_0
    x = x.view(torch.int8)
```

## Root cause
The model loads from a GGUF Q8_0 checkpoint. During inference, `dequantize_blocks_Q8_0` in diffusers performs two dtype-view operations on the packed Q8_0 blocks:
1. `d = d.view(torch.float16)` — uint8 → float16 cross-size bitcast (reinterpret 2 uint8 bytes as one float16)
2. `x = x.view(torch.int8)` — uint8 → int8 same-size bitcast (reinterpret sign)

Both operations lower to `aten.view.dtype` which the TTNN runtime rejects with `INTERNAL: Error code: 13`. The test fails on the first view encountered under compilation (`x.view(torch.int8)` in the `context_embedder_2.linear_1` layer). The loader bugs (wrong URL, missing key mapping, wrong config, wrong in_channels, wrong timestep dtype) were all fixed; the terminal failure is a runtime capability gap in TTNN.

Five loader bugs were fixed in `tt_forge_models`:
1. URL: `resolve/main` → `blob/main` for the HuggingFace GGUF file URL
2. Config: load config from `tencent/HunyuanVideo-1.5` base repo with correct subfolder (`transformer/720p_i2v_distilled` for cfg-distilled variants), patch `patch_size` list→int, `qk_norm` bool→string, `in_channels` 32→65
3. Key mapping: register `_convert_hunyuan_video15_gguf_to_diffusers` in `SINGLE_FILE_LOADABLE_CLASSES` to map jayn7 GGUF keys to diffusers model keys
4. Timestep dtype: construct timestep tensor as `bfloat16` (was defaulting to float32, causing the original `mat1 and mat2 must have the same dtype` error)
5. GGUFParameter recursion: patch `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass` to prevent infinite recursion under `torch._dynamo`

The terminal blocker is Tier B: `aten.view.dtype` (any dtype reinterpret-cast view) is not supported by `ttnn::bitcast` regardless of whether the element sizes match or differ. This requires new infrastructure in the TTNN op set or a decomposition that avoids `view.dtype` entirely.

## Fix
Loader fixes committed to `remediation/hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_cfg_distilled_Q8_0-single_device-inference` branch in `tt_forge_models`:
- `tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py` — complete rewrite with all 5 loader fixes

No compiler-stack fix attempted (Tier B).

The proposed compiler fix would be: implement `aten.view.dtype` support in the TTNN backend, or add a decomposition in tt-xla that replaces `x.view(torch.int8)` / `d.view(torch.float16)` with a byte-level bitcast operation that TTNN supports.

## Tier B justification
**Indicator**: new-infrastructure

`aten.view.dtype` is a reinterpret-cast of tensor memory. TTNN currently has no op that maps to this semantic across the general case. Implementing it requires either:
- A new TTNN kernel supporting arbitrary dtype reinterpretation at the byte level
- Or a graph-level decomposition into a sequence of supported element-wise ops (e.g. bitwise extract + scale for float→int reinterpretation)

Either path is new infrastructure touching more than 3 files across tt-metal and tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    481.81s (0:08:01)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7d7972f3087e29af58b4a831eca01b75cd4b4e99 |
| tt-forge-models | 0f2c78d0272e3e9cdefff76dd441b52694b8f8fb |
