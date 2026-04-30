# Remediation Summary: hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_cfg_distilled_Q6_K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_1_5_i2v_720p_gguf/pytorch-720p_i2v_cfg_distilled_Q6_K-single_device-inference]

## Result
FAIL — Q6_K dequantization performs a cross-element-size aten.view.dtype (uint8→float16) that TT hardware does not support

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

While executing %view_33 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_12, torch.float16), kwargs = {})
Original traceback:
  File ".../diffusers/models/transformers/transformer_hunyuan_video15.py", line 660, in forward
    encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)
  File ".../diffusers/models/transformers/transformer_hunyuan_video15.py", line 407, in forward
    hidden_states = self.linear_2(hidden_states)
  File ".../diffusers/quantizers/gguf/utils.py", line 595, in forward
    return self.forward_native(inputs)
  File ".../diffusers/quantizers/gguf/utils.py", line 598, in forward_native
    weight = dequantize_gguf_tensor(self.weight)
  File ".../diffusers/quantizers/gguf/utils.py", line 528, in dequantize_gguf_tensor
    dequant = dequant_fn(blocks, block_size, type_size)
  File ".../diffusers/quantizers/gguf/utils.py", line 313, in dequantize_blocks_Q6_K
    d = d.view(torch.float16).to(dtype)
```

## Root cause
The loader had five bugs that prevented the test from reaching silicon; all were fixed. After the loader fixes, the test compiles and runs on TT hardware but fails in tt-metal. The Q6_K GGUF format stores a float16 scale factor `d` as raw uint8 bytes; `dequantize_blocks_Q6_K` uses `d.view(torch.float16)` to reinterpret those bytes. This is a cross-element-size bitcast (2 × uint8 bytes → 1 × float16 value). TTNN's bitcast primitive does not support cross-element-size dtype reinterpretation, so the operation fails with `INTERNAL: Error code: 13`. The same bug affects all GGUF Q*_K quantization types that use a float16 scale factor stored as uint8.

The five loader bugs that were fixed:
1. URL: `resolve/main` → `blob/main` in the HuggingFace GGUF file URL
2. Config: load from `tencent/HunyuanVideo-1.5` with subfolder `transformer/720p_i2v_distilled` for cfg-distilled variants; patch `patch_size` list→int, `qk_norm` bool→string, `in_channels` 32→65
3. Key mapping: register `_convert_hunyuan_video15_gguf_to_diffusers` in `SINGLE_FILE_LOADABLE_CLASSES` — maps jayn7 GGUF keys (`img_in.*`, `double_blocks.*`, etc.) to diffusers keys (`x_embedder.*`, `transformer_blocks.*`, etc.)
4. Timestep dtype: construct timestep tensor as `torch.bfloat16` rather than `torch.long`/float32 (original `mat1 and mat2 must have the same dtype, but got BFloat16 and Float` error)
5. GGUFParameter recursion: patch `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass` to prevent infinite recursion under `torch._dynamo`

## Fix
**Loader fixes** committed to `remediation/hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_cfg_distilled_Q6_K-single_device-inference` branch in tt-forge-models:
- `hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py`: all 5 loader fixes applied (same fix set as cfg_distilled_Q8_0 remediation branch, which already included cfg_distilled variant support)

**Compiler-stack fix (proposed, not implemented)**: In tt-metal, add support for cross-element-size `ttnn::bitcast` (uint8→float16 and similar pairs), or add a decomposition in tt-xla that replaces `aten.view.dtype` with an explicit byte-level reshape + cast sequence that TTNN can execute.

## Tier B justification
new-infrastructure — Adding cross-element-size bitcast to TTNN requires either a new kernel or a multi-step decomposition (unpack uint8 pairs → reconstruct float16) that touches the tt-metal kernel infrastructure. This is not a scoped one-function fix; it requires new hardware kernel support or a non-trivial lowering pipeline addition.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 268.05s (0:04:28)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py` (5 loader fixes)

## Submodule hashes
| Submodule       | Commit                                    |
|-----------------|-------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc  |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee  |
| tt-xla          | 701637eae3359fc7441dcab2bee1fc1b122e70a0  |
| tt-forge-models | 4242efa82c9e6266b49d5556058d7398731041fa  |
