# Remediation Summary: hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_cfg_distilled_Q4_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_1_5_i2v_720p_gguf/pytorch-720p_i2v_cfg_distilled_Q4_K_S-single_device-inference]

## Result
FAIL — Q4_K dequantization performs a cross-element-size aten.view.dtype (uint8→float16) that TT hardware does not support

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

While executing %view_33 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_10, torch.float16), kwargs = {})
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
  File ".../diffusers/quantizers/gguf/utils.py", line 361, in dequantize_blocks_Q4_K
    dmin = dmin.view(torch.float16).to(dtype)
```

## Root cause

The Q4_K dequantizer (`dequantize_blocks_Q4_K` in `diffusers/quantizers/gguf/utils.py`) reinterprets a uint8 tensor as float16 via `dmin.view(torch.float16)`. This produces `aten.view.dtype` with mismatched element sizes (uint8=1 byte, float16=2 bytes). TTNN's bitcast does not support cross-element-size reinterpretation, resulting in INTERNAL Error code: 13 in `partition_fx_graph_for_cpu_fallback`. The same bug affects all Q-type GGUF quantization levels (Q4_K, Q5_K, Q6_K, Q8_0) because they all use byte-level `view()` to extract packed sub-byte values.

Five loader-layer bugs were also fixed:

1. **Missing requirements.txt**: `gguf>=0.10.0` was not listed, causing ImportError when not installed.
2. **URL**: original used `resolve/main`; should be `blob/main` for GGUF HF downloads.
3. **Config**: `HunyuanVideo15Transformer3DModel` was not registered in `SINGLE_FILE_LOADABLE_CLASSES` in diffusers 0.37.1 (raises `ValueError: FromOriginalModelMixin is currently only compatible with ...`). Fixed by runtime-patching the dict with a key conversion function. Also: `tencent/HunyuanVideo-1.5` config has three mismatches: `patch_size` stored as list, `qk_norm` as bool, `in_channels=32` instead of 65.
4. **Key mapping**: jayn7 GGUF uses original HunyuanVideo key names; diffusers expects remapped names (`img_in.*→x_embedder.*`, `double_blocks.*→transformer_blocks.*`, etc.).
5. **Timestep dtype**: original loader used `torch.long`; model expects `torch.bfloat16`.
6. **GGUFParameter recursion**: `GGUFParameter.__torch_function__` recursed under TorchDynamo; patched `as_tensor` with `DisableTorchFunctionSubclass`.

## Fix
**Loader fixes** committed to `remediation/hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_cfg_distilled_Q4_K_S-single_device-inference` in tt-forge-models:
- `hunyuan_video_1_5_i2v_720p_gguf/pytorch/requirements.txt`: added with `gguf>=0.10.0`
- `hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py`: all 5 loader fixes applied (same fix set as cfg_distilled_Q6_K remediation branch)

**Compiler-stack fix (proposed, not implemented)**: In tt-metal, add support for cross-element-size `ttnn::bitcast` (uint8→float16 and similar pairs), or add a decomposition in tt-xla that replaces `aten.view.dtype` with an explicit byte-level reshape + cast sequence that TTNN can execute.

## Tier B justification
new-infrastructure — Adding cross-element-size bitcast to TTNN requires either a new kernel or a multi-step decomposition (unpack uint8 pairs → reconstruct float16) that touches the tt-metal kernel infrastructure. This is not a scoped one-function fix; it requires new hardware kernel support or a non-trivial lowering pipeline addition.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 561.23s (0:09:21)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/requirements.txt` (new file)
- `tt-xla/third_party/tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py` (5 loader fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b74df6e36b9802a361f51a2609c4c9ccefdabb37 |
| tt-forge-models | b382376468c68933a508c12abd5f9fcd5501e073 |
