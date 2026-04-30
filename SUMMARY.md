# Remediation Summary: hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_Q6_K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_1_5_i2v_720p_gguf/pytorch-720p_i2v_Q6_K-single_device-inference]

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
The loader layer had five bugs (all fixed): wrong blob/main URL, `HunyuanVideo15Transformer3DModel` not in diffusers `SINGLE_FILE_LOADABLE_CLASSES`, config type mismatches (patch_size list, qk_norm bool→str, in_channels 32→65), timestep dtype (long→bfloat16), and `GGUFParameter.as_tensor` recursing under TorchDynamo.

After fixing the loader, the test reaches silicon execution and fails in tt-metal. The Q6_K GGUF quantization format stores a scale factor `d` as raw uint8 bytes that represent a float16 value; `dequantize_blocks_Q6_K` uses `d.view(torch.float16)` to reinterpret those bytes. This is a cross-element-size bitcast (1-byte uint8 → 2-byte float16). TTNN's `bitcast` primitive only supports same-element-size dtype pairs, so the operation fails with `INTERNAL: Error code: 13`. The same bug affects Q4_K, Q5_K, and all other GGUF Q*_K quantization types that use a float16 scale factor.

## Fix
**Loader fixes** (committed to `remediation/hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_Q6_K-single_device-inference` branch in tt-forge-models, cherry-picked from the Q5_K_M remediation branch):
1. `hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py`: use `blob/main` URL instead of `resolve/main`
2. `loader.py`: register `HunyuanVideo15Transformer3DModel` in `SINGLE_FILE_LOADABLE_CLASSES` with `_convert_hunyuan_video15_gguf_to_diffusers` key-mapping function
3. `loader.py`: patch `HunyuanVideo15Transformer3DModel.from_config` to fix patch_size list, qk_norm bool→str conversion, and in_channels 32→65
4. `loader.py`: use `torch.bfloat16` for timestep dtype in `load_inputs`
5. `loader.py`: patch `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass` to prevent TorchDynamo recursion

**Compiler-stack fix (proposed, not implemented)**: In tt-metal, add support for cross-element-size `ttnn::bitcast` (uint8→float16 and similar pairs), or add a decomposition in tt-xla that replaces `aten.view.dtype` with an explicit byte-level reshape + cast sequence that TTNN can execute.

## Tier B justification
new-infrastructure — Adding cross-element-size bitcast to TTNN requires either a new kernel or a multi-step decomposition (unpack uint8 pairs → reconstruct float16) that touches the tt-metal kernel infrastructure. This is not a scoped one-function fix; it requires new hardware kernel support or a non-trivial lowering pipeline addition.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 685.83s (0:11:25)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py` (5 loader fixes, cherry-picked from Q5_K_M remediation branch)

## Submodule hashes
| Submodule       | Commit                                    |
|-----------------|-------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc  |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee  |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8  |
| tt-forge-models | e32075b8d4d619825994c574d7e56764c2838b29  |
