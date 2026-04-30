# Remediation Summary: hunyuan_video_comfyui_gguf-pytorch-T2V_Q4_K_S-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_comfyui_gguf/pytorch-T2V_Q4_K_S-single_device-inference]

## Result
FAIL — Q4_K_S dequantization hits ttnn-bitcast-cross-size-dtype-unsupported (uint8→int16 cross-size reinterpret in dequantize_blocks_BF16)

## Stack layer
tt-metal

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

While executing %view_103 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%lshift_17, torch.float32), kwargs = {})
Original traceback:
  File ".../transformer_hunyuan_video.py", line 473, in forward
    hidden_states = self.token_refiner(hidden_states, temb, attention_mask)
  ...
  File ".../diffusers/quantizers/gguf/utils.py", line 430, in dequantize_blocks_BF16
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
```

## Root cause
Three loader bugs prevented the model from loading at all:

1. **Missing requirements.txt**: `gguf` was not listed as a dependency, so the ImportError `raise ImportError("Please install torch and gguf>=0.10.0...")` reproduced on machines without gguf in their venv.
2. **Missing GGUFQuantizationConfig**: `from_single_file` was called without `quantization_config`, so Q4_K_S packed weight tensors (e.g. `[3072, 512]` uint8 for a `[256, 3072]` float weight) failed shape validation against `nn.Linear`.
3. **GGUFParameter.as_tensor recursion**: `torch.Tensor._make_subclass` re-enters `__torch_function__`, causing infinite recursion in `dequantize_gguf_tensor`. Fixed by wrapping in `DisableTorchFunctionSubclass`.

After all three loader fixes, the model loads and runs through 680s of compilation before the forward pass hits `dequantize_blocks_BF16`. This function reinterprets a `uint8` tensor as `int16` (cross-size bitcast: 1 byte → 2 bytes) as part of the BF16-scale dequantization step in Q4_K_S. TTNN's bitcast kernel only supports same-size dtype pairs; the `uint8→int16` reinterpret is not implemented, causing INTERNAL error code 13 at runtime.

## Fix
Loader fixes in `tt-xla/third_party/tt_forge_models` (remediation branch `remediation/hunyuan_video_comfyui_gguf-pytorch-T2V_Q4_K_S-single_device-inference`):
- `hunyuan_video_comfyui_gguf/pytorch/requirements.txt` — new file, adds `gguf>=0.10.0`
- `hunyuan_video_comfyui_gguf/pytorch/loader.py` — adds `GGUFQuantizationConfig`; patches `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass`; adds guidance input

Test config fix in `tt-xla` (same remediation branch):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — removes incorrect `KNOWN_FAILURE_XFAIL` (the prior XFAIL cited "28 GB BF16 DRAM overflow" which is wrong for a Q4_K_S quantized checkpoint); replaced with `required_pcc: 0.99`

**Proposed fix for compiler bug**: Implement cross-size bitcast support in TTNN's bitcast kernel to handle `uint8→int16` reinterpret, or lower `aten.view.dtype` cross-size cases through a byte-shuffle path. This is the same blocker that affects Wan2, HunyuanVideo 1.5 I2V, FLUX.1-Fill-dev, FLUX.1-dev, and flux1_arcticlatent GGUF loaders.

## Tier B justification
new-infrastructure — supporting cross-size bitcast (uint8→int16, 1→2 bytes) in TTNN requires a new kernel implementation in tt-metal. The existing bitcast kernel only handles same-size pairs and there is no existing mechanism to byte-reinterpret across element sizes on device.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    680.24s (0:11:20)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hunyuan_video_comfyui_gguf/pytorch/requirements.txt` (new)
- `tt-xla/third_party/tt_forge_models/hunyuan_video_comfyui_gguf/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b9c4c95071c81a2d30ace0276b92e67383c6b2f5 |
| tt-forge-models | 7490920b3cb531ec873cb8947fe166d299fe5500 |
