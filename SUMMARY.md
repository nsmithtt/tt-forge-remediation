# Remediation Summary: hunyuan_video_1_5_i2v_720p_gguf-pytorch-720p_i2v_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hunyuan_video_1_5_i2v_720p_gguf/pytorch-720p_i2v_Q4_K_M-single_device-inference]

## Result
FAIL — `aten.view.dtype` (uint8→float16 cross-size bitcast) in Q4_K GGUF dequantization is not supported on TT device; INTERNAL: Error code 13.

## Stack layer
loader, tt-xla

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
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

While executing %view_33 : [num_users=1] = call_function[target=torch.ops.aten.view.dtype](args = (%slice_10, torch.float16), kwargs = {})
Original traceback:
  File "diffusers/models/transformers/transformer_hunyuan_video15.py", line 660, in forward
    encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)
  File "diffusers/models/transformers/transformer_hunyuan_video15.py", line 407, in forward
    hidden_states = self.linear_2(hidden_states)
  File "diffusers/quantizers/gguf/utils.py", line 598, in forward_native
    weight = dequantize_gguf_tensor(self.weight)
  File "diffusers/quantizers/gguf/utils.py", line 528, in dequantize_gguf_tensor
    dequant = dequant_fn(blocks, block_size, type_size)
  File "diffusers/quantizers/gguf/utils.py", line 361, in dequantize_blocks_Q4_K
    dmin = dmin.view(torch.float16).to(dtype)

## Root cause
Five loader bugs were found and fixed (all in tt-forge-models):

1. **404 URL**: Loader used `resolve/main` in the GGUF URL; diffusers `_extract_repo_id_and_weights_name` strips only `blob/main/`, so `resolve/main/` was included in `weights_name` and then prepended again by `_get_model_file`. Fixed: `blob/main`.

2. **Config type mismatches**: `tencent/HunyuanVideo-1.5` config.json has three wrong values: `patch_size` stored as `[1,1,1]` list (class expects int scalars), `patch_size_t` stored as `null` (class expects int), `qk_norm` stored as `true` (bool, class expects `"rms_norm"` string), and `in_channels: 32` (wrong; i2v model uses 65 input channels). Fixed by monkey-patching `HunyuanVideo15Transformer3DModel.from_config`.

3. **Missing GGUF→diffusers key mapping**: The jayn7 GGUF uses HunyuanVideo's original key naming (`double_blocks.N.img_attn_qkv.*`) while diffusers uses its own (`transformer_blocks.N.attn.to_q.*`). The model class was not registered in `SINGLE_FILE_LOADABLE_CLASSES`, so the identity mapping left all weights on meta device. Fixed by implementing a complete `_convert_hunyuan_video15_gguf_to_diffusers()` key mapping function and registering it.

4. **Timestep dtype mismatch**: `load_inputs` passed `timestep` as `torch.long`, but `HunyuanVideo15TimeEmbedder.forward` casts the Fourier projection output back to `timestep.dtype`, producing a Long tensor that fails at the BFloat16 linear layer. The real diffusion pipeline converts timestep to latents' dtype (BFloat16) before calling the transformer. Fixed: `dtype=torch.bfloat16`.

5. **GGUFParameter.as_tensor dynamo recursion**: During torch.compile tracing of `GGUFLinear.forward_native`, `dequantize_gguf_tensor` calls `tensor.as_tensor()` which internally calls `torch.Tensor._make_subclass`. This re-enters `GGUFParameter.__torch_function__` → `super().__torch_function__` recursively, exhausting the call stack. Fixed by patching `GGUFParameter.as_tensor` to use `torch._C.DisableTorchFunctionSubclass()`.

After all loader fixes, the test reaches the TT device execution step and fails with INTERNAL: Error code 13 at `aten.view.dtype`. The `dequantize_blocks_Q4_K` function calls `dmin.view(torch.float16)` — a cross-size dtype bitcast reinterpreting uint8 storage as float16 (1 byte → 2 bytes). The TT runtime does not support `ttnn::bitcast` for dtype pairs of different element sizes.

## Fix
Five loader fixes in `tt-forge-models` on the remediation branch. The terminal compiler-stack bug is unfixed.

**tt-forge-models** `hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py`:
- Commit `4c62d6b` (amended into `906dcfcb56`): Fix URL `resolve/main` → `blob/main`
- Commit `906dcfcb56` (commit 2 on branch): Full GGUF→diffusers key mapping + config patches + in_channels=32→65 override
- Commit `906dcfcb56`: Timestep dtype `torch.long` → `torch.bfloat16`
- Commit `19a4aeb929`: Patch `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass`

**Proposed compiler-stack fix (Tier B, not implemented):**
In tt-mlir's StableHLO→TTIR lowering for `stablehlo.bitcast_convert`, add support for cross-element-size dtype pairs (e.g., `[N] uint8 → [N/2] float16`). Alternatively, in tt-xla's partition logic, add a CPU fallback for `aten.view.dtype` operations that cross element-size boundaries.

## Tier B justification
- **new-infrastructure**: The `aten.view.dtype` cross-size bitcast (`uint8 → float16`) is not a missing pattern for a known-shaped operation — it requires new kernel infrastructure in tt-metal to support reinterpretation of memory with different element sizes, plus a new lowering path in tt-mlir for `stablehlo.bitcast_convert` when source and target element widths differ. This affects every Q4_K and Q5_K GGUF model, not just HunyuanVideo.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    370.85s (0:06:10)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/hunyuan_video_1_5_i2v_720p_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 951c99a0c2b2daf71214751eeb154a2d7282aa3a |
| tt-forge-models | 19a4aeb929e2b17da09de98f30b7cf1b7b62ef87 |
