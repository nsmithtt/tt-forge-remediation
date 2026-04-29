# Remediation Summary: deepseek/deepseek_v3_mtp/pytorch-MTP_Main_Random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_mtp/pytorch-MTP_Main_Random-single_device-inference]

## Result
FAIL — TT BF16 matmul precision: PCC=0.9864 vs CPU BF16 (required 0.99)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
wh-bf16-matmul-accumulation-pcc

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original:
```
E   ModuleNotFoundError: No module named 'triton'
```

After 5 loader fixes (see Fix section), the test ran on TT silicon for ~336s and failed with:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9864107088779749. Required: pcc=0.99.
```

## Root cause

**Five sequential loader bugs** were fixed before reaching the precision failure:

1. **`ModuleNotFoundError: No module named 'triton'`** — `transformers/integrations/finegrained_fp8.py` imports `triton` at module level. On non-CUDA hardware the quantizer sets `dequantize=True` so triton kernels are never called, but the import still runs. Fixed by `_stub_triton_if_missing()` which creates minimal `types.ModuleType` stubs for `triton` and `triton.language`.

2. **`ValueError: Matrix dimensions must be divisible by block sizes (128, 128)`** — `Fp8Dequantize.convert` requires weight dimensions to be exactly divisible by the FP8 block size. DeepSeek-V3 MLA attention uses `kv_a_proj_with_mqa` of shape `(576, 2560)` where `576 % 128 != 0`. The stored scale already uses the correct ceil-divided block count. Fixed by `_patch_fp8_dequantize()` which pads the weight to `n_rows_b×128` × `n_cols_b×128` before dequantization and unpads with `[..., :rows, :cols].contiguous()` afterwards.

3. **`AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`** — transformers 5.x removed `DynamicCache.get_usable_length`. The remote `modeling_deepseek.py` was written against the older API. Fixed by `_patch_dynamic_cache()` which adds the method as `lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)`.

4. **dtype mismatch (`bfloat16 != float`)** — `Fp8Dequantize` outputs `float32` tensors but model inputs are `bfloat16`. Fixed by adding `model = model.to(dtype_override)` after `from_pretrained` in `load_model`.

5. **`torch._dynamo.exc.TorchRuntimeError: 'ndarray' object has no attribute 'add'`** — `DeepseekV3MoE.moe_infer` uses `tokens_per_expert.cpu().numpy()` plus a Python for-loop with data-dependent slice bounds, which breaks TorchDynamo/XLA compilation. Fixed by `_patch_deepseek_moe(model)` which replaces `moe_infer` with a static batched computation using `scatter_add_` for routing weights and `torch.stack` over all experts.

**Remaining failure after loader fixes**: The test ran successfully on TT silicon and produced `PCC=0.9864` vs CPU BF16 golden. The test requires `pcc=0.99`. The gap of ~0.014 between TT and CPU BF16 is consistent with the known WH Wormhole hardware BF16 matmul accumulation precision issue documented in tt-xla #2861, which has previously caused:
- Gemma 7B: PCC ~0.915
- Qwen3 4B (36 layers): PCC ~0.864
- GPT-J 6B: PCC ~0.75
- DeepSeek-V3-MTP Draft: PCC ~0.9611

The Main variant (luccafong/deepseek_mtp_main_random) uses fewer MoE experts than the full DeepSeek-V3 but still exhibits WH BF16 accumulation error.

## Fix

**Loader fixes (tt_forge_models)**:
- File: `deepseek/deepseek_v3_mtp/pytorch/loader.py`
- Branch in tt_forge_models: `remediation/deepseek-deepseek-v3-mtp-pytorch-mtp-main-random-single-device-inference`
- Commit: `dae996263727fe3e5a9663f15317d2491db4a832`
- All 5 fixes identical to those from the MTP_Draft_Random remediation, applied to the shared loader.

**Proposed compiler fix (Tier B — not attempted)**:
The WH BF16 matmul accumulation issue lives in tt-metal. The fix would require using higher-precision intermediate accumulation (FP32 or at least higher-fidelity BF16 dot products) in the matmul kernels. This is the same fix tracked in tt-xla #2861. It requires cross-cutting changes to the matmul kernel math fidelity settings across all operations, touching multiple files across tt-metal and tt-mlir.

## Tier B justification

`cross-cutting` — The WH BF16 matmul precision issue affects all BF16 matrix multiply operations across the entire stack, not a single file or pattern. Fixing it requires coordinated changes to math fidelity settings in tt-metal matmul kernels across multiple operations and files, and potentially changes to how tt-mlir selects precision modes. This exceeds the scope of a Tier A single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    336.66s (0:05:36)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_mtp/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6ffbad05942599a4f46343f9361da0f6769dca4a |
| tt-forge-models | dae996263727fe3e5a9663f15317d2491db4a832 |
