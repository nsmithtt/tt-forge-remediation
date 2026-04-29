# Remediation Summary: deepseek/deepseek_v3_mtp/pytorch-MTP_Draft_Random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_mtp/pytorch-MTP_Draft_Random-single_device-inference]

## Result
FAIL — TT BF16 matmul precision: PCC=0.9611 vs CPU BF16 (required 0.99)

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

After 5 loader fixes (see Fix section), the test ran on TT silicon for ~628s and failed with:
```
AssertionError: PCC comparison failed. Calculated: pcc=0.9611789067122737. Required: pcc=0.99.
```

## Root cause

**Five sequential loader bugs** were fixed before reaching the precision failure:

1. **`ModuleNotFoundError: No module named 'triton'`** — `transformers/integrations/finegrained_fp8.py` imports `triton` at module level. On non-CUDA hardware the quantizer sets `dequantize=True` so triton kernels are never called, but the import still runs. Fixed by `_stub_triton_if_missing()` which creates minimal `types.ModuleType` stubs for `triton` and `triton.language`.

2. **`ValueError: Matrix dimensions (576, 2560) must be divisible by block sizes (128, 128)`** — `Fp8Dequantize.convert` requires weight dimensions to be exactly divisible by the FP8 block size. DeepSeek-V3 MLA attention uses `kv_a_proj_with_mqa` of shape `(576, 2560)` where `576 % 128 != 0`. The stored scale already uses the correct ceil-divided block count. Fixed by `_patch_fp8_dequantize()` which pads the weight to `n_rows_b×128` × `n_cols_b×128` before dequantization and unpads with `[..., :rows, :cols].contiguous()` afterwards.

3. **`AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`** — transformers 5.x removed `DynamicCache.get_usable_length`. The remote `modeling_deepseek.py` was written against the older API. Fixed by `_patch_dynamic_cache()` which adds the method as `lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)`.

4. **`RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float`** — `Fp8Dequantize` outputs `float32` tensors but model inputs are `bfloat16`. Fixed by adding `model = model.to(dtype_override)` after `from_pretrained` in `load_model`.

5. **`torch._dynamo.exc.TorchRuntimeError: ... 'ndarray' object has no attribute 'add'`** — `DeepseekV3MoE.moe_infer` uses `tokens_per_expert.cpu().numpy()` plus a Python for-loop with data-dependent slice bounds (`end_idx = start_idx + num_tokens`), which breaks TorchDynamo/XLA compilation. Fixed by `_patch_deepseek_moe(model)` which replaces `moe_infer` with a static batched computation: `scatter_add_` for routing weights and `torch.stack` over all experts.

**Remaining failure after loader fixes**: The test ran successfully on TT silicon and produced `PCC=0.9611` vs CPU BF16 golden. The test requires `pcc=0.99`. The golden is CPU BF16 output (same dtype, same model, same inputs). The gap of ~0.039 between TT and CPU BF16 is consistent with the known WH Wormhole hardware BF16 matmul accumulation precision issue documented in tt-xla #2861, which has previously caused:
- Gemma 7B: PCC ~0.915
- Qwen3 4B (36 layers): PCC ~0.864
- GPT-J 6B: PCC ~0.75

The DeepSeek V3 MTP Draft model uses MoE with 72 routed experts (top-6 per token) across 10 decoder layers plus Multi-head Latent Attention. The batched MoE implementation (running all 72 experts on every token and applying routing weights) may additionally exacerbate TT-vs-CPU divergence, since WH BF16 matmul arithmetic is not exactly IEEE 754 compliant for the zero-weight expert paths.

## Fix

**Loader fixes (tt_forge_models)**:
- File: `deepseek/deepseek_v3_mtp/pytorch/loader.py`
- Branch: `remediation/deepseek-deepseek-v3-mtp-pytorch-mtp-draft-random-single-device-inference`
- Commits:
  1. `1ec795e1` — stub triton for FP8 dequantization on non-CUDA hardware (`_stub_triton_if_missing`)
  2. `3a3adb39` — patch `Fp8Dequantize` for non-aligned MLA weights (`_patch_fp8_dequantize`)
  3. `2cdf46946` — add `DynamicCache.get_usable_length` shim for transformers 5.x
  4. `263a98115` — cast model to `dtype_override` after FP8 dequantization
  5. `6a04d553` — replace `moe_infer` with batched static computation (`_patch_deepseek_moe`)

**Proposed compiler fix (Tier B — not attempted)**:
The WH BF16 matmul accumulation issue lives in tt-metal. The fix would require using higher-precision intermediate accumulation (FP32 or at least higher-fidelity BF16 dot products) in the matmul kernels. This is the same fix tracked in tt-xla #2861. It requires cross-cutting changes to the matmul kernel math fidelity settings across all operations, touching multiple files across tt-metal and tt-mlir.

## Tier B justification

`cross-cutting` — The WH BF16 matmul precision issue affects all BF16 matrix multiply operations across the entire stack, not a single file or pattern. Fixing it requires coordinated changes to math fidelity settings in tt-metal matmul kernels across multiple operations and files, and potentially changes to how tt-mlir selects precision modes. This exceeds the scope of a Tier A single-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~628s (wall-clock for the silicon run that produced PCC=0.9611)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_mtp/pytorch/loader.py` (in tt_forge_models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7e6f59cb757c4a9f986b429af7a12f237269cccc |
| tt-forge-models | 6a04d553c9ba75e5ba4bee654bb7f562754be9f6 |
