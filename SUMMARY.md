# Remediation Summary: minicpm_v_2_6_int4-pytorch-Default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[minicpm_v_2_6_int4/pytorch-Default-single_device-inference]

## Result
FAIL — PCC 0.8995 < required 0.99; root cause is ttmlir-f32-precision-not-preserved in SiglipVisionTransformer attention (26 layers × FP32 softmax not honored by TT MLIR compiler)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-f32-precision-not-preserved

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8995916579379379. Required: pcc=0.99.

## Root cause
The SiglipVisionTransformer (navit_siglip, 26 attention layers) uses eager attention with an explicit FP32 softmax upcast for numerical stability:

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

On CPU-BF16, torch computes the softmax in FP32 then casts the result back to BF16 (as requested by `dtype=torch.float32`). On TT, the tt-mlir compiler does not preserve this FP32 upcast and runs the softmax in BF16 instead (ttmlir-f32-precision-not-preserved). Over 26 vision encoder layers, accumulated BF16 softmax error versus the CPU-FP32 softmax yields PCC ≈ 0.8995 in the final logits.

Multiple loader-level fixes were required before reaching this compiler-stack bug (see loader fixes below). The loader is now correct: the model runs with fully dequantized BF16 weights, proper tgt_sizes handling, correct image-bound scatter reconstruction, and aligned padding for TT's 8-element DRAM alignment requirement. After all loader fixes, the remaining PCC gap is solely due to the FP32-precision loss in the vision encoder.

## Fix
Not attempted (Tier B). The fix belongs in tt-mlir's StableHLO→TTIR lowering: when `stablehlo.softmax` (or the equivalent reduce pattern) is lowered with a float32 compute type annotation, the lowered TTIR op must preserve float32 accumulation rather than clamping to the operand element type (BF16). This would require changes to the softmax lowering path in tt-mlir, impacting all models that rely on FP32 upcasting in attention softmax.

Loader-layer fixes committed to `remediation/minicpm_v_2_6_int4-pytorch-Default-single_device-inference` on `tt-forge-models`:

1. **Fix 1** (`modeling_minicpmv.py`): Add `self.post_init()` call in `MiniCPMV.__init__` to initialize `all_tied_weights_keys` (transformers 5.x).
2. **Fix 2** (`modeling_minicpmv.py`): Wrap `torch.vstack(tgt_sizes)` in an `if isinstance(tgt_sizes[0], torch.Tensor)` guard; add `else: torch.tensor(tgt_sizes)` for Python-list tgt_sizes.
3. **Fix 2b** (`modeling_minicpmv.py`): Add `.tolist()` after Fix 2's tensor creation to immediately convert back to a Python list.
4. **Fix 12** (`modeling_minicpmv.py`): Replace the Fix-2+2b block (which creates a traced tensor inside the Dynamo-compiled function) with a direct Python-list pass-through; eliminates `aten._local_scalar_dense` graph break risk.
5. **Fix 3** (`image_processing_minicpmv.py`): Guard `cast_tensor()` against non-Tensor inputs (int values in tgt_sizes cause TypeError).
6. **Fix 4** (`resampler.py`): Use element-access for bs=1 in `_adjust_pos_cache` max_h/max_w to avoid dynamic XLA scalar.
7. **Fix 4b** (`resampler.py`): Branch on `isinstance(tgt_sizes, (list, tuple))` for Python-list tgt_sizes in `_adjust_pos_cache`.
8. **Fix assert-list / Fix patch-len-list** (`resampler.py`): Guard `tgt_sizes.shape[0]` and `tgt_sizes[:, 0]` column-slice against Python list.
9. **Fix 5/5b** (`resampler.py`): Use Python `max()` for list `patch_len` in `max_patch_len` computation.
10. **Fix 6** (`resampler.py`): Align `max_patch_len` to next multiple of 8; change `key_padding_mask` to float (BF16, -inf for padding); explicitly pad `pos_embed` and `x` to `max_patch_len`.
11. **Fix 7 / Fix 10** (`modeling_minicpmv.py`): Align `max_patches` to next multiple of 8; use pure-Python arithmetic from `data['tgt_sizes']` to avoid traced TT tensor for `max_patches`.
12. **Fix 8** (`load_inputs()` in loader.py): Extract `tgt_sizes` as flat Python `[[h,w],…]` list; extract `image_bound` as Python int lists to prevent device-to-host errors.
13. **Fix 9** (`modeling_minicpmv.py`): Replace `scatter_`-on-view / `aten.scatter.src` with cat-based reconstruction of `vllm_embedding`.
14. **Fix 11** (`modeling_minicpmv.py`): Create `patch_attn_mask` on CPU (bool), fill positions, then `.to(device)` to avoid in-place bool assignment on TT tensor.
15. **BnB dequantize** (`loader.py`): Replace `bnb.nn.Linear4bit` modules with `nn.Linear` BF16 via `bnb.functional.dequantize_4bit`.

## Tier B justification
The root cause is cross-cutting: `ttmlir-f32-precision-not-preserved` affects every model that uses `nn.functional.softmax(..., dtype=torch.float32)` in attention. Fixing it requires changes to the softmax lowering path in tt-mlir that would impact the entire attention stack. This is not a scoped single-file change.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    344.78s (0:05:44)
- Tier A attempts: N/A

## Files changed
- `minicpm_v_2_6_int4/pytorch/loader.py` (remediation branch in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 294b6bebb2c68e374054d48eec94f17b5f297509 |
