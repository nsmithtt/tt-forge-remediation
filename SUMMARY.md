# Remediation Summary: chronos2-pytorch-Chronos_2_Synth-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chronos2/pytorch-Chronos_2_Synth-single_device-inference]

## Result
FAIL — TTNN regular SDPA produces pcc=nan for non-tile-aligned seq_len=34 in TimeSelfAttention

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttnn-sdpa-nonaligned-kv-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.

## Root cause

Six bugs were resolved to reach the final failure; the remaining blocker is in the tt-mlir / TTNN SDPA layer.

**Resolved bugs (loader layer):**
- `chronos-forecasting>=2.0` had `transformers<5` as a transitive dependency, which downgraded the global transformers installation and caused `torch._dynamo` guard evaluation to fail. Fix: move the package to `requirements.nodeps.txt` so it installs without pulling its transitive deps.

**Resolved bugs (tt-mlir layer):**
1. **SDPA verifier rejected broadcast mask dim[2]=1**: Chronos2's `_expand_and_invert_time_attention_mask` produces a mask with shape `(B,1,1,S)` where dim[2]=1 is a broadcast over all query positions. The `ScaledDotProductAttentionOp::verify()` in both `TTIROps.cpp` and `TTNNOps.cpp` required dim[2]==seqLen exactly. Relaxed to allow dim[2]==1.
2. **GatherToSliceRepeatConcat fired for maxIndex==0**: Cherry-picked upstream fix `0b70ff4a7`. When maxIndex==0 the pattern double-counts indices and produces a concat with wrong size.
3. **GatherToSliceRepeatConcat produced size-mismatched ConcatOp**: Added pre-flight check comparing concat size to gather output type; falls back to embedding pattern on mismatch.
4. **TTNN SDPA kernel rejected broadcast mask dim[2]=1 at runtime**: The TTNN SDPA kernel asserts `mask_shape[2] == q_shape[2]` at runtime. Added `expandMaskQueryDim` in `TTIRToTTNN.cpp` to expand dim[2] from 1 to seqLen via RepeatOp before lowering to `ttnn::ScaledDotProductAttentionOp`.
5. **shouldUseDecode routed to decode when kv_seq_len%32≠0**: GroupSelfAttention rearranges to `(time, batch, d)` making Q.seq_len=1, which triggered the decode path. The decode kernel requires kv_seq_len%32==0 but kv_seq_len=1 fails this. Added the `kvSeqLen % 32 == 0` guard to `shouldUseDecode`.

**Remaining Tier B bug:**
After all five tt-mlir fixes, the model compiles and runs on silicon but produces `pcc=nan`. Chronos2 uses two attention mechanisms:
- **TimeSelfAttention**: Q/K/V `(1, 12, 34, 64)` — seq_len=34, 34%32=2, non-tile-aligned
- **GroupSelfAttention**: Q/K/V `(34, 12, 1, 64)` — seq_len=1 (trivial, kv=1)

The 34-token sequence comes from 512/16=32 patches plus 2 special tokens (`use_reg_token=True` and a PAD token). The TTNN regular SDPA kernel cannot handle seq_len=34 correctly (non-tile-aligned). With 12 transformer blocks each having a TimeSelfAttention, the accumulated error produces a numerically degenerate model output (pcc=nan rather than a low but nonzero PCC).

This is the known Tier B bug `ttnn-sdpa-nonaligned-kv-pcc-wrong`. Fixing it requires padding K/V to the next multiple of 32 in the TTNN SDPA kernel and masking out the padded positions, which is a cross-cutting kernel change in tt-metal.

## Fix
**Proposed fix (not implemented):** Pad Q/K/V to the next tile boundary (multiple of 32) in the TTNN SDPA kernel when seq_len is not tile-aligned. Mask padded positions with -inf in the attention bias. This would be located in tt-metal's SDPA kernel and program factory. It is a cross-cutting change affecting all non-tile-aligned SDPA computations.

**Changes committed to remediation branches:**

*tt_forge_models (via tt-xla)*:
- `chronos2/pytorch/requirements.txt` — removed `chronos-forecasting>=2.0`; added comment explaining the nodeps approach
- `chronos2/pytorch/requirements.nodeps.txt` — added `chronos-forecasting>=2.0` (installed without transitive deps)

*tt-mlir*:
- `lib/Dialect/TTIR/IR/TTIROps.cpp` — allow dim[2]==1 in SDPA mask verifier
- `lib/Dialect/TTNN/IR/TTNNOps.cpp` — allow dim[2]==1 in SDPA mask verifier
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — GatherToSliceRepeatConcat: guard maxIndex==0 (cherry-pick 0b70ff4a7) + size mismatch pre-flight check
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` — `expandMaskQueryDim` to expand broadcast mask dim[2]; `shouldUseDecode` guard requiring `kvSeqLen % 32 == 0`

## Tier B justification
**Indicator:** cross-cutting

The non-tile-aligned SDPA fix requires changes to the TTNN SDPA kernel's memory layout, tiling strategy, and masking logic in tt-metal. The kernel handles many SDPA variants (prefill, decode, paged, flash-MLA); correctly padding and masking across all of them is a cross-cutting change that goes beyond a scoped one or two file fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    25.15s (model ran to completion but pcc=nan)
- Tier A attempts: N/A

## Files changed
*tt_forge_models*:
- `chronos2/pytorch/requirements.txt`
- `chronos2/pytorch/requirements.nodeps.txt` (new)

*tt-mlir*:
- `lib/Dialect/TTIR/IR/TTIROps.cpp`
- `lib/Dialect/TTNN/IR/TTNNOps.cpp`
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 6929e98f8c9e965bd2402aff7519da67e236046f |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c690ed30e47153a3fc746323a88eaec47e214a81 |
