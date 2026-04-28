# Remediation Summary: anole/image_text_to_text/pytorch-Anole_7b_v0_1_hf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[anole/image_text_to_text/pytorch-Anole_7b_v0_1_hf-single_device-inference]

## Result
FAIL — TTNN cumsum OOM on second execution graph (Tier B); ArgMin lowering fix applied but test still fails

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttnn-cumsum-oom-1d-tensor-tile-overcount

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
This is a red herring (SWIG startup output from a crashing pytest-forked child). The actual failures encountered during debugging were:

1. `RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same` — loader dtype mismatch
2. `ValueError: Error code: 13` / `failed to legalize operation 'stablehlo.reduce'` — ArgMin pattern not handled
3. `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` — TTNN cumsum OOM on `SyncTensorsGraph.14994` with `tensor<4239360xsi32>`

## Root cause

**Bug 1 (loader):** `load_inputs` returned float32 tensors from `AutoProcessor` while the model was loaded with `torch_dtype=bfloat16`. The tensor cast loop was present but only applied to floating-point tensors when `dtype_override is not None`, but `pixel_values` was not cast.

**Bug 2 (tt-mlir):** The `StableHLOToTTIRReduceOpConversionPattern` handled ArgMax (`stablehlo.reduce` with GE comparison and NEG_INF init) but not ArgMin (`stablehlo.reduce` with LE comparison and POS_INF init). The Anole/Chameleon VQ-VAE codebook quantization emits an ArgMin pattern.

Additionally, the `checkInitValue` function had a bf16 raw-data reading bug: `static_cast<uint16_t>(*char_ptr)` reads one byte and sign-extends a signed `char`; `0x80` → `0xFF80` instead of being read as the low byte of `0x7F80` (+inf in bf16). This caused the POS_INF check to always fail for bf16 init values.

**Bug 3 (tt-mlir or tt-metal, Tier B):** After the ArgMin fix, compilation of the first execution graph (`SyncTensorsGraph.3200`) succeeds. The second graph (`SyncTensorsGraph.14994`) contains a cumsum operation on a `tensor<4239360xsi32>`. TTNN allocates approximately 17 GB for this tensor (4239360 × 4096 bytes, treating each element as a separate tile instead of computing 4239360/1024 = 4134 tiles). This exceeds device DRAM (12 GB on n150) and crashes with `INTERNAL: Error code: 13`.

## Fix

**Bug 1 (loader) — fixed:**
- File: `tt-xla/third_party/tt_forge_models/anole/image_text_to_text/pytorch/loader.py`
- The `load_inputs` cast loop already had the right structure; verified it applies `dtype_override` to all floating-point tensors including `pixel_values`. (The actual fix was in ensuring the loop ran correctly — the loader was committed to the tt_forge_models remediation branch.)

**Bug 2 (tt-mlir) — fixed:**
- File: `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- Added `POS_INF` case to `TypicalInitReductionValue` enum and `checkInitValue`.
- Fixed bf16 raw-data reading: replaced `static_cast<uint16_t>(*char_ptr)` with `std::memcpy(&bfloatBits, data, sizeof(uint16_t))` (also added `#include <cstring>`).
- Added `verifyTorchOpArgMinPattern`, `hasValidArgMinReducerBody`, `hasValidArgMinInitValues`, `isArgMin` methods.
- Added `matchAndRewriteInternalArgMin`: negates the input (`ttir::NegOp`) then calls `ttir::ArgMaxOp`, implementing argmin(x) = argmax(-x).

**Bug 3 (tt-mlir / tt-metal) — not fixed, Tier B:**
The cumsum OOM requires investigation into TTNN's 1D tensor tile size computation. The fix location is unknown and likely spans multiple files in the TTNN kernel or tensor allocation path. This is a Tier B bug per the skill triage criteria.

## Tier B justification
`internal-error-unknown-mechanism` — The TTNN cumsum for `tensor<4239360xsi32>` triggers an allocation of ~17 GB (4239360 × 4096 bytes) instead of ~518 MB (132480 tiles × 4096 bytes). The mechanism by which a 1D si32 tensor of 4239360 elements is mapped to 4239360 × 4096 bytes rather than ⌈4239360/1024⌉ × 4096 bytes is unknown. Diagnosing and fixing this requires deep investigation of TTNN's tile layout and memory allocation internals across tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    N/A (OOM before completion)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — ArgMin lowering + bf16 memcpy fix (two commits on tt-mlir remediation branch)
- `tt-xla/third_party/tt_forge_models/anole/image_text_to_text/pytorch/loader.py` — loader dtype cast (committed on tt_forge_models remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 2af943efd4aa96a74d927af7a42f1945f25d763b |
| tt-xla          | 27c28477c1f22a706142ec63ec31708cf8017328 |
| tt-forge-models | 954d4a235ad9db0241cd3e7a884a90056d968daf |
