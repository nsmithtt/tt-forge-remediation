# Remediation Summary: emu3/vision_tokenizer/pytorch-Default-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[emu3/vision_tokenizer/pytorch-Default-single_device-inference]

## Result
FAIL — second compiler-stack bug: ttnn::argmax CB over-allocation (2831488 B > 1572864 B L1) on 4096×32768 input tensor

## Stack layer
tt-metal

## Tier
A

## Bug fingerprint
ttnn-argmax-cb-l1-overflow

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure from ticket:
```
The image processor of type `Emu3VisionVQImageProcessor` is now loaded as a fast processor by
default, even if the model checkpoint was saved with a slow processor. This is a breaking change
and may produce slightly different outputs. To continue using the slow processor, instantiate this
class with `use_fast=False`.
```

This is a `logger.warning_once()`, not an exception. The actual silicon failure was:
```
loc("reduce.88"): error: failed to legalize operation 'stablehlo.reduce'
Failed to convert from SHLO to TTIR module
ValueError: Error code: 13
```

After the ArgMin lowering fix (Tier A), the test reached execution and hit:
```
TT_THROW @ program.cpp:1136:
Statically allocated circular buffers on core range [(x=0,y=0) - (x=3,y=3)] grow to
2831488 B which is beyond max L1 size of 1572864 B
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Three bugs were identified and two were fixed:

**Bug 1 (loader, fixed):** `AutoImageProcessor.from_pretrained` in
`emu3/vision_tokenizer/pytorch/loader.py` was missing `use_fast=False`, triggering a transformers
5.x warning about the fast processor default change.

**Bug 2 (tt-mlir, fixed, Tier A):** `StableHLOToTTIRReduceOpConversionPattern` handled ArgMax
but not ArgMin. Emu3's VQ quantizer finds nearest codebook entries by computing L2 distances then
running argmin via a 2-input `stablehlo.reduce` with LE/EQ/min reducer and +inf init value.
This reduce failed to legalize.

A sub-bug was also found and fixed: `checkInitValue()` read bf16 raw bits as `static_cast<uint16_t>(*data())`,
which takes only the first byte and sign-extends it. For +inf (0x7F80), the low byte is 0x80,
which sign-extends to 0xFF80 ≠ 0x7F80, so the POS_INF check always failed. Fixed with `memcpy`.

**Bug 3 (tt-metal, unfixed):** After the ArgMin fix, the argmax kernel running on the negated
4096×32768 bf16 tensor crashes with a static CB over-allocation: 2831488 B exceeds the 1572864 B
L1 per-core limit. This is in `ttnn::prim::ArgMaxDeviceOperation` → `validate_circular_buffer_region`.

## Fix

**Loader fix** (tt_forge_models, `emu3/vision_tokenizer/pytorch/loader.py`):
```python
self.image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name, trust_remote_code=True, use_fast=False
)
```

**tt-mlir fix** (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`):
1. Added `POS_INF` to `TypicalInitReductionValue` enum.
2. Fixed `checkInitValue` bf16 byte-reading: replaced `static_cast<uint16_t>(*data())` with
   `memcpy(&bfloatBits, data(), sizeof(uint16_t))`.
3. Added `matchAndRewriteInternalArgMin`: inserts `ttir.neg` before the input, then calls
   `ttir.argmax` (ArgMin(x) == ArgMax(-x) for finding the index).
4. Added `hasValidArgMinInitValues`, `hasValidArgMinReducerBody`, and `isArgMin` helpers
   following the existing ArgMax infrastructure.

**Proposed fix for Bug 3** (tt-metal, not attempted per one-fix-per-report rule):
The argmax CB size formula in the ArgMax kernel program factory likely computes page sizes
based on input tensor dimensions without capping to available L1. A Tier A-eligible fix
would clamp or tile the CB allocation to stay within 1.5 MB L1 for large inputs.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A — This is a Tier A fix (succeeded at its target) that uncovered a second compiler bug.
The second bug (argmax CB overflow) is not attempted per the one-fix-per-report rule.

## Verification
- pytest exit: FAIL (RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13)
- Hardware:    n150
- Duration:    149.18s (2:29)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (ArgMin lowering + bf16 fix)
- `tt-xla/third_party/tt_forge_models/emu3/vision_tokenizer/pytorch/loader.py` (use_fast=False)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | ada096f971861c7a05ee701a219ab86170e25b68 |
| tt-xla          | 02cee8fed1ff1f79adf5dbf33c64c968caf4a04b |
| tt-forge-models | 1f5f74111329507235558561ebc2f89c60a60ec5 |
