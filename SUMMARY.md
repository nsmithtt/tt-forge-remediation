# Remediation Summary: llama_4_pytorch-4_Tiny_Random-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama/llama_4/pytorch-4_Tiny_Random-single_device-inference]

## Result
FAIL — PCC=0.183 < 0.99; root cause is Tier B ttnn-sdpa-nonaligned-kv-pcc-wrong (kv_len=164, 164%32=4)

## Stack layer
tt-xla, tt-mlir

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
Original reported error:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

After debugging, the actual compilation failures were:

1. `stablehlo.reduce_window` on `s64` (INT64) could not be legalized:
   ```
   error: 'func.func' op passing reduce-window.327 to ttnn failed
   ```
   Root cause: `masked_scatter` decomposition used `mask_f.long()` (INT64) for cumsum; INT64 `reduce_window` fails to lower in ttnn.

2. `stablehlo.reduce_window` on `i1` (bool) could not be legalized:
   ```
   error: 'func.func' op passing reduce-window.328 to ttnn failed
   ```
   Root cause: `checkInitValue` with `desired=NEG_INF` sets `desiredI1=false`; since `false == false`, a bool `false` init constant incorrectly matches `NEG_INF` (checked before `ZERO`) → misclassification. Additionally `isCumOR` pattern was missing entirely.

3. `'ttir.concat' op folder produced a value of incorrect type`:
   ```
   'tensor<1xsi32>', expected: 'tensor<1x1x1x1xsi32>'
   ```
   Root cause: `StableHLOGatherToSliceRepeatConcatPattern` created a single-input `ConcatOp` when the gather indices had no repeated leading or trailing values (`starts == 0, ends == 0` after decrement); `foldUnitConcatOp` returned the input with the wrong type.

Terminal failure after all three fixes:
```
PCC: 0.183  (required: 0.99)
```
Root cause: `ttnn-sdpa-nonaligned-kv-pcc-wrong` — SDPA with kv_len not divisible by 32 produces wrong outputs. Llama4 uses chunked attention (chunk_size=128) with seq_len=164 (576 vision patches → 144 image tokens + ~20 text tokens). Second chunk has kv_len=36 (36%32=4); full-attention layer has kv_len=164 (164%32=4).

## Root cause
Three sequential compiler bugs blocked Llama4 from compiling:

**Bug 1 (tt-xla, loader layer):** `masked_scatter` in `decompositions.py` used `mask_f.long()` for cumsum, producing an INT64 `stablehlo.reduce_window`. ttnn cannot lower INT64 reduce_window.

**Bug 2 (tt-mlir):** `checkInitValue()` in `StableHLOToTTIRPatterns.cpp` — the NEG_INF branch sets `desiredI1=false` and then checks `(initValueOp == false)`, which is true for the ZERO case (bool false). Since NEG_INF is checked first, bool `false` is misclassified as NEG_INF rather than ZERO. Furthermore, `isCumOR` did not exist: a `stablehlo.reduce_window` with an `OrOp` body was unhandled.

**Bug 3 (tt-mlir):** `StableHLOGatherToSliceRepeatConcatPattern` builds a `SmallVector<Value> slicesToConcat` with one entry per repeated leading/trailing region. When a gather's indices have no repeated regions, the vector contains only one element (the main slice). A single-operand `ConcatOp` is then folded by `foldUnitConcatOp` to its bare input, but the gather result type (e.g., `tensor<1x1x1x1xsi32>`) differs from the slice type (`tensor<1xsi32>`), crashing the verifier.

**Terminal bug (tt-metal, Tier B):** `ttnn-sdpa-nonaligned-kv-pcc-wrong` — SDPA produces numerically wrong results for K/V sequence lengths not divisible by 32. Llama4 with a 336×336 input image produces seq_len=164 (164%32=4). This is a known Tier B bug requiring changes to tt-metal's SDPA kernel padding/masking.

## Fix
**Fix 1 — tt-xla `python_package/tt_torch/backend/decompositions.py`:**
Changed `masked_scatter` decomposition to use `float32` instead of `int64` for the cumsum index:
```python
# Use float32 cumsum: ttnn does not support INT64 cumsum
mask_i = mask_f.float()
source_idx = (torch.cumsum(mask_i, 0) - 1).long()
```

**Fix 2 — tt-mlir `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:**
a) In `checkInitValue`, added early return for `i1` type in the `NEG_INF` branch to prevent bool `false` from matching `NEG_INF`.
b) Added `isCumOR` function that matches `reduce_window` with `OrOp`/`LogicalOrOp` body and ZERO init value.
c) Added `OrOp`/`LogicalOrOp` to the body-op whitelist.
d) Added `isCumOR` handler in `matchAndRewrite`: converts `i1` input to `bf16` via `TypecastOp`, applies `CumSumOp`, then applies `GreaterThanOp(result > 0.0)` to produce the bool output.

**Fix 3 — tt-mlir `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`:**
Added guard in `StableHLOGatherToSliceRepeatConcatPattern` after `starts--; ends--;`:
```cpp
if (starts <= 0 && ends <= 0) {
  return rewriter.notifyMatchFailure(
      srcOp,
      "No repeated leading/trailing indices; slice-repeat-concat pattern "
      "does not apply.");
}
```

## Tier B justification (FAIL with Tier=B only — omit otherwise)
`ttnn-sdpa-nonaligned-kv-pcc-wrong`

**Indicator:** cross-cutting — correctly handling non-tile-aligned SDPA requires changes to tt-metal's SDPA kernel to pad K/V to the next multiple of 32 and apply appropriate masking. This likely requires coordinated changes across the SDPA kernel, the program factory, and possibly the tt-mlir SDPA lowering.

The same bug is documented in memory (`sdpa_decode_constraint.md`) and affects any model with sequence lengths not divisible by 32.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    119.72s
- Tier A attempts: 3 (one per compiler bug fixed)

## Files changed
- `tt-xla/python_package/tt_torch/backend/decompositions.py` — float32 cumsum in masked_scatter
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — checkInitValue i1 fix, isCumOR function, gather no-repeated-indices guard

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 2985ff1d429a429210b1cd70c9493c6dd0ae8621 |
| tt-xla          | 90a626afa6fb824c6e4e6795682082361a18e5c3 |
| tt-forge-models | 7792f6a788d6a356d0f04f12d175ae58cff12881 |
