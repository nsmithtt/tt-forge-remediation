# Remediation Summary: chameleon-pytorch-7B-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[chameleon/pytorch-7B-single_device-inference]

## Result
FAIL — Runtime OOM: cumsum allocates 17.4 GB output tensor (8.68B elements) that cannot fit alongside the 7B model on a single device; root cause of the wrong shape is unknown without IR dump tooling

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttnn-cumsum-output-size-exploding

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

Actual failure (reproduced):
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Root cause 1 (fixed): `AttributeError: module 'spacy' has no attribute 'Language'`
Root cause 2 (fixed): `loc("reduce.3194"): error: failed to legalize operation 'stablehlo.reduce'`
Root cause 3 (unfixed, Tier B):
```
Out of Memory: Not enough space to allocate 17364418560 B DRAM buffer across 8 banks,
where each bank needs to store 2170552320 B, but bank size is 4273390016 B
(allocated: 4002916096 B, free: 270473920 B, largest free block: 204611072 B)
  --- ttnn::prim::AccumulationDeviceOperation::create_output_tensors(...)
  --- ttnn::cumsum(...)
  --- tt::runtime::ttnn::operations::reduction::cumsum::run(tt::target::ttnn::CumSumOp const*, ...)
```

## Root cause

Three layered bugs were found, two of which were fixed:

**Bug 1 (loader, fixed):** `huspacy/pytorch/loader.py` imported `spacy` at module level. Because `tt_forge_models/spacy/` is a Spanish NLP model directory that becomes a Python namespace package when `models_root` is added to `sys.path` during test discovery, the fake namespace package is cached in `sys.modules["spacy"]`. Later, when HuggingFace's `datasets._dill.py` checks `if "spacy" in sys.modules: import spacy; if issubclass(obj_type, spacy.Language)`, it raises `AttributeError: module 'spacy' has no attribute 'Language'`. Fixed by moving `import spacy` inside `_load_nlp()`.

**Bug 2 (tt-mlir, fixed):** Chameleon's VQ-VAE visual encoder contains a `stablehlo.reduce` with an ArgMin body (compare LE → select → compare EQ → minimum → select → select, with `+inf` init value). The `StableHLOToTTIRReduceOpConversionPattern` had no ArgMin handler. Fixed by:
- Adding `POS_INF` to `TypicalInitReductionValue` enum
- Fixing the bf16 constant comparison to use `std::memcpy` (the previous sign-extension of a single `char` read `0x7F80 → 0xFF80` = NEG_INF)
- Adding `isArgMin` / `hasValidArgMinReducerBody` / `hasValidArgMinInitValues` detection
- Adding `matchAndRewriteInternalArgMin` to lower ArgMin as ArgMax(-x)

**Bug 3 (tt-mlir or tt-metal, unfixed):** After both fixes above, compilation succeeds but runtime fails with OOM. A `stablehlo.reduce_window` with SUM body is lowered to `ttir.CumSum` → `ttnn.cumsum`. At runtime, `ttnn::prim::AccumulationDeviceOperation::create_output_tensors` attempts to allocate a 17,364,418,560-byte (17.4 GB) output tensor. Since `output_shape = input.logical_shape()`, the cumsum's input must also be 8.68 billion elements—impossibly large for any legitimate activation in a 7B LM with 1,035-token input. The root cause (which `reduce_window` operation generates this wrong shape, and whether the `isCumSum` pattern match is too permissive) is unknown without dumping the MLIR IR at each pipeline stage.

## Fix

**Bug 1:** `tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py` — removed top-level `import spacy`, moved it inside `_load_nlp()`. Committed on branch `remediation/chameleon-pytorch-7B-single_device-inference` in tt_forge_models.

**Bug 2:** `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — added `POS_INF` enum value, fixed bf16 bit-read with `std::memcpy`, added ArgMin detection and `ArgMax(-x)` lowering. Committed on branch `remediation/chameleon-pytorch-7B-single_device-inference` in tt-mlir.

**Bug 3 (proposed, not implemented):** Identify the `stablehlo.reduce_window` that produces the 8.68B-element cumsum input by enabling MLIR IR export (`compile_options.export_path`). Likely either (a) the `isCumSum` pattern is too permissive and misidentifies a different reduce_window (global pool, average pool, etc.) as a prefix sum, creating a wrong output shape; or (b) a shape-propagation error in an earlier lowering pass inflates one dimension before the reduce_window. Fix location: `StableHLOToTTIRPatterns.cpp` (if over-eager pattern matching) or an earlier MLIR pass.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting | internal-error-unknown-mechanism

The cumsum output is 8.68 billion elements for a 7B model with 1,035-token input—clearly wrong. Diagnosing which `reduce_window` generates this shape and why requires IR dumps across all five MLIR pipeline stages (VHLO → SHLO → SHLO-frontend → SHLO-compiler → TTIR → TTNN). Without the IR, it is not possible to determine whether the `isCumSum` check is too permissive, whether a shape-propagation bug precedes the reduce_window, or whether the TTNN cumsum allocator itself contains a bug. The fix scope is unknown and the root cause has no known mechanism.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    81.07s (0:01:21) — test ran to runtime OOM after both loader and ArgMin fixes
- Tier A attempts: 1 (ArgMin lowering — compilation error resolved, runtime OOM revealed)

## Files changed
- `tt-xla/third_party/tt_forge_models/huspacy/pytorch/loader.py` — move `import spacy` inside `_load_nlp()`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — ArgMin lowering + POS_INF + bf16 fix

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 376a5d054b3f2a9d8f2fb482c3ce6ecad4381469 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 3ac989525dfd3f730e743d27d02f51c5bddabea5 |
