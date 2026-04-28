# Remediation Summary: bit-image_classification-pytorch-bit-50-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[bit/image_classification/pytorch-bit-50-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: transformers 5.x loader breaking change and MLIR constant-aliasing in batch_norm_training lowering

## Stack layer
loader | tt-mlir

  - `loader` — BitImageProcessor use_fast=False for transformers 5.x
  - `tt-mlir` — StableHLO→TTIR batch_norm_training used ZerosOp/OnesOp for synthetic running stats, causing MLIR CSE to alias them with the scale/offset constants

## Tier
A

  - Tier A fix in tt-mlir: single-file change to StableHLOToTTIRPatterns.cpp, replaced ZerosOp/OnesOp with EmptyOp for the synthetic running_mean/running_variance, rebuilt, test passed.

## Bug fingerprint
stablehlo-batch-norm-running-stats-alias-scale-offset

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `BitImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

After fixing the loader: PCC comparison failed. Calculated: pcc=0.309289543901706. Required: pcc=0.99.

## Root cause

**Bug 1 (loader):** transformers 5.x changed the default for `AutoImageProcessor.from_pretrained` to load `BitImageProcessor` as the fast variant. The BiT loader in `bit/image_classification/pytorch/loader.py` did not pass `use_fast=False`, causing a deprecation warning that was treated as an error. Additionally, the `huspacy/pytorch/loader.py` imported `spacy` at module level; because `tt_forge_models/spacy/` is a namespace package on sys.path, this import resolved to the namespace package (not the real spacy library), polluting `sys.modules['spacy']` and causing `datasets._dill` to crash with `AttributeError: module 'spacy' has no attribute 'Language'` during dataset loading.

**Bug 2 (tt-mlir):** In `StableHLOToTTIRPatterns.cpp`, the `StableHLOToBatchNormTrainingOpConversionPattern` creates synthetic `running_mean` and `running_variance` buffers (not present in StableHLO's `batch_norm_training` signature) to satisfy the TTIR op. These were created as `ttir::ZerosOp` and `ttir::OnesOp` respectively — exactly the same op types and values as the `scale` (ones[C]) and `offset` (zeros[C]) inputs from StableHLO. MLIR's CSE pass deduplicated them, making `running_mean` alias `offset` and `running_variance` alias `scale` at the SSA level.

At runtime, `ttnn::batch_norm` (training mode) calls `prim::running_statistics` with `momentum=1.0` which fully overwrites `running_mean←batch_mean` and `running_variance←batch_var` in-place. Because of the aliasing, this also overwrites the `bias` (offset) and `weight` (scale) buffers. The subsequent `prim::batch_norm` call then sees `weight=batch_var` and `bias=batch_mean` instead of ones/zeros, producing wrong output. For BiT's weight standardization (49 conv layers all using this path), this gives PCC≈0.309.

## Fix

**Fix 1 (loader):** Added `use_fast=False` to `AutoImageProcessor.from_pretrained()` in `tt-forge-models/bit/image_classification/pytorch/loader.py`. Also moved the top-level `import spacy` inside `_load_nlp()` in `huspacy/pytorch/loader.py` using `importlib.util.find_spec` to detect and reject the namespace package before importing, preventing sys.modules pollution. Both committed to `remediation/bit-image_classification-pytorch-bit-50-single_device-inference` branch of tt-forge-models.

**Fix 2 (tt-mlir):** In `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (lines 831–834), replaced:
```cpp
auto runningMean = rewriter.create<ttir::ZerosOp>(loc, meanType, ...);
auto runningVariance = rewriter.create<ttir::OnesOp>(loc, varianceType, ...);
```
with:
```cpp
auto runningMean = rewriter.create<ttir::EmptyOp>(loc, meanType.getShape(), meanType.getElementType());
auto runningVariance = rewriter.create<ttir::EmptyOp>(loc, varianceType.getShape(), varianceType.getElementType());
```
`EmptyOp` declares `MemoryEffectsOpInterface` (allocation side effect), so MLIR CSE will not deduplicate it with the pure ZerosOp/OnesOp scale/offset constants. With `momentum=1.0` the initial values of the running stats are fully overwritten before use, so uninitialized buffers are semantically correct. Committed to `remediation/bit-image_classification-pytorch-bit-50-single_device-inference` branch of tt-mlir.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    53.94s
- Tier A attempts: 1

## Files changed
- `tt-forge-models/bit/image_classification/pytorch/loader.py` — `use_fast=False` for BitImageProcessor
- `tt-forge-models/huspacy/pytorch/loader.py` — guard against spacy namespace package collision
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — EmptyOp for running_mean/running_variance

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0f5c906e56dc16b8d4c76f9053c9de04a7f1b2f8 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 536cc63f87e36d2cf7792b70644b745cd200db6c |
