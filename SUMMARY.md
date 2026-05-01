# Remediation Summary: layoutxlm-pytorch-base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[layoutxlm/pytorch-Base-single_device-inference]

## Result
FAIL — TT vs CPU BF16 PCC gap (0.9855 measured, 0.99 required) is not explained by BF16 quantization (CPU BF16 baseline = 0.9985); compile-path bugs fixed but PCC gate fails

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-layoutxlmv2

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

After loader fixes, the compilation failure:
```
loc("gather.3950"): error: 'ttir.concat' op Output tensor dimension 0 does not match
the sum of input tensor dimensions: 512 vs. 1023.
```

After both loader and MLIR fixes, the terminal failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=0.9854967292178881. Required: pcc=0.99.
```

## Root cause

### Loader bugs (fixed)

1. **detectron2 not installed**: No `requirements.txt` for the layoutxlm loader; detectron2
   (needed by `LayoutLMv2Model`) was absent, causing `ImportError`.

2. **`is_detectron2_available()` lru_cache stale**: transformers caches the detectron2 check
   at module-import time (test collection), before the requirements manager installs detectron2.
   Even after clearing the cache via `cache_clear()`, the module-level conditional
   `if is_detectron2_available(): import detectron2` had already executed with False, leaving
   `detectron2` and `META_ARCH_REGISTRY` unbound in the `modeling_layoutlmv2` and
   `configuration_layoutlmv2` module namespaces.

3. **Module namespace patching needed**: After clearing the cache, all already-loaded
   layoutlmv2/layoutxlm sys.modules entries must have `detectron2` and `META_ARCH_REGISTRY`
   injected directly, plus `FrozenBatchNorm2d._load_from_state_dict` monkey-patched to the
   base Module implementation.

### MLIR compiler bug (fixed, Tier A)

`StableHLOGatherToSliceRepeatConcatPattern` in `StableHLOToTTIRPatterns.cpp` had two bugs:

**Bug 1 (RepeatOp shape, line 5257):** `repeatShape[indexedDim] = numberOfRepeats` discarded
`sliceShape[indexedDim]` (the sliceSize). `ttir.RepeatOp` semantics require
`output[d] = input[d] * repeatDims[d]`, so the correct formula is
`sliceShape[indexedDim] * numberOfRepeats`.

**Bug 2 (starts/ends double-counting, line 5179):** When `maxIndex == 0`
(i.e., `inputShape[indexedDim] == sliceSize` — only one valid gather index exists),
every index satisfies both `index == 0` AND `index == maxIndex`, so both `starts` and `ends`
were incremented for each of the N identical indices. After the decrement, this produced
`starts = N-1, ends = N-1`. The resulting concat had
`(N-1)*sliceSize + 1 + (N-1)*sliceSize = 2N-1` rows instead of `N`. The fix: guard
`ends++` with `maxIndex != 0`.

The specific gather triggering bug 2 in LayoutXLMv2 is an embedding lookup where
`inputShape[indexedDim] == sliceSize` (all 512 token indices map to the same embedding row).

### Precision failure (unfixed, Tier B)

After the compile-path fixes, the model runs end-to-end but achieves:
- TT BF16 vs CPU FP32 PCC: 0.9855
- CPU BF16 vs CPU FP32 PCC: 0.9985 (measured directly)

The 0.013 gap between TT and CPU BF16 is not explained by BF16 quantization alone. The
root cause is unknown — likely accumulated numerical differences in the text encoder (12 BERT
layers) and/or the visual backbone (detectron2 ResNet-101 + FPN) on TT hardware. Fixing this
would require layer-by-layer precision investigation across a complex multimodal architecture.

## Fix

### Loader (tt_forge_models, branch remediation/layoutxlm-pytorch-base-single_device-inference)

- `layoutxlm/pytorch/requirements.txt` (NEW): fvcore, cloudpickle, hydra-core, omegaconf
  (detectron2 runtime dependencies)
- `layoutxlm/pytorch/requirements.nodeps.nobuildisolation.txt` (NEW):
  `git+https://github.com/facebookresearch/detectron2.git` (must be built from source with
  `--no-build-isolation`)
- `layoutxlm/pytorch/loader.py` (MODIFIED): `load_model()` now calls
  `is_detectron2_available.cache_clear()`, then patches `detectron2` and `META_ARCH_REGISTRY`
  into all layoutlmv2/layoutxlm sys.modules entries, plus monkey-patches
  `FrozenBatchNorm2d._load_from_state_dict`

### MLIR (tt-mlir, branch remediation/layoutxlm-pytorch-base-single_device-inference)

- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (MODIFIED):
  - Line 5257: `repeatShape[indexedDim] = sliceShape[indexedDim] * numberOfRepeats;`
  - Line 5179: `if (index == maxIndex && maxIndex != 0) { ends++; }`

### Proposed fix for precision gap

Investigate layer-by-layer numerical differences in LayoutXLMv2 on TT. Likely candidates:
- LayerNorm / attention softmax precision in the text encoder
- Conv + BatchNorm precision in the detectron2 visual backbone
- Accumulation order in tile-based matmul vs. CPU sequential matmul

## Tier B justification

Indicator: **cross-cutting** — improving numerical precision across a 12-layer BERT text
encoder plus a ResNet-101 visual backbone running on TT hardware would require investigating
and potentially adjusting precision-sensitive operations (matmul accumulation, softmax,
LayerNorm) across multiple lowering passes.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    185.76s (0:03:05)
- Tier A attempts: 1 (the MLIR fix — applied, correct, but revealed PCC failure)

## Files changed

### tt_forge_models
- `layoutxlm/pytorch/requirements.txt` (new)
- `layoutxlm/pytorch/requirements.nodeps.nobuildisolation.txt` (new)
- `layoutxlm/pytorch/loader.py` (modified)

### tt-mlir
- `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` (modified)

### tt-xla
- `third_party/tt_forge_models` (submodule pointer bumped)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 26df2840e6145933c3e662a08ef550c2bb9eb11e |
| tt-xla          | 6ecf15460b120ef07741a98a95a68004570283bb |
| tt-forge-models | fe3c591292ed713e554183f82d5e9f6d3ea77a56 |
