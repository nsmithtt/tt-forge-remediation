# Remediation Summary: mobileone-pytorch-s0-apple-in1k

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mobileone/pytorch-S0_Apple_IN1K-single_device-inference]

## Result
FAIL — Loader bug fixed (load_dataset spacy/dill crash); residual Tier B TT BF16 precision issue (PCC 0.9837 < required 0.99)

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-bf16-precision-consteval-removed

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure:
```
AttributeError: module 'spacy' has no attribute 'Language'
```
After loader fix, residual failure:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9837437216783377. Required: pcc=0.99.
```

## Root cause
Two issues:

1. **Loader bug (fixed):** `mobileone/pytorch/loader.py` called `load_dataset("huggingface/cats-image", split="test")` in `load_inputs`. The `tt_forge_models/spacy/` namespace package pollutes `sys.modules` with an incomplete `spacy` module (no `Language` attribute). `load_dataset` triggers `dill` pickling which calls `issubclass(obj_type, spacy.Language)`, crashing with `AttributeError: module 'spacy' has no attribute 'Language'`. Fix: remove `load_dataset` call and pass `image=None` to `VisionPreprocessor.preprocess()`, which falls back to the default COCO image URL.

2. **Compiler precision issue (unfixed, Tier B):** After the loader fix, the model runs on TT silicon but achieves PCC 0.9837 vs the required 0.99. The BF16 CPU floor is 0.998 (measured: FP32 vs CPU BF16 on MobileOne-S0), so the 0.015 gap is TT-specific precision loss beyond BF16 accumulation. This is a known cross-cutting issue tracked at https://github.com/tenstorrent/tt-xla/issues/1242 ("Exposed by removal of consteval on host") which affects 27+ conv-heavy models in the test config.

## Fix
**Loader fix (applied):** Removed the `load_dataset` import and call from `tt_forge_models/mobileone/pytorch/loader.py`. The `load_inputs` method now passes `image=None` directly to `input_preprocess`, which delegates to `VisionPreprocessor` that uses the default COCO image URL fallback.

Branch: `remediation/mobileone-pytorch-s0-apple-in1k` in tt-forge-models repo.
Commit: `dc808f10b50aaa1dac831d37b2e4437a814f702c`

**Precision fix (proposed, not applied):** The TT BF16 precision regression (issue #1242, removal of consteval on host) needs to be resolved in the compiler stack. 27+ other conv-heavy models in the test config work around this by setting `required_pcc: 0.98` — but this is only valid if/when the upstream issue is resolved or confirmed as the TT hardware floor.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
cross-cutting: The TT BF16 precision issue (issue #1242, consteval removal) is cross-cutting — it affects 27+ models across the test suite and would require coordinated changes across the compiler stack to restore full-precision constant folding for weights.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    135.73s (0:02:15)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mobileone/pytorch/loader.py` — removed `load_dataset` import and call; use VisionPreprocessor default image instead

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | dc808f10b50aaa1dac831d37b2e4437a814f702c |
