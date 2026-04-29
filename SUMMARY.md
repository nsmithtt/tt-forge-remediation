# Remediation Summary: airealnet-image_classification-pytorch-AIRealNet-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[airealnet/image_classification/pytorch-AIRealNet-single_device-inference]

## Result
FAIL — SwinV2ForImageClassification hangs silently after PJRT compilation on device

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
swinv2-device-exec-hang-after-pjrt-compile

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

Actual failure discovered on reproduction:
```
AttributeError: module 'spacy' has no attribute 'Language'
```
at `datasets/utils/_dill.py:42` inside `load_dataset("huggingface/cats-image")` in the airealnet loader.

After fixing the loader (replacing `load_dataset` with `PIL.Image.new` and fixing the huspacy
top-level spacy import), the test gets past the original error and hangs silently after PJRT
compilation. Two independent runs both hung for 1+ hours with CPU dropping from ~65% (compilation)
to ~3% (inference hang), with no error output after the last module_builder.cc log line.

## Root cause

**Loader bug (fixed):** The `tt_forge_models/spacy/` model directory creates a Python namespace
package named `spacy` when `models_root` is added to `sys.path` by `dynamic_loader.py`. During
test collection, `huspacy/pytorch/loader.py` imports `spacy` at module level, which registers
this fake namespace package in `sys.modules`. When the airealnet loader then calls
`load_dataset("huggingface/cats-image")`, the `datasets._dill` pickler sees `"spacy" in
sys.modules`, tries to access `spacy.Language`, and raises `AttributeError`.

**Device execution hang (not fixed — Tier B):** After the loader fix, the test proceeds through:
1. Input generation via PIL (fast)
2. Model loading: `Modotte/AIRealNet` (Swinv2ForImageClassification, 780 MB, 473 params in BF16)
3. PJRT compilation: ~37 seconds, 4 compilation passes each reporting
   `Failed to deserialize executable: UNIMPLEMENTED`
4. **Silent hang** during device inference execution

The last log entries are `module_builder.cc` warnings at elapsed 36.9 s; the process then
holds 130 threads in `futex_do_wait` indefinitely. `TT_METAL_OPERATION_TIMEOUT_SECONDS=30`
does not trigger, indicating the hang is above the tt-metal operation dispatch layer — likely
in the PJRT execution path between the compiled artifact dispatch and the device→host result
transfer. After 1+ hours, CPU is ~3% (consistent with threads blocked on device response).
The model architecture (SwinV2-L: depths=[2,2,18,2], embed_dim=192, window_size=16,
image_size=256) involves intensive window-attention with large activation tensors that
may be exposing a hang in the PJRT dispatch or device execution infrastructure.

## Fix

**Loader fix committed** to `tt_forge_models` branch
`remediation/airealnet-image_classification-pytorch-AIRealNet-single_device-inference`:

1. `huspacy/pytorch/loader.py`: Moved `import spacy` from module level inside `_load_nlp()`
   to prevent the `tt_forge_models/spacy/` namespace package from being added to `sys.modules`
   during test collection (cherry-picked from `nsmith/fix-align-spacy-namespace`).

2. `airealnet/image_classification/pytorch/loader.py`: Replaced
   `load_dataset("huggingface/cats-image")["test"]` with `PIL.Image.new("RGB", (224, 224))`
   to make the loader independent of the spacy namespace conflict (following the same pattern
   applied to `align/pytorch/loader.py` in the existing fix branch).

**Compiler-stack hang not fixed** — Tier B diagnosis required.

Proposed fix direction: investigate the PJRT execution path in `tt-xla` for SwinV2 graphs
with large window-attention tensors. Check whether `pjrt_computation_client.cpp` properly
handles the async result-wait for large graph outputs, and whether the device execution
dispatcher has a timeout mechanism that covers the post-compilation inference phase.

## Tier B justification

Indicator: `internal-error-unknown-mechanism`

The hang produces no error message and no stack trace. The process enters a silent
`futex_do_wait` state with 130 threads after a successful PJRT compilation of the SwinV2
graph. `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` does not trigger, meaning the hang is not
caught by the tt-metal operation timeout machinery. Without device-side debugger access or
PJRT execution tracing, the exact hang location cannot be determined from the available logs.
Two independent hour-plus runs confirm the hang is 100% reproducible at the same phase.

## Verification
- pytest exit: TIMEOUT (test killed after 1+ hour — two independent runs)
- Hardware:    n150 (4× Wormhole n150, BOARD_ID_HIGH=0x461)
- Duration:    >3600s (hung; killed manually)
- Tier A attempts: N/A

## Files changed
**tt_forge_models** (`remediation/airealnet-image_classification-pytorch-AIRealNet-single_device-inference`):
- `huspacy/pytorch/loader.py` — lazy spacy import inside `_load_nlp()`
- `airealnet/image_classification/pytorch/loader.py` — replace `load_dataset` with `PIL.Image.new`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | de4e4cf45bc1f939844cc23d1a1c35c9b0940ce4 |
