# Remediation Summary: dpt_swinv2_large-pytorch-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dpt_swinv2_large/pytorch-single_device-inference]

## Result
FAIL — hang in PJRT execution after model compilation; all 250 threads stuck in pthread_cond_wait on tt-metal thread_status condition variable

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-execution-hang-futex-wait

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The image processor of type `DPTImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`.

## Root cause
Two loader bugs were identified and fixed:

1. **Loader bug (use_fast)**: `AutoImageProcessor.from_pretrained()` in `_load_processor()` was missing `use_fast=False`, causing transformers 5.x to default to fast processor mode for `DPTImageProcessor`. This produces the reported warning/error.

2. **Loader bug (spacy collision)**: `load_dataset("huggingface/cats-image")` was called in `load_inputs()`. The `tt_forge_models/spacy/` directory shadows the real spacy package in sys.path, causing `datasets._dill` to crash at `issubclass(obj_type, spacy.Language)` with `AttributeError: module 'spacy' has no attribute 'Language'`.

After both loader bugs were fixed, the test progresses through device initialization and compiles 4 MLIR modules (t=75s, 75.5s, 79s, 89s). After the 4th module compilation completes (~t=89.8s), the test hangs indefinitely.

**Compiler-stack hang**: GDB inspection of the stuck process shows all 250 worker threads in `pthread_cond_wait(thread_status, ...)` (tt-metal worker pool, idle), while the main Python thread is blocked on `__futex_abstimed_wait_common64(futex_word=0x2f1a9bf8)`. The main thread is waiting for PJRT execution to complete but the completion signal is never delivered. `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` does not fire, indicating the hang is in the PJRT layer (above tt-metal hardware dispatch), not in a tt-metal operation itself.

The pattern is reproducible across two independent runs: both hang at exactly the same point (after 4 module compilations, main thread enters futex wait, all worker threads enter pthread_cond_wait).

## Fix
**Loader fixes applied** (`tt_forge_models/dpt_swinv2_large/pytorch/loader.py`):
- Added `use_fast=False` to `AutoImageProcessor.from_pretrained()` call
- Replaced `load_dataset("huggingface/cats-image")["test"][0]["image"]` with `PIL.Image.new("RGB", (384, 384))`

**Compiler-stack hang**: Proposed fix would be in `tt-xla/pjrt_implementation/` — specifically in the PJRT execution dispatch that sends compiled programs to the TT hardware and waits for completion. The deadlock suggests the completion callback or event signaling is not being delivered back to the calling thread. Investigation requires:
1. Identifying which specific PJRT function call hangs (PJRT_LoadedExecutable_Execute or associated Wait)
2. Determining whether the tt-metal worker threads are completing their work and whether the PJRT future/promise is being fulfilled
3. Checking for a missing signal/notify_one on the condition variable that the main thread waits on

## Tier B justification
**Indicator**: internal-error-unknown-mechanism

The root cause of the deadlock between the PJRT main thread and the tt-metal worker pool is unclear without deep C++ debugging of the PJRT execution path. The hang mechanism requires diagnosis first — there is no single obvious function or formula to fix.

## Verification
- pytest exit: FAIL (killed after >10 min hang)
- Hardware:    blackhole-p150b
- Duration:    ~90s compilation then hung indefinitely
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/dpt_swinv2_large/pytorch/loader.py` (loader fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c461a7b5e3b2fdb660e5240bfeb308ddba5bfa3d |
| tt-forge-models | 7976af5b24b8bbc21c8b1bc1456007afae188528 |
