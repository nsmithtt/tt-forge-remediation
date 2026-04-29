# Remediation Summary: ai_image_detector_deploy-image_classification-pytorch-ai-image-detector-deploy-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[ai_image_detector_deploy/image_classification/pytorch-ai-image-detector-deploy-single_device-inference]

## Result
FAIL — spaCy namespace shadowing fixed (loader); SwinV2 inference takes 92+ seconds per pass on Blackhole P150B with incorrect output values (BF16 overflow), Tier B compiler-stack bug unfixed

## Stack layer
loader, tt-xla

## Tier
B

## Bug fingerprint
swinv2-blackhole-inference-timeout-incorrect-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
Two independent bugs found:

**Bug 1 (loader, fixed):** `setup_models_path` in `tests/runner/utils/dynamic_loader.py` called `sys.path.insert(0, models_root)`, inserting the tt_forge_models workspace root into `sys.path`. The directory `tt_forge_models/spacy/` exists without an `__init__.py`, so Python treats it as a namespace package named `spacy`, silently shadowing the real `spacy` library. When `datasets._dill` then calls `issubclass(obj_type, spacy.Language)`, it fails with `AttributeError: module 'spacy' has no attribute 'Language'`, crashing the dataset load.

**Bug 2 (tt-xla / tt-metal, unfixed, Tier B):** After the loader fix, the model compiles and runs, but each inference pass takes approximately 92 seconds on Blackhole P150B (vs. an expected 1–5 seconds). The model produces incorrect output values (`tensor([[-3.2634e+36, -4.5485e+36]])`) indicating BF16 overflow or a fundamentally wrong computation. The SwinV2-Large model generates ~36 MLIR subgraphs and compiles to 316 TT-metal programs. With 6 total device dispatch rounds (3 warmup + 2 perf + 1 final inference), the total runtime exceeds the CI timeout. The `TT_METAL_OPERATION_TIMEOUT_SECONDS=30` guard does not fire because `loop_and_wait_with_timeout` resets its clock whenever dispatch makes any progress—each of the 316 sequential programs is counted as progress. The root cause of the slow dispatch and incorrect output is unknown; likely candidates include: (a) inefficient lowering of `torch.roll` (used by SwinV2's shifted-window attention cyclic shift, no TTNN lowering found in tt-mlir), causing excessive graph breaks or CPU round-trips; (b) the relative position bias lookup (large gather operation) generating inefficient device kernels; or (c) a BF16 accumulation error in the windowed multi-head self-attention that compounds across the 22-layer SwinV2 stack.

## Fix
**Bug 1:** Removed 4 lines from `tests/runner/utils/dynamic_loader.py` in tt-xla:
```
-        # Add the models root to sys.path so relative imports work
-        if models_root not in sys.path:
-            sys.path.insert(0, models_root)
-
```
The `tt_forge_models` namespace package registration via `sys.modules` already handles relative imports within loaders, so removing the `sys.path.insert` does not break anything.

Branch: `remediation/ai_image_detector_deploy-image_classification-pytorch-ai-image-detector-deploy-single_device-inference` in tenstorrent/tt-xla

**Bug 2:** No fix attempted (Tier B). Proposed investigation: check whether `aten.roll` has a TTNN lowering in tt-mlir; if not, add one or file an issue for the missing lowering. Check whether the 92-second runtime is entirely in dispatch overhead (316 programs × ~290ms/program) or whether individual kernels are hanging. Also verify whether output overflow is caused by the wrong `torch.roll` behavior or a matmul precision issue.

## Tier B justification
`cross-cutting | internal-error-unknown-mechanism` — the 92-second per-inference slowdown and BF16 overflow span multiple possible causes (missing op lowering, dispatch overhead, precision error), all of which would require cross-file or cross-module investigation and likely coordinated changes across tt-mlir and tt-metal. The root cause is not pinpointed to a single named function.

## Verification
- pytest exit: TIMEOUT
- Hardware:    blackhole-p150b
- Duration:    N/A (test timed out; individual inference measured at ~92s/pass)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py` — removed `sys.path.insert(0, models_root)` block

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ea5e2d19dbb91b89c4ca409e806471cc8b6cdacf |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
