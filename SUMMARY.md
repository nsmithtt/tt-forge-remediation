# Remediation Summary: mdeberta-zero_shot_cls-pytorch-Xenova_mDeBERTa_V3_Base_Xnli_Multilingual_Nli_2mil7-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mdeberta/zero_shot_cls/pytorch-Xenova_mDeBERTa_V3_Base_Xnli_Multilingual_Nli_2mil7-single_device-inference]

## Result
SILICON_PASS — loader model-ID fix + SharedLHSMatmulFusion output-rank guard

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
xenova-onnx-only-repo-wrong-model-id, ttir-shared-lhs-matmul-fusion-mixed-rank-output-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
OSError: Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 does not appear to have a file named pytorch_model.bin or model.safetensors.
```
(The trailing `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a Python SWIG extension warning unrelated to the failure.)

After fixing the loader, the test hit a second failure:
```
python3: .../llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
```
in `partition_fx_graph_for_cpu_fallback` → `extract_graph_helper`.

## Root cause

**Bug 1 — loader:** The HuggingFace user `Xenova/` distributes only ONNX files for use with Transformers.js; the repo has no `pytorch_model.bin` or `model.safetensors`. The PyTorch weights are hosted at `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.

**Bug 2 — tt-mlir:** `SharedLHSMatmulFusion<LinearOp>` in `TTIRFusing.cpp` collects candidates that share the same LHS but only guards on RHS-rank equality, not output-rank equality. DeBERTa-v3's disentangled attention produces LinearOps with both rank-2 and rank-3 outputs that share an LHS. `replaceWithSlices` computes `outputFusedDim = rootOutputRank - 1 = 2` and accesses `shape[2]` on a rank-2 candidate's shape array (length 2), triggering an ArrayRef out-of-bounds assertion.

## Fix

**Loader fix** (`tt-forge-models`, branch `remediation/mdeberta-zero_shot_cls-pytorch-Xenova_mDeBERTa_V3_Base_Xnli_Multilingual_Nli_2mil7-single_device-inference`):
- `mdeberta/zero_shot_cls/pytorch/loader.py`: changed `pretrained_model_name` from `Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` to `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.

**Compiler fix** (`tt-mlir`, branch `remediation/mdeberta-zero_shot_cls-pytorch-Xenova_mDeBERTa_V3_Base_Xnli_Multilingual_Nli_2mil7-single_device-inference`, commit `fecdef6ad`):
- `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`: cherry-picked `bec72757a` — added output-rank guard in `collectCandidates` so LinearOp candidates whose result rank differs from the root op are excluded from the fusion group rather than triggering a crash.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    112.40s
- Tier A attempts: 1

## Files changed
- `tt-forge-models`: `mdeberta/zero_shot_cls/pytorch/loader.py`
- `tt-mlir`: `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | fecdef6addffcc6374ce4a19a2ea691000668806 |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | aae2f4556b85ebea302de74770ed58b12644d8fb |
