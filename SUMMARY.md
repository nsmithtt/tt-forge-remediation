# Remediation Summary: domain_classifier-sequence_classification-pytorch-nvidia_domain_classifier-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[domain_classifier/sequence_classification/pytorch-nvidia_domain_classifier-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
pytorch-hub-mixin-torch-dtype-kwarg, shared-lhs-matmul-fusion-mixed-rank-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two bugs, both needed to reach silicon:

1. `TypeError: DomainClassifierModel.__init__() got an unexpected keyword argument 'torch_dtype'`
2. `TypeError: DomainClassifierModel.forward() got an unexpected keyword argument 'token_type_ids'`
3. (after loader fixed) `Fatal Python error: python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion 'Index < Length && "Invalid index!"' failed.` — SIGABRT during TT device compilation

The third failure (SIGABRT) is the original reported failure — it produced the "Extension modules: ..." crash report.

## Root cause

**Loader (2 bugs):**

1. `DomainClassifierModel` uses `PyTorchModelHubMixin` (not `transformers.PreTrainedModel`). Its `from_pretrained` passes unrecognized kwargs directly into `__init__()`, which only accepts `config`. The loader was passing `torch_dtype` (intended for `PreTrainedModel.from_pretrained`) into `PyTorchModelHubMixin.from_pretrained`, which forwarded it to `__init__()`.

2. The DeBERTa-v3 tokenizer returns `input_ids`, `attention_mask`, and `token_type_ids`, but `DomainClassifierModel.forward()` only accepts `input_ids` and `attention_mask`. The raw tokenizer dict was returned as inputs without filtering.

**Compiler (tt-mlir):**

`SharedLHSMatmulFusion<LinearOp>` in `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — `collectCandidates` checked that all LHS-sharing LinearOps have the same RHS rank but NOT that their output ranks match the root op. DeBERTa-v3's disentangled attention mixes rank-2 and rank-3 projections that share the same LHS. `replaceWithSlices` then accesses `shape[outputFusedDim]` where `outputFusedDim = rootOutputRank-1 = 2` but a rank-2 candidate's shape has only indices 0 and 1, triggering an `ArrayRef` OOB assertion (SIGABRT). The fix (`bec72757a`) existed on a prior remediation branch but had not been cherry-picked to the current HEAD.

## Fix

**tt_forge_models** (`remediation/domain_classifier-sequence_classification-pytorch-nvidia_domain_classifier-single_device-inference`):

- `domain_classifier/sequence_classification/pytorch/loader.py`: Remove `torch_dtype` from `from_pretrained` kwargs; call `model.to(dtype_override)` after loading instead.
- `domain_classifier/sequence_classification/pytorch/loader.py`: Filter tokenizer output in `load_inputs` to only `input_ids` and `attention_mask`.

**tt-mlir** (`remediation/domain_classifier-sequence_classification-pytorch-nvidia_domain_classifier-single_device-inference`):

- `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`: Cherry-pick of `bec72757a` — add output-rank guard in `collectCandidates` so candidates with different output rank than the root op are skipped.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    107.43s
- Tier A attempts: 1

## Files changed
- `tt_forge_models/domain_classifier/sequence_classification/pytorch/loader.py` (2 commits)
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (cherry-pick of bec72757a)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 058664c9f8ac2addc1f130fb1fdf5c4fd72cc72f |
| tt-xla          | 03fdc2b12b88b33dc75d9cf5e76588198175fd2e |
| tt-forge-models | 8b4794d17ffb169cc892ecbfa17eec34f44dfc67 |
