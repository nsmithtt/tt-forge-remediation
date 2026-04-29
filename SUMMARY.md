# Remediation Summary: amlm_hard/masked_lm/pytorch-amlm_hard-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[amlm_hard/masked_lm/pytorch-amlm_hard-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: loader tokenizer dispatch and tt-mlir SharedLHSMatmulFusion mixed-rank OOB

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
ttmlir-shared-lhs-matmul-fusion-mixed-rank-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 19:27:06.907 | critical |          Always | TT_FATAL: Chip 0 logical eth core (x=0,y=11) connects to a remote mmio device (assert.hpp:104)

(Current reproduction: KeyError: 0 in tokenizer loading, then SIGABRT from ArrayRef OOB in tt-mlir compilation)

## Root cause

Two bugs:

**Bug 1 — loader:** `AutoTokenizer.from_pretrained("leukas/amlm_hard")` dispatches to `DebertaV2TokenizerFast`, whose class attribute `cls.model = tokenizers.models.Unigram`. `convert_to_native_format` hits the `elif cls.model.__name__ == "Unigram":` branch and executes `vocab[0]` on the BPE vocab dict (string keys, no integer key 0) → `KeyError: 0`. The tokenizer JSON actually contains a BPE model, not Unigram. Fix: use `PreTrainedTokenizerFast.from_pretrained` directly (`cls.model = None`) to bypass the model-type check.

**Bug 2 — tt-mlir:** `SharedLHSMatmulFusion::collectCandidates` guards against RHS rank mismatch but not output rank mismatch. DeBERTa-v2 disentangled attention projections share the same LHS but some produce rank-3 outputs and others rank-2. `replaceWithSlices` computes `outputFusedDim = rootRank - 1` from the root op and accesses `shape[outputFusedDim]` on each candidate; when a candidate has lower rank the index exceeds the array length, triggering `ArrayRef<long>::operator[]` assertion failure (SIGABRT) inside `extract_graph_helper` during `_xla_warm_up_cache`.

## Fix

**Loader fix** (`tt_forge_models`, branch `remediation/amlm_hard-masked_lm-pytorch-amlm_hard-single_device-inference`):

File: `amlm_hard/masked_lm/pytorch/loader.py`
- Change `from transformers import AutoModelForMaskedLM, AutoTokenizer` → `from transformers import AutoModelForMaskedLM, PreTrainedTokenizerFast`
- Change `self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)` → `self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.model_name)`

**Compiler fix** (`tt-mlir`, branch `remediation/amlm_hard-masked_lm-pytorch-amlm_hard-single_device-inference`):

File: `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`, function `collectCandidates`
- Capture `rootOutputRank = mlir::cast<RankedTensorType>(rootOp.getType()).getRank()` before the candidate loop
- Add guard `if (mlir::cast<RankedTensorType>(op.getType()).getRank() != rootOutputRank) continue;` after the RHS rank check

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    83.67s
- Tier A attempts: 1

## Files changed
- `amlm_hard/masked_lm/pytorch/loader.py` (tt-forge-models)
- `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (tt-mlir)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 20c39036531012ac068169209a519e22d93c1abb |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | b7686b0bdc (branch: remediation/amlm_hard-masked_lm-pytorch-amlm_hard-single_device-inference) |
