# Remediation Summary: deberta-token_classification-hugogiddins-ticker_multi_fine_tune_v4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deberta/token_classification/pytorch-HugoGiddins/ticker_multi_fine_tune_v4-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
transformers-5x-invalid-problem_type-config-field, shared-lhs-matmul-fusion-mixed-rank

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Aborted

python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.

The original reported failure was the extension-modules crash dump from a Python SIGABRT, which is the symptom of this assertion failure during compilation.

Before the compiler bug could be hit, there was also a loader bug:
  ValueError: The config parameter `problem_type` was not understood: received token-classification but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid.

## Root cause

Two bugs chained:

**Bug 1 — Loader (transformers 5.x strict validation):**
`HugoGiddins/ticker_multi_fine_tune_v4` stores `problem_type: "token-classification"` in its
`config.json`. In transformers 5.x, `PreTrainedConfig.__init__` now validates `problem_type`
against a strict allowlist (`"regression"`, `"single_label_classification"`,
`"multi_label_classification"`). The invalid value raises `ValueError` before model weights
are loaded. Fix: load the raw config dict with `PreTrainedConfig.get_config_dict`, strip
the invalid field, rebuild the config with `AutoConfig.for_model`, and pass it explicitly.

**Bug 2 — tt-mlir TTIRFusing.cpp (SharedLHSMatmulFusion mixed-rank OOB):**
`SharedLHSMatmulFusion<LinearOp>::collectCandidates` in
`tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` validated that all fusion candidates
share the same RHS rank, but NOT the same output rank. DeBERTa-v2's disentangled attention
mixes rank-3 attention projections (input [1,128,384]) with rank-2 ops that share the same
LHS. The fused-output-dimension index (`rootOutputRank - 1 = 2`) exceeds the rank of the
rank-2 candidate's shape, causing an `ArrayRef out-of-bounds` assertion (SIGABRT) inside
`replaceWithSlices`. This is the exact same bug previously fixed for the DeBERTa-v3 xsmall
test (commit bec72757a); the fix was cherry-picked onto the current HEAD.

## Fix

**Loader fix** (`tt_forge_models` repo,
`remediation/deberta-token_classification-hugogiddins-ticker_multi_fine_tune_v4` branch):
- `deberta/token_classification/pytorch/loader.py`: Added `AutoConfig`, `PreTrainedConfig`
  imports; in `load_model`, call `PreTrainedConfig.get_config_dict(self.model_name)`,
  strip invalid `problem_type` if present, rebuild config via `AutoConfig.for_model(**config_dict)`,
  and pass config explicitly to both `AutoTokenizer.from_pretrained` and
  `AutoModelForTokenClassification.from_pretrained`.

**Compiler fix** (`tt-mlir` repo,
`remediation/deberta-token_classification-hugogiddins-ticker_multi_fine_tune_v4` branch):
- Cherry-picked commit `bec72757a` ("[TTIRFusing] Guard SharedLHSMatmulFusion against
  mixed-rank outputs") onto current HEAD (`553c0632b`). New commit: `c7833d572`.
- `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`: Added output-rank guard in
  `collectCandidates` — skip any candidate whose result rank differs from the root op's
  result rank, preventing the OOB index in `replaceWithSlices`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    61.94s (0:01:01)
- Tier A attempts: 1

## Files changed
- `third_party/tt_forge_models/deberta/token_classification/pytorch/loader.py` (loader fix)
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (Tier A compiler fix, cherry-pick of bec72757a)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | c7833d572b114c0c7610b1bbd5111a0d4258c5b7 |
| tt-xla          | e8a314af9229dc1a864966d377dd6b9f4d4f625b |
| tt-forge-models | 95c2ada62d2eff787d4659d671ecf48ee52114f8 |
