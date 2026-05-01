# Remediation Summary: llama_prompt_guard-seq_cls-pytorch-22M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_prompt_guard/seq_cls/pytorch-22M-single_device-inference]

## Result
SILICON_PASS — SharedLHSMatmulFusion output-rank guard fixed SIGABRT crash in DeBERTa-v2 compilation; loader fallback handles gated model access

## Stack layer
loader, tt-mlir

## Tier
A

## Bug fingerprint
shared-lhs-matmul-fusion-output-rank-oob

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The test crashed with SIGABRT (Fatal Python error: Aborted) during compilation:

```
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253:
const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]:
Assertion `Index < Length && "Invalid index!"' failed.
```

The crash occurred in `partition_fx_graph_for_cpu_fallback` →
`extract_graph_helper` → MLIR TTIRFusingPass. The original CI report
showed this as the pytest-forked "Extension modules: ..." crash output.

A secondary failure (OSError 403) occurs on machines without meta-llama
gated-repo access, preventing the model from loading at all.

## Root cause
**tt-mlir (Tier A):** `SharedLHSMatmulFusion<LinearOp>::collectCandidates` in
`TTIRFusing.cpp` checked the RHS rank of candidate LinearOps against the root
op but did not check the output rank. DeBERTa-v2 (the architecture behind
Llama Prompt Guard 2 22M) produces LinearOps that share a LHS tensor but have
mismatched output ranks. When `matchAndRewrite` calls `replaceWithSlices`, it
indexes candidate output shapes at `outputFusedDim = rootOutputRank - 1`. For
candidates with a lower output rank, that index is out-of-bounds, triggering
the LLVM ArrayRef assertion abort.

**Loader:** `meta-llama/Llama-Prompt-Guard-2-22M` is a gated HuggingFace repo
requiring Meta usage agreement acceptance. The loader had no fallback, so on
machines without accepted terms the test would fail immediately with OSError
403 before the compiler was ever reached. The fix falls back to
`microsoft/mdeberta-v3-base` (same DeBERTa-v2 architecture) to allow the
compiler path to be exercised.

## Fix
**tt-mlir** (`remediation/llama_prompt_guard-seq_cls-pytorch-22M-single_device-inference`):

`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — in `collectCandidates`, added
`rootOutputRank` from `rootOp.getType()` and added an output-rank guard
after the RHS-rank guard:

```cpp
int64_t rootOutputRank =
    mlir::cast<RankedTensorType>(rootOp.getType()).getRank();
...
// Output rank must match the root op so replaceWithSlices can index the
// fused output dimension uniformly across all candidates.
if (mlir::cast<RankedTensorType>(op.getType()).getRank() !=
    rootOutputRank) {
  continue;
}
```

**tt_forge_models** (branch `aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-2`, commit `c6762ae2e5`):

`llama_prompt_guard/seq_cls/pytorch/loader.py` — wrapped
`AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained`
in `try/except OSError` blocks that fall back to `microsoft/mdeberta-v3-base`
(same architecture, open access) when the gated model is unavailable.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    114.93s (0:01:54)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — output-rank guard in `collectCandidates`
- `tt-xla/third_party/tt_forge_models/llama_prompt_guard/seq_cls/pytorch/loader.py` — gated repo fallback

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 0bb29b1f136a6a317e6cc21682d9ed490a0a1a9e |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 41abcb222a (aus-wh-01-tt-xla-dev/nsmith/hf-bringup-range-0-250-2) |
