# Remediation Summary: gliner-pytorch-Multi_PII_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gliner/pytorch-Multi_PII_v1-single_device-inference]

## Result
FAIL — second compiler-stack bug (Tier B): torch_xla CPU-fallback partitioner cannot handle data-dependent shapes from `torch.where` in GLiNER

## Stack layer
tt-mlir, tt-xla

## Tier
B

## Bug fingerprint
torch-xla-cpu-fallback-xla-args-missing-data-dependent-shapes

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (fixed):
```
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:253: const T &llvm::ArrayRef<long>::operator[](size_t) const [T = long]: Assertion `Index < Length && "Invalid index!"' failed.
Fatal Python error: Aborted
```
GDB backtrace frame #7:
```
mlir::tt::ttir::(anonymous namespace)::SharedLHSMatmulFusion<mlir::tt::ttir::LinearOp>::matchAndRewrite(mlir::tt::ttir::LinearOp, mlir::PatternRewriter&)
```
in TTIRFusingPass::runOnOperation().

Second failure (unfixed):
```
AttributeError: 'fused_0' object has no attribute 'xla_args'
```
in `torch_xla/_dynamo/dynamo_bridge.py:348` inside `partition_fx_graph_for_cpu_fallback`, triggered by `torch.where(words_mask > 0)` at `gliner/modeling/base.py:285`.

## Root cause

**First bug (fixed — Tier A, tt-mlir):** `SharedLHSMatmulFusion<LinearOp>::matchAndRewrite` in `TTIRFusingPass` collected LinearOp candidates without checking whether each candidate's output rank matches the root op's output rank. `replaceWithSlices` then indexed into candidate shapes at `outputFusedDim = rootOutputRank - 1`, but a candidate with a lower output rank caused an out-of-bounds `ArrayRef::operator[]` assertion. The GLiNER model (DeBERTa-based encoder) produces LinearOps sharing the same LHS value but with mismatched output ranks, triggering the crash.

**Second bug (Tier B — tt-xla / torch_xla):** GLiNER's `get_representations` uses `batch_indices, word_idx = torch.where(words_mask > 0)` to find valid token positions. This produces a tensor with data-dependent size that cannot be compiled to a static XLA graph. The torch_xla `partition_fx_graph_for_cpu_fallback` path is invoked to run this subgraph on CPU. The `InputCollector.run()` step that is supposed to populate `xla_args` on each `fused_*` submodule does not succeed for this data-dependent operation. Consequently, when `extract_internal(fused_0)` calls `extract_graph_helper(xla_model, ...)`, `xla_model.xla_args` does not exist, raising `AttributeError`.

## Fix

**Applied (tt-mlir):** Added an output-rank equality guard in `SharedLHSMatmulFusion::collectCandidates` (`lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`). Before adding a candidate to the fusion list, the code now checks `candidateOutputType.getRank() == rootOutputRank`. Candidates with different output rank are skipped, preventing the out-of-bounds index in `replaceWithSlices`.

Commit: `77906f01c` on branch `remediation/gliner-pytorch-Multi_PII_v1-single_device-inference` in tt-mlir.

**Not applied (tt-xla):** The `torch.where` data-dependent shape issue requires the torch_xla CPU fallback partitioner to correctly propagate `xla_args` through submodules that contain data-dependent operations. The fix would require changes inside `torch_xla/_dynamo/dynamo_bridge.py` (InputCollector logic or the extract_internal/partition path), which is not in the tt-xla codebase.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

The `torch.where` operation produces variable-length output tensors (data-dependent shapes). The torch_xla CPU fallback partitioner (`partition_fx_graph_for_cpu_fallback`) does not have infrastructure to propagate `xla_args` through the `InputCollector` when data-dependent shapes are present. Fixing this requires new support in torch_xla's dynamo bridge, not a scoped one- or two-file change in the tt stack.

## Verification
- pytest exit: FAIL
- Hardware:    n300
- Duration:    135.63s (0:02:15)
- Tier A attempts: 1

## Files changed
- `tt-mlir/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` — added output-rank check in `SharedLHSMatmulFusion::collectCandidates`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550 |
| tt-mlir         | 77906f01c |
| tt-xla          | 94362e631 |
| tt-forge-models | 0f7b734348 |
