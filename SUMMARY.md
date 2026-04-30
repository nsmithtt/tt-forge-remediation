# Remediation Summary: esmfold/pytorch-facebook/esmfold_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[esmfold/pytorch-facebook/esmfold_v1-single_device-inference]

## Result
FAIL — INTERNAL: Error code: 13 (kInternal) during tt-mlir compilation of EsmFoldStructureModuleTransition subgraph; root cause unknown

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttmlir-compilation-internal-error-unknown-mechanism

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Raised at `tt-xla/pjrt_implementation/src/api/module_builder/module_builder.cc` inside
`extract_compiled_graph`, which wraps the tt-mlir pipeline and returns `kInternal` (= 13)
when any internal compilation step fails.

## Root cause
ESMFold's `EsmFoldStructureModule.forward` loops 8 times, and at the end of each iteration
calls `rigids.stop_rot_gradient()` which calls `.detach()`.  `detach()` is a graph break for
`torch.compile` / dynamo, so each of the 8 iterations is compiled as a separate subgraph.
The later subgraphs include `EsmFoldStructureModuleTransition` (a simple block: 3 linear
layers + ReLU + residual + LayerNorm on a `[1, 21, 384]` tensor).

After ~25 minutes of successfully compiling the earlier subgraphs, the compilation of the
`EsmFoldStructureModuleTransition` subgraph inside the tt-mlir pipeline returns `kInternal`
(status code 13) after approximately 397 seconds of silent C++ processing.  No `LOG_F(ERROR,
...)` message from the module builder is captured even with `TTXLA_LOGGER_LEVEL=DEBUG`, which
means the failure occurs inside a lower-level pass (likely TTIR→TTNN lowering or the TTNN
program factory) without surfacing an error string through the PJRT log channel.

The exact failing pass was not identifiable from the debug log.  The `scatter_reduce_type =
<invalid>` warnings visible in the log are legitimate (they indicate a pure-copy scatter with
no reduction, which tt-metal expects), and are not the cause of the failure.

The loader already has the correct RotaryEmbedding patch (`_patch_rotary_embedding`) applied
on the current `tt_forge_models` branch (commit `bcf8f40eb3`), so the CI-era SpeculationLog-
Divergence bug is already fixed.  No loader-layer changes are needed.

## Fix
No fix applied.  Proposed next steps for a human investigator:

1. Add `LOG_F(ERROR, ...)` to every `return` site that yields `kInternal` in
   `module_builder.cc`, capture the specific pass name and any MLIR diagnostic.
2. Alternatively, run the StableHLO module for `EsmFoldStructureModuleTransition` through the
   tt-mlir pipeline in isolation (e.g. via `ttmlir-opt` or a small C++ driver) to reproduce
   the failure without the 25-minute warm-up, then bisect passes.
3. The subgraph is tiny (`[1,21,384]` inputs through 3 linear layers + ReLU + LayerNorm), so
   the failure is unlikely to be capacity-related — it is more likely a malformed IR or a
   lowering bug triggered by this specific shape.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
internal-error-unknown-mechanism

The kInternal status is returned after ~397 seconds of silent C++ compilation with no error
string captured by the PJRT log.  The exact failing pass is unknown; the fix is also unknown.
Diagnosis-first work (instrumenting module_builder.cc and running in isolation) is required
before any single-file Tier A fix can be formulated.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    1402s (wall-clock for reproduction run)
- Tier A attempts: 0

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | bcf8f40eb3f1c185963c002a931c6566054b14a8 |
