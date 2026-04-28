# Remediation Summary: granite_3_1_1b_a400m_instruct-causal_lm-pytorch-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[granite_3_1_1b_a400m_instruct/causal_lm/pytorch-3.1_1B_A400M_Instruct-single_device-inference]

## Result
FAIL — after loader fix, INTERNAL Error code 13 from aten._local_scalar_dense in MoE routing (Tier B)

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
aten-local-scalar-dense-moe-routing-graph-break-internal

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.symbolic_convert.SpeculationLogDivergence:
SpeculationLog diverged at index 677 (log had 17555 entries):
- Expected: .../minicpmv_2_6/pytorch/loader.py:46 (CALL at ip=19)
- Actual: .../minicpm_o_2_6/pytorch/loader.py:48 (CALL at ip=19)
...
Speculation entries are only added under certain conditions ...; those conditions may have changed on restart.

(After loader fix, the residual failure is:)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
at torch_xla._XLAC._xla_step_marker → torch_xla.sync → bridge.extract_compiled_graph

## Root cause
**Two bugs, one loader and one compiler-stack:**

1. **Loader bug (fixed):** Five MiniCPM loaders (`minicpm_o_2_6`, `minicpm_o_4_5`,
   `minicpm_v_2`, `minicpm_v_2_6_int4`, `minicpmv_2_6`) each patched
   `nn.Module.__getattr__` at module-import time with a different function (different
   source file and line).  When pytest collected all loaders, torch.compile's dynamo
   speculation log recorded one loader's `patched_getattr` on the first trace pass but
   found a different loader's version on restart, raising `SpeculationLogDivergence`.
   The granite model is an innocent victim: it is compiled after MiniCPM loaders are
   imported, so it sees the already-corrupted `__getattr__`.

2. **Compiler-stack bug (Tier B):** After the loader fix, GraniteMoE fails because
   `GraniteMoeTopKGating.forward()` (transformers/models/granitemoe/modeling_granitemoe.py
   line 212) calls `expert_size = expert_size.tolist()` on a device tensor.  `.tolist()`
   internally calls `.item()` per element, generating `aten._local_scalar_dense.default`
   for each of the model's experts.  The tt backend raises a `BackendCompilerException`
   for each such op (it is unsupported), triggering a graph break per call.  With many
   experts across many layers, the device enters an invalid state and the next
   `torch_xla.sync()` call fails with `INTERNAL: Error code: 13`.  The root mechanism
   of the INTERNAL error (why repeated graph breaks corrupt device state) is not known.

## Fix
**Loader fix applied (remediation branch):**
- Moved `nn.Module.__getattr__` patch inside `load_model()` in all five MiniCPM loaders,
  guarded with `_resampler_compat` sentinel so it is applied exactly once.
- Files changed in `tt_forge_models`:
  - `minicpm_o_2_6/pytorch/loader.py`
  - `minicpm_o_4_5/pytorch/loader.py`
  - `minicpm_v_2/pytorch/loader.py`
  - `minicpm_v_2_6_int4/pytorch/loader.py`
  - `minicpmv_2_6/pytorch/loader.py`
- Commit: `36dc005e4acb361a8ddbaa0e4f3c42e73897dddd` on branch
  `remediation/granite_3_1_1b_a400m_instruct-causal_lm-pytorch-single_device-inference`
  in tt_forge_models.

**Residual Tier B bug (proposed fix):**
The tt-xla backend needs to handle `aten._local_scalar_dense` gracefully when it
appears in MoE routing graphs.  Two possible approaches:
1. Support device-to-host scalar extraction in the PJRT bridge so `item()` can be
   executed during compilation without causing a graph break.
2. Detect repeated `_local_scalar_dense` graph breaks and route the entire MoE layer
   to a CPU fallback rather than breaking per-element.

## Tier B justification
`new-infrastructure` — Supporting `aten._local_scalar_dense` in the TT backend requires
implementing device-to-host transfer paths during compilation (PJRT scalar reads), which
is new infrastructure.  Additionally, the INTERNAL error mechanism (why repeated graph
breaks corrupt device state) is not known; diagnosis must precede any fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    267.38s (0:04:27) — with loader fix applied
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/minicpm_o_2_6/pytorch/loader.py`
- `tt_forge_models/minicpm_o_4_5/pytorch/loader.py`
- `tt_forge_models/minicpm_v_2/pytorch/loader.py`
- `tt_forge_models/minicpm_v_2_6_int4/pytorch/loader.py`
- `tt_forge_models/minicpmv_2_6/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c58463f60729061aeab05bf8001331c9023a1919 |
| tt-forge-models | 36dc005e4acb361a8ddbaa0e4f3c42e73897dddd |
