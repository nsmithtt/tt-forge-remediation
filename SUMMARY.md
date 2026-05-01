# Remediation Summary: granite_moe-causal_lm-pytorch-3.0_1B_A400M_Base-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_moe/causal_lm/pytorch-3.0_1B_A400M_Base-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
granitemoe-expert-dispatch-tolist-int-comparison-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

## Root cause
Three loader-layer bugs, each requiring its own fix:

**Bug 1 — minicpm `nn.Module.__getattr__` patch leaks across tests.**
Five minicpm loaders (`minicpm_o_2_6`, `minicpm_o_4_5`, `minicpm_v_2`,
`minicpm_v_2_6_int4`, `minicpmv_2_6`) patched `nn.Module.__getattr__` at
module-import time (module-level code outside any function), leaving the
replacement in place for the entire pytest session.  When pytest collected
all parametrized test IDs it imported every loader, causing the minicpm
patch to be active while tracing `granite_moe`, which diverged Dynamo's
SpeculationLog and caused the test to fail before reaching the PJRT layer.

**Bug 2 — `GraniteMoeTopKGating.forward` calls `expert_size.tolist()`.**
`expert_size` is a device tensor; `.tolist()` synchronously copies it to
host, which the PJRT TT backend rejects with `INTERNAL: Error code: 13`.

**Bug 3 — `zeros.index_add(0, batch_index, expert_outputs)` hits a scatter
rank mismatch.**  `index_add` lowers to `stablehlo.scatter`; the tt-mlir
scatter lowering creates a 2-element begins slice for the 2-D flat index
tensor, but TTNN promotes 2-D tensors to 4-D at runtime →
`TT_FATAL: Input rank 4 and begins 2 must have the same size`.

A fourth issue surfaced after bugs 2 and 3 were patched: integer equality
comparisons (`int32 == e`, `int64 == e`) on TT silicon produce wrong
values, causing PCC=0.60 regardless of whether the index tensors were kept
as int64 or cast to int32.  This was fixed by avoiding on-device integer
comparison entirely (see Fix below).

## Fix
All fixes are in `tt-xla/third_party/tt_forge_models` on branch
`remediation/granite_moe-causal_lm-pytorch-3.0_1B_A400M_Base-single_device-inference`.

**Bug 1**: Moved the `nn.Module.__getattr__` patch from module-level code
into the `load_model()` method body, scoped with `try/finally` so it is
active only during `AutoModel.from_pretrained()` and is always restored
afterwards.  Applied to all five minicpm loaders.

**Bugs 2, 3, and integer comparison**: Replaced the entire
`GraniteMoeMoE.forward` with a dense-over-experts implementation that:
1. Computes the per-expert gate matrix `[T, E]` using floating-point
   arithmetic (`abs(top_k_indices.to(dtype) - expert_id) < 0.5`) instead
   of integer equality comparison, avoiding the int-comparison bug.
2. Runs each expert for all T tokens and gates the output to 0 for
   non-assigned tokens via the float gate matrix — no sort, no dynamic
   gather, and no scatter (`index_add`) in the computation graph.

The new forward is mathematically equivalent to the original and gives
PCC=1.0 vs the unpatched model on CPU.

Files changed:
- `granite_moe/causal_lm/pytorch/loader.py`
- `minicpm_o_2_6/pytorch/loader.py`
- `minicpm_o_4_5/pytorch/loader.py`
- `minicpm_v_2/pytorch/loader.py`
- `minicpm_v_2_6_int4/pytorch/loader.py`
- `minicpmv_2_6/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    316.14s (0:05:16)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/granite_moe/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_o_2_6/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_o_4_5/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_v_2/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpm_v_2_6_int4/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/minicpmv_2_6/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 137dec415f1543e5b491c339b6aadc085b5721c5 |
