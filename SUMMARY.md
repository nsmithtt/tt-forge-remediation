# Remediation Summary: moody_wild_mix_v1_gguf-pytorch-moodyWildMix_v10Base50steps_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[moody_wild_mix_v1_gguf/pytorch-moodyWildMix_v10Base50steps_Q4_K_M-single_device-inference]

## Result
FAIL — Lumina2 complex-valued RoPE (torch.polar + view_as_complex/view_as_real) not supported by TT XLA backend; INTERNAL error code 13 at compilation

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
xla-complex-tensor-not-supported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   ValueError: Error code: 13

from user code:
   File ".../diffusers/models/transformers/transformer_lumina2.py", in _get_freqs_cis
   File ".../diffusers/models/embeddings.py", in apply_rotary_emb
     x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
     x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

## Root cause
Lumina2's RoPE implementation uses complex64 tensors throughout:
`_precompute_freqs_cis` calls `torch.polar(ones, freqs)` to produce complex64 frequency
tables. `_get_freqs_cis` moves these to the XLA device via `.to(ids.device)`. The
`apply_rotary_emb` function (called with `use_real=False`) then uses
`torch.view_as_complex` and `torch.view_as_real` on these tensors.

The TT XLA backend does not support complex dtypes (XLAComplexFloatType). When the
`_get_freqs_cis` subgraph is compiled, tt-mlir raises an INTERNAL error (code 13) because
complex tensor types have no representation in the TTIR type system.

The original reported failure (`RecursionError: maximum recursion depth exceeded`) was
one level earlier: `GGUFParameter.as_tensor()` called `_make_subclass` without
`DisableTorchFunctionSubclass`, causing infinite `__torch_function__` re-dispatch under
TorchDynamo. Fixing that (and 6 prior loader bugs on the existing remediation branch)
advances execution to the compilation stage, which then surfaces the complex-tensor bug.

## Fix
**Loader fixes (in tt_forge_models):** The existing remediation branch already contained
6 loader fixes (QKV split dims, cap_feat_dim, adaLN key mapping, LuminaRMSNormZero
conditioning dim, timestep embedding bottleneck, axes_dim_rope for head_dim=128). One
additional fix was added in this session:

- `moody_wild_mix_v1_gguf/pytorch/loader.py` (commit `fde0063f57`): patch
  `GGUFParameter.as_tensor` with `DisableTorchFunctionSubclass` to break the infinite
  `__torch_function__` recursion under TorchDynamo.

**Proposed compiler fix (not attempted — Tier B):** Implement complex tensor support in
tt-mlir: add complex dtype representation to the TTIR type system, lower
`torch.view_as_complex` / `torch.view_as_real` / complex multiplication / `torch.polar`
through StableHLO to TTIR. This is new infrastructure touching multiple files across the
compiler stack.

## Tier B justification
Indicator: **new-infrastructure**.
Complex tensor types (`XLAComplexFloatType`) have no representation in the TTIR type
system and no lowering path. Supporting them requires adding a new dtype to the type
system, adding lowering patterns for `chlo.complex`, `chlo.real`, `chlo.imag`, and
complex arithmetic ops, and plumbing them through all passes. This is a cross-cutting
change across multiple files in tt-mlir.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    161.02s (2:41) — compilation reached before INTERNAL error
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/moody_wild_mix_v1_gguf/pytorch/loader.py` — patch GGUFParameter.as_tensor with DisableTorchFunctionSubclass

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | fde0063f57d98515a3432ca68ade86a7a9679070 |
