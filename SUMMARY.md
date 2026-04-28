# Remediation Summary: granite_guardian-causal_lm-pytorch-3_2_3B_A800M-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[granite_guardian/causal_lm/pytorch-3.2_3B_A800M-single_device-inference]

## Result
SILICON_PASS

## Stack layer
tt-mlir

## Tier
A

## Bug fingerprint
stablehlo-embedding-backward-pattern-1d-index-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
torch._dynamo.symbolic_convert.SpeculationLogDivergence (original):
  SpeculationLog diverged at index 314: Expected minicpmv_2_6/pytorch/loader.py:46,
  Actual minicpm_o_2_6/pytorch/loader.py:48 — "have changed on restart."

After loader fix, second failure:
  RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
  TT_FATAL @ embedding_backward_device_operation.cpp:67:
    grad_tensor_shape[2] == index_tensor_shape[0] * index_tensor_shape[-1]
    "Number of rows in gradient tensor must be equal to number of indices in index tensor"

## Root cause

**Fix 1 — loader:** Five MiniCPM model loaders (`minicpm_o_2_6`, `minicpmv_2_6`,
`minicpm_o_4_5`, `minicpm_v_2`, `minicpm_v_2_6_int4`) each apply an
`nn.Module.__getattr__` monkey-patch at module-import time to work around a
missing `Resampler._initialize_weights` method in torch 2.7.0+. Because
`setup_test_discovery` imports every loader during pytest collection, both
patches chain on `nn.Module.__getattr__`. Between Dynamo's first and second
analysis pass of the Granite Guardian model, the chain order changed (one loader
was re-executed via a different `sys.modules` key), causing the
SpeculationLogDivergence "have changed on restart."

**Fix 2 — tt-mlir:** GraniteMoE is a mixture-of-experts model whose forward
pass calls `index_add_` (scatter-add) in a per-expert loop. XLA lowers this to
`stablehlo.scatter` with 1D scatter indices `[K]` (or 2D `[K, 1]`),
sum-reduction, `scatterDimsToOperandDims=[0]`, and `insertedWindowDims=[0]`.
`StableHLOToTTIREmbeddingBackwardOpConversionPattern` was designed for 2D `[N,1]`
indices with `indexVectorDim=1` (the XLA embedding_backward shape), but its
guards only checked operand rank, scatter dimension, and embedding-dim equality —
all of which the MoE scatter also satisfies. It then emitted `ttir::EmbeddingBackwardOp`.
The downstream TTIRToTTNN lowering sets `R = product(grad_dims[:-1])` and expects
`grad_shape[2] == index_shape[0] * index_shape[-1]`. For 1D indices `[K]`, this
becomes `K == K*K` (both dims are the same axis), which fails for K > 1.

## Fix

**tt_forge_models (`remediation/granite_guardian-causal_lm-pytorch-3_2_3B_A800M-single_device-inference`):**
Moved the `nn.Module.__getattr__` Resampler patch from module-level (applied at
import time) to inside `load_model()` as a scoped try/finally block in all five
MiniCPM loaders:
- `minicpm_o_2_6/pytorch/loader.py`
- `minicpmv_2_6/pytorch/loader.py`
- `minicpm_o_4_5/pytorch/loader.py`
- `minicpm_v_2/pytorch/loader.py`
- `minicpm_v_2_6_int4/pytorch/loader.py`

**tt-mlir (`remediation/granite_guardian-causal_lm-pytorch-3_2_3B_A800M-single_device-inference`):**
Added an early guard in `StableHLOToTTIREmbeddingBackwardOpConversionPattern`
(`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`, ~line 5630):
```cpp
if (scatterIndicesType.getRank() != 2 || indexVectorDim != 1) {
  return rewriter.notifyMatchFailure(srcOp,
      "EmbeddingBackward pattern requires 2D scatter indices with "
      "indexVectorDim=1; other shapes handled by generic scatter");
}
```
Non-matching scatters fall through to `StableHLOToTTIRScatterOpConversionPattern`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    ~16 minutes (11:45–12:01)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/minicpm_o_2_6/pytorch/loader.py`
- `tt_forge_models/minicpmv_2_6/pytorch/loader.py`
- `tt_forge_models/minicpm_o_4_5/pytorch/loader.py`
- `tt_forge_models/minicpm_v_2/pytorch/loader.py`
- `tt_forge_models/minicpm_v_2_6_int4/pytorch/loader.py`
- `tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 65cf619a449be105b19a61c354e8f85f9ff8e01e |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 4b399c3e89e3d5583de543dd05703f5c099c7013 |
