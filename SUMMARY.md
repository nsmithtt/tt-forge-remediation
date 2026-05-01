# Remediation Summary: isaac_0_1-pytorch-Isaac_0_1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[isaac_0_1/pytorch-Isaac_0_1-single_device-inference]

## Result
FAIL — Residual PCC=-0.166 is the Tier B ttnn-sdpa-nonaligned-kv-pcc-wrong bug (seq_len=1551, 1551%32=15)

## Stack layer
tt-mlir

## Tier
B

## Bug fingerprint
ttnn-sdpa-nonaligned-kv-pcc-wrong

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original error (before loader fixes):
```
RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations:
prims::view_of(Tensor(a) a) -> Tensor(a). We only support functionalizing operators
whose outputs do not have alias annotations...
```

Residual failure after fixes:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed.
Calculated: pcc=-0.16578252459019363. Required: pcc=0.99.
```

## Root cause
Four bugs were encountered in sequence:

1. **prims::view_of (tt-xla, Tier A, fixed)**: `run_decompositions` in `torch_pass_pipeline`
   introduces `prims.view_of` nodes for trivial identity sub-graphs produced by the many
   torch.compile graph breaks in perceptron's TensorStream code. XLA's
   `partition_fx_graph_for_cpu_fallback` cannot functionalize the non-ATen op with alias
   annotations. Fixed by adding a `bypass_prims_view_of` pass in `passes.py` before the XLA
   bridge.

2. **MiniCPM Resampler Dynamo guard (loader, fixed)**: All five MiniCPM loaders
   (`minicpm_o_2_6`, `minicpm_o_4_5`, `minicpm_v_2`, `minicpm_v_2_6_int4`, `minicpmv_2_6`)
   define `_original_getattr` inside the `if not ... _tt_resampler_getattr_patched` guard.
   When another MiniCPM loader ran first, the guard prevented assignment, and Dynamo's
   CLOSURE_MATCH guard later raised `AttributeError: module has no attribute '_original_getattr'`
   during Isaac compilation. Fixed by hoisting the assignment before the guard block.

3. **F.interpolate bfloat16 antialias (loader, fixed)**: `IsaacVisionEmbeddings.
   resize_positional_embeddings` only casts positional embeddings to float32 when
   `device.type == "cpu"`, but XLA devices also require float32 for `antialias=True`.
   Fixed by patching the method in `load_model()` to cast unconditionally when dtype is
   bfloat16 or float16.

4. **SDPA non-tile-aligned K/V (tt-mlir, Tier B, unfixed)**: Isaac produces a sequence of
   1551 tokens (1551 % 32 = 15, non-tile-aligned). The TTNN SDPA kernel gives wrong results
   for non-tile-aligned K/V sequences, producing PCC=-0.166. This is the known
   `ttnn-sdpa-nonaligned-kv-pcc-wrong` Tier B bug documented in prior reports (e.g.,
   Chronos2, DNABERT-S). The Qwen3 backbone's 28-layer attention uses SDPA for all
   1551 × 1551 position pairs; with 15 non-aligned tail positions, the attention scores
   are corrupted across all layers.

## Fix
Loader fixes (tt_forge_models `remediation/isaac_0_1-pytorch-Isaac_0_1-single_device-inference`):
- `isaac_0_1/pytorch/loader.py`: Added `_patch_isaac_vision_embeddings()` to fix the
  bfloat16 antialias cast and called it in `load_model()`. (commit 31f03daea9)
- All five MiniCPM loaders: hoisted `_original_getattr` assignment before the
  `_tt_resampler_getattr_patched` guard block. (commit e701a0cc8e)

Compiler fix (tt-xla `remediation/isaac_0_1-pytorch-Isaac_0_1-single_device-inference`):
- `python_package/tt_torch/backend/passes.py`: Added `bypass_prims_view_of()` pass that
  replaces `prims.view_of(x)` nodes with their input `x` before the XLA bridge. (commit d8e578b62)
- `python_package/tt_torch/backend/backend.py`: Call `bypass_prims_view_of` after
  `insert_argument_type_markers` in `torch_pass_pipeline`. (commit d8e578b62)

The residual Tier B SDPA bug is unfixed. The proposed fix (padding K/V to the next tile
boundary with masking in the TTNN SDPA kernel) would require coordinated changes across
`tt-metal`'s SDPA kernel, TTIR lowering in `tt-mlir`, and StableHLO SDPA pattern matching
in `tt-xla`.

## Tier B justification
`cross-cutting`: Fixing SDPA for non-tile-aligned sequences requires changes in the TTNN
kernel (padding K/V to tile boundary and masking), the TTIR→TTNN lowering pass, and
potentially the StableHLO SDPA conversion pattern. It touches at least 3 files across
2 repos and affects every model with non-tile-aligned sequence lengths.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 302.95s (0:05:02)
- Tier A attempts: 1 (prims::view_of bypass — PASS for that specific bug)

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py` (bypass_prims_view_of pass)
- `tt-xla/python_package/tt_torch/backend/backend.py` (call bypass_prims_view_of)
- `tt-xla/third_party/tt_forge_models/isaac_0_1/pytorch/loader.py` (antialias bfloat16 patch)
- `tt-xla/third_party/tt_forge_models/minicpm_o_2_6/pytorch/loader.py` (_original_getattr hoist)
- `tt-xla/third_party/tt_forge_models/minicpm_o_4_5/pytorch/loader.py` (_original_getattr hoist)
- `tt-xla/third_party/tt_forge_models/minicpm_v_2/pytorch/loader.py` (_original_getattr hoist)
- `tt-xla/third_party/tt_forge_models/minicpm_v_2_6_int4/pytorch/loader.py` (_original_getattr hoist)
- `tt-xla/third_party/tt_forge_models/minicpmv_2_6/pytorch/loader.py` (_original_getattr hoist)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8e0463a5edb3874bcef2bda0ba3e4788bf4e04d2 |
| tt-forge-models | 31f03daea9cb41de8cf69fe4de175a4fc2d418af |
