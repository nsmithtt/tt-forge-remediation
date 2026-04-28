# Remediation Summary: gemma_3_12b_it_fp8_dynamic-multimodal-pytorch-RedHatAI-gemma-3-12b-it-FP8-dynamic-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_it_fp8_dynamic/multimodal/pytorch-RedHatAI/gemma-3-12b-it-FP8-dynamic-single_device-inference]

## Result
FAIL — compressed_tensors FP8 quantized ops trigger XLA dynamo bridge CPU-fallback partitioning that fails with AttributeError on xla_args

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
compressed-tensors-fp8-ops-xla-partition-failure

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The reported CI failure message is a warning elevated to an exception:

```
The image processor of type `Gemma3ImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

After fixing the processor warning and the missing `compressed-tensors` dependency, the real
compiler-stack failure is:

```
AttributeError: 'fused_0' object has no attribute 'xla_args'
```

Full traceback root:
```
transformers/models/siglip/modeling_siglip.py: in forward
    queries = self.q_proj(hidden_states)
compressed_tensors/quantization/lifecycle/forward.py: in quantized_forward
    ???
python_package/tt_torch/torch_overrides.py: in __torch_function__
tt_torch/backend/backend.py: in _call_experimental_compile
    self.compiled_graph = bridge.extract_compiled_graph(...)
torch_xla/_dynamo/dynamo_bridge.py: in extract_compiled_graph_helper
    return partition_fx_graph_for_cpu_fallback(xla_model, xla_args)
torch_xla/_dynamo/dynamo_bridge.py: in partition_fx_graph_for_cpu_fallback
    extract_internal(fused_module), node.args, None)
torch_xla/_dynamo/dynamo_bridge.py: in extract_graph_helper
    xla_args = xla_model.xla_args
AttributeError: 'fused_0' object has no attribute 'xla_args'
```

## Root cause
Three bugs were found:

1. **Loader bug (fixed):** `AutoProcessor.from_pretrained()` without `use_fast=False` emits a
   transformers 5.x warning about `Gemma3ImageProcessor` being loaded as a fast processor. In the
   CI environment, Python warnings are elevated to errors, so this is the first failure.

2. **Loader bug (fixed):** `compressed-tensors` was not listed in the model's dependencies.
   `RedHatAI/gemma-3-12b-it-FP8-dynamic` uses the `compressed-tensors` quantization format, which
   requires the `compressed_tensors` Python package to load via `Gemma3ForConditionalGeneration.from_pretrained`.

3. **Compiler-stack bug (unfixed):** After loading, the `compressed_tensors`-quantized linear layers
   use a custom `quantized_forward` function that dispatches through PyTorch's `__torch_function__`
   mechanism. When `tt_torch/backend/backend.py` calls `bridge.extract_compiled_graph` on the
   exported program, the XLA dynamo bridge determines that these custom quantized ops cannot run on
   XLA and invokes `partition_fx_graph_for_cpu_fallback`. That function creates a `fused_0`
   `nn.Module` for the XLA-compilable subgraph and calls `extract_internal(fused_0)`. Inside
   `extract_internal` → `extract_graph_helper`, the code accesses `xla_model.xla_args`, but
   `fused_0` is a plain `nn.Module` without that attribute, raising `AttributeError`. The root cause
   is that the `compressed_tensors` FP8 custom ops are not lowerable by the TT compiler, causing the
   dynamo bridge to attempt CPU partitioning which itself is broken for this model shape.

This is the same failure as `gemma_3_4b_it_fp8_dynamic` (same `compressed-tensors-fp8-ops-xla-partition-failure` bug fingerprint).

## Fix
Two loader fixes applied:

- **`tt_forge_models/gemma_3_12b_it_fp8_dynamic/multimodal/pytorch/loader.py`**: Added
  `use_fast=False` to `AutoProcessor.from_pretrained()` to suppress the transformers 5.x
  `Gemma3ImageProcessor` fast-processor warning.
- **`tt_forge_models/gemma_3_12b_it_fp8_dynamic/multimodal/pytorch/requirements.txt`** (created):
  Added `compressed-tensors` so the model can be loaded without `ImportError`.

Proposed fix for the compiler bug: Implement lowerings for `compressed_tensors` FP8 quantized
matrix-multiply ops in the TT compiler pipeline, so that the quantized SiGLIP attention projections
in `Gemma3ForConditionalGeneration` trace through StableHLO without triggering CPU fallback
partitioning. Alternatively, fix `partition_fx_graph_for_cpu_fallback` in
`torch_xla/_dynamo/dynamo_bridge.py` to properly initialize `xla_args` on `fused_X` subgraph
modules before calling `extract_internal`.

## Tier B justification
new-infrastructure — The `compressed_tensors` FP8 quantized ops (custom `quantized_forward` on
SiGLIP linear layers) are not supported by the current TT compiler pipeline. Supporting them requires
new lowering patterns for the quantized matmul ops, which constitutes new infrastructure. The CPU-
fallback path in `torch_xla/_dynamo/dynamo_bridge.py:partition_fx_graph_for_cpu_fallback` is also
broken for this case (`fused_0` missing `xla_args`), suggesting additional fixes are needed in the
XLA bridge layer.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: 234.20s (0:03:54)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gemma_3_12b_it_fp8_dynamic/multimodal/pytorch/loader.py` (modified: added `use_fast=False`)
- `tt_forge_models/gemma_3_12b_it_fp8_dynamic/multimodal/pytorch/requirements.txt` (created)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 27bfc17d53476c7f1dc3f7165cbd90f03e73747b |
| tt-forge-models | 69d5c2762170c8c7ea48c99e0403835870297809 |
