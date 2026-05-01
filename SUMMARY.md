# Remediation Summary: longformer/sequence_classification/pytorch-longformer-harmful-ro-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longformer/sequence_classification/pytorch-longformer-harmful-ro-single_device-inference]

## Result
FAIL — Tier B: embedding CB L1 overflow in tt-metal EmbeddingsRMProgramFactory; streaming/paged gather kernel required

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
embedding-cb-l1-overflow-wide-row

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: Error code: 13

Traceback (condensed):
  python_package/tt_torch/backend/backend.py:215: in _call_experimental_compile
    self.compiled_graph = bridge.extract_compiled_graph(...)
  torch_xla/_dynamo/dynamo_bridge.py:804: in partition_fx_graph_for_cpu_fallback
    extract_internal(fused_module), node.args, None)
  torch_xla/_dynamo/dynamo_bridge.py:483: in extract_graph_helper
    torch_xla._XLAC._xla_warm_up_cache(args_and_out_tensor_only, [])
  ValueError: Error code: 13

The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` in the failure message is a harmless Python exit-time warning from SWIG-generated code; the real failure is the Error code: 13 from `_xla_warm_up_cache`.

## Root cause
LongformerSelfAttention uses sliding-window attention implemented via `_sliding_chunks_matmul_qk`, which lowers to `stablehlo.gather` → `ttnn.embedding`. With `max_length=512` and 12 attention heads, the embedding row width is approximately 12 heads × 256 seq_chunks × 513 window_span = 1,575,936 × 2 B ≈ 3.15 MB per embedding row. This exceeds the Wormhole L1 capacity of 1,572,864 B (1.5 MB).

In `tt-metal`'s `EmbeddingsRMProgramFactory`, the circular buffer size is computed as `buffering_size * weight_page_size`. Even at `buffering_size=1`, a 3 MB row width exceeds L1. There is no existing blocking parameter that can work around this; sub-row chunking (a streaming gather kernel) is required.

The error surfaces as `INTERNAL: Error code: 13` (wrapped to `ValueError: Error code: 13` in Python) during `_xla_warm_up_cache` — the hardware compilation phase.

This is a consistent failure across all Longformer model variants (feature_extraction, masked_lm, sequence_classification, token_classification), regardless of sequence length within the typical 512–4096 range.

## Fix
No fix was attempted. The proposed fix would live in `tt-metal`, specifically in the `EmbeddingsRMProgramFactory` (likely `tt_metal/impl/kernels/dataflow/embeddings_rm_program_factory.cpp` or equivalent). It would require implementing a streaming/paged gather kernel that processes the wide embedding row in sub-row chunks that fit within L1, rather than allocating the full row as a single CB.

## Tier B justification
**new-infrastructure**: The fix requires implementing a sub-row chunking (streaming/paged) gather kernel in `EmbeddingsRMProgramFactory`. No existing blocking parameter can accommodate a 3 MB row in 1.5 MB L1. This is a new execution primitive, not a scoped one-file fix.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole
- Duration:    49.35s
- Tier A attempts: N/A

## Files changed
None (Tier B — no fix attempted)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
