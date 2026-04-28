# Remediation Summary: longformer_zh-feature_extraction-pytorch-Longformer_Zh-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longformer/feature_extraction/pytorch-Longformer_Zh-single_device-inference]

## Result
FAIL — tt-metal embedding kernel CB allocation exceeds L1 when embedding row width > 1.5 MB (Longformer sliding window attention)

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
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=1)] grow to 3263488 B which is beyond max L1 size of 1572864 B
```
Wrapped at Python level as:
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
The Python traceback points to `modeling_longformer.py:1511` (encoder call) inside `extract_compiled_graph_helper → torch_xla.sync`.

## Root cause
The Longformer sliding window attention mechanism in `LongformerSelfAttention._sliding_chunks_matmul_qk` generates large intermediate tensors that are accessed via `stablehlo.gather`. After lowering, the largest gather ops are:

- `ttnn.embedding(tensor<2x1xui32>, tensor<1x1575936xbf16>) → tensor<2x1x1575936xbf16>`

The embedding table row width is `1,575,936 × 2 bytes = 3,151,872 bytes ≈ 3.0 MB`. The `EmbeddingsRMProgramFactory` (in `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp`) allocates the output circular buffer as:

```cpp
uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
// 1,575,936 * 2 = 3,151,872 bytes
uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);
out_cb_size = buffering_size * rounded_weight_page_size;
// Even at buffering_size=1: 3,151,872 > L1 limit of 1,572,864
```

Even with `buffering_size = 1` (the minimum), the output CB requires 3.0 MB — 2× the Wormhole L1 limit of 1,572,864 bytes. The CB grows to 3,263,488 B on the core range, triggering `validate_circular_buffer_region`.

The 1,575,936 comes from the Longformer attention sliding window: `12 heads × 256 seq_chunks × 513 window_span = 1,575,936`. A second set of large embeddings appears as `tensor<256x1xui32>` into `tensor<255x6156xbf16>` → `tensor<256x1x6156xbf16>` (256 × 6156 × 2 = 3.15 MB).

## Fix
The fix would live in `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp` (and its corresponding reader/writer kernels). It requires adding horizontal sub-chunking: split the embedding row into blocks of at most `L1_size / 2` bytes each, process each block independently, and concatenate the output. This means:

1. **Host side** (`embeddings_rm_program_factory.cpp`): compute `chunk_width = floor(L1_limit / 2 / element_size)`, iterate over `ceil(row_width / chunk_width)` chunks, allocate a CB of `chunk_width × element_size` bytes.
2. **Reader kernel** (`tt-metal/ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_rm.cpp`): read only the relevant chunk of each embedding row from DRAM.
3. **Writer kernel**: write each chunk to the corresponding slice of the output tensor in DRAM.

This is new kernel infrastructure that doesn't currently exist in any of the three embedding program factories (`embeddings_rm`, `embeddings_fused`, `embeddings_tilized_indices`).

## Tier B justification
**Indicator:** `new-infrastructure`

The minimum unit of work for the embedding/gather kernel is one row of the embedding table. With a row width of 3.0 MB and an L1 size of 1.5 MB, no existing blocking parameter can make this work. The fix requires implementing a streaming/paged gather kernel that reads and writes in sub-row chunks — this is new device kernel infrastructure not present in tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole (n150)
- Duration:    201.09s (0:03:21)
- Tier A attempts: N/A

## Files changed
None (Tier B FAIL, no code changes made)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
