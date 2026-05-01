# Remediation Summary: longformer-masked_lm-pytorch-yorko-scibert_scivocab_uncased_long_4096-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[longformer/masked_lm/pytorch-yorko/scibert_scivocab_uncased_long_4096-single_device-inference]

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
The Python traceback points to `modeling_longformer.py:1616` (LongformerForMaskedLM calling `self.longformer()`) inside `extract_compiled_graph_helper → torch_xla.sync`.

## Root cause
The Longformer sliding window attention mechanism generates large intermediate tensors that lower to `stablehlo.gather`, which tt-mlir compiles to `ttnn.embedding`. The embedding table row width exceeds Wormhole B0 L1 size.

The `EmbeddingsRMProgramFactory` in `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp` allocates the output circular buffer as:

```cpp
uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);
out_cb_size = buffering_size * rounded_weight_page_size;
```

For the scibert_scivocab_uncased_long_4096 model (12 heads, attention_window=512, seq_len=512), the sliding window attention in `LongformerSelfAttention._sliding_chunks_matmul_qk` produces gather ops with embedding row width `1,575,936 × 2 bytes = 3,151,872 bytes`. The total CB grows to `3,263,488 B`, exceeding the Wormhole L1 limit of `1,572,864 B`.

This is identical to the failure for `longformer/feature_extraction/pytorch-Longformer_Zh-single_device-inference` (bug fingerprint `embedding-cb-l1-overflow-wide-row`). Both models share the same Longformer sliding window attention kernel structure and produce the same embedding dimensions.

## Fix
The fix would live in `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp` (and its corresponding reader/writer kernels). It requires adding horizontal sub-chunking: split the embedding row into blocks of at most `L1_size / 2` bytes each, process each block independently, and concatenate the output. This means:

1. **Host side** (`embeddings_rm_program_factory.cpp`): compute `chunk_width = floor(L1_limit / 2 / element_size)`, iterate over `ceil(row_width / chunk_width)` chunks, allocate a CB of `chunk_width × element_size` bytes.
2. **Reader kernel** (`tt-metal/ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_rm.cpp`): read only the relevant chunk of each embedding row from DRAM.
3. **Writer kernel**: write each chunk to the corresponding slice of the output tensor in DRAM.

This is new kernel infrastructure that does not currently exist in any of the three embedding program factories (`embeddings_rm`, `embeddings_fused`, `embeddings_tilized_indices`).

## Tier B justification
**Indicator:** `new-infrastructure`

The minimum unit of work for the embedding/gather kernel is one row of the embedding table. With a row width of 3.0 MB and an L1 size of 1.5 MB, no existing blocking parameter can make this work. The fix requires implementing a streaming/paged gather kernel that reads and writes in sub-row chunks — this is new device kernel infrastructure not present in tt-metal.

## Verification
- pytest exit: FAIL
- Hardware:    wormhole (n150)
- Duration:    185.38s (0:03:05)
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
