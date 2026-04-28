# Remediation Summary: led-summarization-pytorch-base-book-summary-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[led/summarization/pytorch-base-book-summary-single_device-inference]

## Result
FAIL — EmbeddingsRMProgramFactory allocates a 12.6 MB circular buffer for a 1×6297600 Longformer attention weight, exceeding the 1.5 MB L1 on Blackhole

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
embedding-rm-weight-row-exceeds-l1

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

Full kernel-level error:
```
TT_THROW @ tt_metal/impl/program/program.cpp:1136: tt::exception
info:
Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=1)] grow to 12706816 B which is beyond max L1 size of 1572864 B
...
 --- ttnn::prim::embedding(...)
 --- ttnn::embedding(...)
 --- tt::runtime::ttnn::operations::embedding::run(...)
```

## Root cause
The LED (Longformer Encoder-Decoder) model's encoder uses Longformer sparse attention.  During the forward pass one of the global-attention sub-computations produces a weight tensor of shape `1×6297600×bf16` (1 row, 6297600 columns; `6297600 = num_heads × attention_window × (2×attention_window+1) = 12 × 512 × 1025`).  This is lowered by tt-mlir to `ttnn.embedding` with index tensor `2×1×ui32` (2 global tokens, 1 index each).

At runtime, `tt::runtime::ttnn::operations::embedding::run` calls `ttnn::embedding` with `layout=TILE_LAYOUT`.  Inside `ttnn::embedding` the `fused_tilized` flag is only set to `true` when `embedding_input_tensor.padded_shape()[-1] % TILE_HEIGHT == 0`.  After reshaping the `[2,1]` index tensor to `[2,1,1,1]`, the last dimension padded by DRAM alignment (64 bytes / 4 bytes per ui32 = 16 elements) gives `padded_shape()[-1] = 16`.  Because `16 % 32 ≠ 0`, the guard fails and `fused_tilized = false`.

With `fused_tilized = false`, `EmbeddingsRMProgramFactory` is selected.  That factory allocates a circular buffer equal to `buffering_size × weight_page_size = 1 × (6297600 × 2) = 12595200 bytes`.  Adding the L1-unreserved-base offset (`111104 bytes`) gives `12706816 bytes`, which exceeds the device L1 size of `1572864 bytes`, triggering the `validate_circular_buffer_region` assertion.

`EmbeddingsFusedProgramFactory`, by contrast, has a `max_l1_budget_bytes = 1 MB` cap and uses chunked processing when the weight row is too large — but it cannot be selected here because the sentence-size tile-alignment guard fires first.

## Fix
The fix must make `EmbeddingsRMProgramFactory` safe for weight rows larger than L1, either by:

1. **Adding L1-budget-aware chunked processing to `EmbeddingsRMProgramFactory`** (parallel to the existing logic in `EmbeddingsFusedProgramFactory`).  This requires rewriting the RM dataflow kernel (`embeddings.cpp`) to read the weight row in L1-sized chunks and stream output in pieces, touching at minimum:
   - `ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp`
   - `ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp`

2. **Or relaxing the tile-alignment guard in `ttnn::embedding`** so that when `layout == TILE_LAYOUT` and the RM path would overflow L1, the input is padded to `TILE_HEIGHT` before calling `EmbeddingsFusedProgramFactory` (which already has chunked processing).  This would require:
   - `ttnn/cpp/ttnn/operations/embedding/embedding.cpp` — add input padding when `weight_page_size > L1 budget` and `fused_tilized` would otherwise be false.

## Tier B justification
**new-infrastructure** — `EmbeddingsRMProgramFactory` has no mechanism to read a weight row in chunks; adding it requires new kernel logic (dataflow kernel rewrite) alongside the factory changes, constituting new infrastructure.

## Verification
- pytest exit: FAIL
- Hardware:    blackhole-p150b
- Duration:    143.33s
- Tier A attempts: N/A

## Files changed
None (no fix attempted — Tier B)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a1548bf12317b3d943f22ff2e8be45ef43c23b01 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
