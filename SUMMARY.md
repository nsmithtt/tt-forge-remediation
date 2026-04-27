# Remediation Summary: axionml_qwen3_5_2b_nvfp4/image_to_text/pytorch-2B_NVFP4-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[axionml_qwen3_5_2b_nvfp4/image_to_text/pytorch-2B_NVFP4-single_device-inference]

## Result
FAIL — tt-metal static L1 buffer allocation overflows for vision encoder (2247168 B > 1572864 B max)

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to
2247168 B which is beyond max L1 size of 1572864 B (assert.hpp:104)
```

## Root cause
**Layer: tt-mlir / tt-metal (compiler core / backend runtime)**

The model is `AxionML/Qwen3.5-2B-NVFP4` — a Qwen3.5 vision-language model with a 24-layer ViT
vision encoder (hidden_size=1024, intermediate_size=4096, num_heads=16) and a 24-layer Mamba+MHA
hybrid language model.

There were two loader-layer bugs blocking compilation, both fixed in the remediation branch
(see Fix section below). After those fixes the model loads and begins compiling, but the
tt-metal backend throws a fatal error during XLA compilation of the vision encoder's
computation graph: the statically allocated circular buffers for the core range
`(x=0,y=0)-(x=10,y=9)` grow to **2247168 B** which exceeds the Blackhole p150b L1 limit of
**1572864 B**. The overflow likely originates in the ViT self-attention SDPA or the MLP's large
matmuls (e.g. QKV projection weight [3072, 1024] or MLP intermediate [4096, 1024]).

This is a compiler-stack bug: tt-mlir is scheduling operations or allocating intermediate
buffers in a way that exhausts L1 for the vision encoder. No model-side workaround is possible
without trimming or offloading, both of which are forbidden.

## Fix
**Loader-layer fixes applied (tt-forge-models `remediation/axionml-qwen3-5-2b-nvfp4`):**

1. **`ignore_mismatched_sizes=True`** — `AxionML/Qwen3.5-2B-NVFP4` stores weights in NVIDIA's
   NVFP4 format (4-bit float, packed as `uint8`, half the size of the bf16 model parameters).
   Without `nvidia-modelopt` installed, transformers creates unquantized bf16 layers with 2×
   the checkpoint weight shape; the new `log_state_dict_report` in transformers 5.2 raises a
   hard error on this mismatch. Setting `ignore_mismatched_sizes=True` allows the model to load
   (quantized-layer weights are randomly initialised, which is acceptable for a compiler test
   that compares TT vs CPU outputs with the same weights).

2. **`_patch_qwen35_for_tt_device()`** — `Qwen3_5VisionModel.fast_pos_embed_interpolate`,
   `rot_pos_emb`, `Qwen3_5Model.get_image_features`, and `get_rope_index` call `.tolist()` on
   tensors that are moved to TT device with the model inputs. TT device does not support Python-
   side data access on device tensors. The patch moves grid-dimension metadata tensors (`grid_thw`,
   `image_grid_thw`, `input_ids`, `attention_mask`) to CPU for the control-flow operations that
   need them as Python lists, then moves the resulting position_ids/rope_deltas back to TT.
   The vision encoder and language model computations themselves remain on TT.

**Proposed fix for the L1 overflow (tt-mlir / tt-metal):**

Improve the circular-buffer allocation / tiling strategy in tt-mlir's compilation pipeline for
large matmuls so that the total statically allocated L1 across a compiled subgraph does not
exceed the hardware limit. Possible approaches:
- Reduce the tile shape used for large (≥1024×1024) matmuls in the vision encoder.
- Switch from fully-static allocation to a streaming allocation scheme where buffers are reused
  across sequential operations.
- Split the vision encoder into multiple separately-compiled subgraphs so that per-subgraph
  peak L1 stays within budget.

## Verification
FAIL — compilation aborts with L1 overflow before any output is produced.
Hardware: Blackhole p150b (single chip).

## Files changed
- `axionml_qwen3_5_2b_nvfp4/image_to_text/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | e2f0fc7a03a838b79009089d3cfe44851777df94 |
| tt-forge-models | 4a25e73274989c44e2ea7a587c9a97f0f8f46a30 |
