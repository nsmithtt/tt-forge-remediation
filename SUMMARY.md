# Remediation Summary: gigachat_3_1_10b_a1_8b_gguf-causal_lm-pytorch-GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gigachat_3_1_10b_a1_8b_gguf/causal_lm/pytorch-GIGACHAT_3_1_10B_A1_8B_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Tier B: RM embedding CB overflow when MoE expert weight row exceeds L1 (3.75 MB > 1.5 MB L1)

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
embedding-rm-cb-weight-row-exceeds-l1

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=10,y=9)] grow to 7975936 B which is beyond max L1 size of 1572864 B
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause
GigaChat 3.1 10B is a DeepSeek-V2-based MoE model. The `batched_mm_experts_forward` path gathers expert weights as `gate_up_proj[expert_ids_clamped]`, which XLA lowers to `aten.embedding`. The embedding weight has shape `[64, 1966080]` — 64 experts, each with a flattened gate+up weight matrix of 1966080 elements (1280 × 1536 × 2, i.e., moe_intermediate_size × hidden_size × gate_and_up). Each "embedding row" is 1,966,080 × 2 = 3,932,160 bytes (3.75 MB).

`EmbeddingsRMProgramFactory` computes `out_cb_size = 2 × rounded_weight_page_size = 7,864,320 bytes`. `validate_circular_buffer_region` in `program.cpp:1136` then asserts that CBs fit within L1 (1,572,864 bytes), which they do not. The crash surfaces via PJRT as INTERNAL Error code: 13.

The Fused factory has a `max_l1_budget_bytes = 1 MB` guard with chunked-processing fallback, but the RM factory has no such guard. The RM factory is selected here because `a.padded_shape()[-1] = 1` (not divisible by 32), forcing the non-fused path.

## Fix
Add chunked row-processing to the RM embedding path, analogous to the existing chunked path in `EmbeddingsFusedProgramFactory`:

1. `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embeddings_rm_program_factory.cpp`:
   Add `max_l1_budget_bytes` check; compute `chunk_size` and `num_chunks` when `weight_page_size > max_l1_budget`; set `out_cb_size = chunk_size` instead of full `rounded_weight_page_size`.

2. `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp`:
   Add `chunk_size` and `num_chunks` compile-time args; wrap the single-page weight read in a chunk loop, advancing the DRAM source address by `chunk_size` each iteration.

3. `tt-metal/ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` (or a RM-specific writer):
   Accept `chunk_size` / `num_chunks` parameters; write each chunk sequentially to the correct DRAM offset within the output stick.

## Tier B justification
**new-infrastructure**: The RM embedding kernel (`embeddings.cpp`) reads a full weight stick in a single `noc_async_read<weight_stick_size>` with a compile-time template size. Chunking requires changing both the kernel and the program factory, plus the writer kernel, to support partial-stick I/O. No chunked path exists in the RM factory; adding one is new infrastructure touching 3 files.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~120s (XLA compilation ~90s, then runtime crash)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/models/gigachat_3_1_10b_a1_8b_gguf/causal_lm/pytorch/loader.py` — 8 loader fixes: deepseek2 arch registration, MLA kv-split tensor combiner, real-arithmetic RoPE, batched_mm experts, load_config context manager, pad_token guard; also `requirements.txt` with `gguf>=0.10.0`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 00612accf4eb94d1b7a77c3245fe3032e1a6ed25 |
| tt-forge-models | 8b3aba5443156df68c33b5390ac080888239bac9 |
