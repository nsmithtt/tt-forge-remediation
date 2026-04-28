# Remediation Summary: glm-glm_5_awq-pytorch-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glm/glm_5_awq/pytorch-single_device-inference]

## Result
FAIL — tt-metal CB overflow in EmbeddingsDeviceOperation; statically allocated CBs grow to 4305920 B on cores (x=0,y=0)-(x=0,y=9), exceeding max L1 size of 1572864 B

## Stack layer
tt-metal

## Tier
B

## Bug fingerprint
ttnn-embedding-cb-overflow-large-vocab

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

Underlying tt-metal error:
```
TT_THROW: Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=9)] grow to 4305920 B which is beyond max L1 size of 1572864 B
```

Backtrace (abbreviated):
```
tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region
tt::tt_metal::distributed::MeshWorkloadImpl::compile
EmbeddingsDeviceOperation
ttnn::prim::embedding
ttnn::embedding
tt::runtime::ttnn::operations::embedding::run
```

## Root cause

**First bug (loader, fixed):** The GLM-5 AWQ model uses a MoE architecture that defaults to `_experts_implementation = "grouped_mm"`. The `grouped_mm_experts_forward` function in `transformers/integrations/moe.py` (line 270) selects `expert_ids_g.int()` for non-CPU devices (`device.type != "cpu"`). When running under XLA, `torch.histc` is subsequently called on this integer tensor, which fails because `torch.histc` does not support integer input (`"histogram_cpu" not implemented for 'Int'`). **Fixed** by setting `config._experts_implementation = "batched_mm"` in the loader before `AutoModelForCausalLM.from_config`, which routes expert dispatch to `batched_mm_experts_forward` — an XLA-compatible path.

**Second bug (tt-metal, unfixed):** After the loader fix, execution reaches the TT silicon and fails in `EmbeddingsDeviceOperation` during program compilation. The `validate_circular_buffer_region` check in `tt_metal/impl/program/program.cpp:1136` reports that statically allocated CBs on core range `(x=0,y=0)-(x=0,y=9)` total 4305920 B, exceeding the max L1 size of 1572864 B (2.74× over budget).

The model uses `vocab_size=154880` (unchanged from the HuggingFace config) and `hidden_size=1024` (overridden by loader). The three embedding program factories (`EmbeddingsFusedProgramFactory`, `EmbeddingsRMProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory`) each have CB-size guards (`max_l1_budget_bytes = 1 MB`, double-buffered `weight_page_size = 2048 B`) that should produce ≤ 262 KB. The 4305920 B size does not match any formula from static analysis. Identifying the exact code path producing this CB size requires runtime instrumentation (printing `out_cb_size`, `output_sharded`, `output.buffer()->aligned_size_per_bank()` in the factory).

## Fix

**Loader fix (committed):** In `glm/glm_5_awq/pytorch/loader.py`, added `config._experts_implementation = "batched_mm"` before `AutoModelForCausalLM.from_config(config, ...)`. This avoids `grouped_mm_experts_forward` (which calls `torch.histc` with `.int()` on XLA) and uses `batched_mm_experts_forward` which works on XLA.

**Proposed fix for the CB overflow (not implemented):** Instrument one or more of the embedding program factories (`embeddings_rm_program_factory.cpp`, `embeddings_fused_program_factory.cpp`, `embeddings_tilized_indices_program_factory.cpp`) to log `out_cb_size`, `output_sharded`, `aligned_size_per_bank`, `weight_page_size`, and `num_blocks_per_core_group_1` at the point of program creation. Identify the formula path producing 4305920 B and either:
- Add a guard/clamp similar to `max_l1_budget_bytes = 1 MB` that already exists in `EmbeddingsFusedProgramFactory`, or
- Fix the incorrect allocation formula.

File: `tt-metal/ttnn/cpp/ttnn/operations/embedding/device/` (one or more factory files).

## Tier B justification

**internal-error-unknown-mechanism**: While `Bad StatusOr access: INTERNAL: Error code: 13` is known to originate from the CB overflow in `program.cpp:1136`, the specific formula path in the embedding program factory that produces 4305920 B for `vocab_size=154880, hidden_size=1024, seq_len=5, cores=(x=0,y=0)-(x=0,y=9)` cannot be identified from static source inspection alone. The `EmbeddingsFusedProgramFactory` has a `max_l1_budget_bytes = 1 MB` guard, but the 4.1 MB CB size bypasses this guard — indicating either a different factory or an unguarded code path is selected. Identifying the code path requires runtime instrumentation (a rebuild with logging added).

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: ~191s (3:11)
- Tier A attempts: N/A

## Files changed
- `glm/glm_5_awq/pytorch/loader.py` (tt-forge-models, remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b  |
| tt-xla          | 8f9361157  |
| tt-forge-models | a06023d055 |
