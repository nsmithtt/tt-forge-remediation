# Remediation Summary: bartowski_ai21labs_ai21_jamba2_mini_gguf-causal_lm-pytorch-AI21_Jamba2_Mini_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_ai21labs_ai21_jamba2_mini_gguf/causal_lm/pytorch-AI21_Jamba2_Mini_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — Jamba MoE dynamic-shape for-loop segfaults during tt-xla CPU fallback partitioning

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
jamba-moe-dynamic-shape-for-loop-segfault

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Fatal Python error: Segmentation fault in torch/_ops.py:__call__ via
tt_torch/torch_overrides.py:34 __torch_function__ during
partition_fx_graph_for_cpu_fallback while executing JambaModel.forward at
modeling_jamba.py:847.

## Root cause
The loader bugs (jamba GGUF architecture unregistered, old-style
_patched_load_gguf_checkpoint kwargs) were fixed and the model loads
successfully. The segfault occurs in the tt-xla compiler frontend during
`partition_fx_graph_for_cpu_fallback`, which executes ops on CPU during
compilation to identify the TT-vs-CPU boundary.

`JambaExperts.forward` (modeling_jamba.py:614) uses:
1. `expert_hit = nonzero(...)` — produces a dynamically-shaped tensor whose
   size depends on input values
2. `for expert_idx in expert_hit:` — Python control flow iterating over
   a device tensor, requiring host-device transfer
3. `token_idx = torch.where(expert_mask[expert_idx])` — another dynamic-shape op
4. `final_hidden_states.index_add_(0, token_idx, ...)` — in-place scatter

When the dynamo bridge tries to execute these ops on CPU during graph
partitioning, the dynamic shapes and scatter/gather pattern crash in native
PyTorch code. This is the same class of bug as the Qwen3MoE segfault
(qwen3moe_batched_mm), but Jamba has no `_experts_implementation` config
switch to select a non-looping implementation.

## Fix
Two loader fixes were applied (committed to tt_forge_models
remediation/bartowski_ai21labs_ai21_jamba2_mini_gguf... branch):

1. **Register jamba GGUF architecture** (commit fcc2313562):
   `bartowski_ai21labs_ai21_jamba2_mini_gguf/causal_lm/pytorch/loader.py`
   Added `_patch_jamba_support()` that registers jamba in
   `GGUF_TO_TRANSFORMERS_MAPPING` (field mapping: block_count →
   num_hidden_layers, embedding_length → hidden_size, ssm.* → mamba_d_*,
   expert_count/expert_used_count → MoE fields), `TENSOR_PROCESSORS`
   (MambaTensorProcessor for SSM A_log and conv1d shape handling), and
   `GGUF_TO_FAST_CONVERTERS` (GGUFLlamaConverter — jamba uses llama-style
   tokenizer per GGUF metadata tokenizer.ggml.model='llama').

2. **Fix narrow _patched_load_gguf_checkpoint signatures** (commit 746ec0ff53):
   26 GGUF loaders had `_patched_load_gguf_checkpoint(gguf_path,
   return_tensors=False)` which rejected the `model_to_load` kwarg added in
   transformers 5.2.0. Because the test suite imports all loaders during
   discovery, any of these loaders' patches poisoned the global
   `load_gguf_checkpoint`. Fixed by adding `**kwargs` and passing through.

The remaining compiler-stack bug (segfault in MoE dispatch) requires
replacing `JambaExperts.forward` with a batched-matmul implementation that
avoids dynamic nonzero/for-loop/index_add patterns. This would need to be
added to the loader as a monkey-patch of `JambaExperts.forward`, or fixed
upstream in transformers.

## Tier B justification
`internal-error-unknown-mechanism`: The segfault occurs deep in native
PyTorch C++ code during CPU fallback partitioning. The exact pointer or
memory access causing the crash is not visible from the Python traceback. A
fix requires either: (a) a loader-level monkey-patch implementing a batched
matmul MoE dispatch for JambaExperts (cross-cutting change involving
dynamic-shape semantics) or (b) a fix in the tt-xla dynamo bridge to handle
dynamic-shape ops without crashing. Both require diagnosis beyond this report.

## Verification
- pytest exit: FAIL (Segmentation fault — process killed by SIGSEGV)
- Hardware:    n150
- Duration:    ~211s to first loader error; segfault at ~75s into model run
- Tier A attempts: N/A

## Files changed
- `bartowski_ai21labs_ai21_jamba2_mini_gguf/causal_lm/pytorch/loader.py` (jamba arch registration)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 746ec0ff53f1f4340aa2336ef1820fb01dedbba2 |
