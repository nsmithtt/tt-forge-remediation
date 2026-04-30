# Remediation Summary: gemma3_npc_1b_float16_i1_gguf-causal_lm-pytorch-1B_FLOAT16_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_npc_1b_float16_i1_gguf/causal_lm/pytorch-1B_FLOAT16_I1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS — two fixes: (1) loader cross-contamination from broken _patched_load_gguf_checkpoint signatures; (2) aten.slice.Tensor OOB start in TorchFunctionOverride

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_29, 2, -511, 9223372036854775807), kwargs = {})
Original traceback: transformers/cache_utils.py:214:
  self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two bugs in sequence:

**Bug 1 (loader layer):** 26 co-collected GGUF loaders in tt_forge_models define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` at module level without `**kwargs`. When pytest imports all loaders during test_all_models_torch parametrization, these module-level assignments replace `gguf_utils.load_gguf_checkpoint` globally. Then transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which the broken patcher rejects with `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Bug fingerprint: `gguf-load-checkpoint-model-to-load-kwarg`.

**Bug 2 (tt-xla layer):** Gemma3NPC-1B has `sliding_window=512`. `SlidingWindowCache.update()` in transformers computes `full_value_states[:, :, -sliding_window + 1:, :]` = `full_value_states[:, :, -511:, :]`. With seq_len=23, the dim size is 23 so valid range is `[-23, 22]`. PyTorch CPU silently clamps this to 0, but `TorchFunctionOverride.__torch_function__` dispatches to `aten.slice.Tensor` which validates strictly and raises `RuntimeError: Value out of range`. Bug fingerprint: `aten-slice-tensor-out-of-bounds-start`. Tier A.

## Fix
**Fix 1 (tt_forge_models — loader):** Updated all 26 broken loaders from `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):`, and passed `**kwargs` to the inner `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)` call. Committed on remediation branch in tenstorrent/tt-forge-models.

Files changed (26 loaders):
- bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

**Fix 2 (tt-xla — Tier A):** Added slice start index clamping in `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py`. When `func is torch.ops.aten.slice.Tensor`, clamp `args[2]` (start) to `max(-size, start)` where `size = tensor.shape[dim]`. This mirrors PyTorch CPU behavior, preventing the XLA strict bounds check from firing during `partition_fx_graph_for_cpu_fallback`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    333.63s (0:05:33)
- Tier A attempts: 1

## Files changed
- python_package/tt_torch/torch_overrides.py (tt-xla remediation branch)
- 26 loader.py files in tt_forge_models (tt_forge_models remediation branch)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0f67632926101ae2690928d3bd88a6d73a0f4c8b |
| tt-forge-models | b91653d82926ad4df402714ef86fa79637e3ad13 |
