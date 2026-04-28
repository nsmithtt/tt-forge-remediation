# Remediation Summary: gemma3_qat_gguf-causal_lm-pytorch-4B_IT_QAT_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_qat_gguf/causal_lm/pytorch-4B_IT_QAT_GGUF-single_device-inference]

## Result
SILICON_PASS — two bugs fixed: (1) other loaders' broken GGUF patch signatures; (2) out-of-range slice start from SlidingWindowCache with short seq_len

## Stack layer
loader | tt-xla

Two fixes were required:
1. **Loader layer** — 26 co-collected loaders had `_patched_load_gguf_checkpoint` with a fixed signature missing `**kwargs`, blocking the `model_to_load` arg added in transformers 5.2.0.
2. **tt-xla layer** — `aten.slice.Tensor` start=-1023 out of range [-23, 22] when SlidingWindowCache (sliding_window=1024) slices key/value states shorter than the window.

## Tier
N/A (loader fix) | A (tt-xla FX pass, one file)

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg | aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
First failure (collection cross-contamination):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Second failure (after loader fix):
```
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807), kwargs = {})
Original traceback:
  File "transformers/models/gemma3/modeling_gemma3.py", line 371, in forward
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
  File "transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

(The originally reported `TT_THROW @ silicon_sysmem_manager.cpp:326` was a different manifestation of the same path with a different test collection order on the CI machine; the loader GGUF patch error prevented reaching that point locally.)

## Root cause

**Bug 1 (loader):** transformers 5.2.0 added `model_to_load=None` to `load_gguf_checkpoint` and calls it at `modeling_utils.py:4016` with `model_to_load=dummy_model`. Twenty-six model loaders (gpt_oss_swallow, qwen3_5 variants, etc.) monkey-patch `gguf_utils.load_gguf_checkpoint` at module import time with a fixed signature `(gguf_path, return_tensors=False)` that drops the new kwarg. When pytest collects all tests, these loaders are imported before `gemma3_qat_gguf`, leaving the broken patch in effect. The gemma3_qat_gguf loader itself does not patch anything, so it is a pure victim.

**Bug 2 (tt-xla):** Gemma 3 uses `SlidingWindowCache` with `sliding_window=1024`. With seq_len=23 tokens, `cache_utils.py:214` slices `full_value_states[:, :, -1023:, :]`. PyTorch CPU semantics clamp -1023 to 0 (valid), but the XLA/TT FX tracing layer rejects `start=-1023` as out of range for a dimension of size 23, raising RuntimeError before any kernel is dispatched.

## Fix

**Fix 1 — tt_forge_models (26 loaders):** Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):` and added `**kwargs` to the `_orig_load_gguf_checkpoint(...)` call in each of the 26 broken loaders. Branch: `remediation/gemma3_qat_gguf-causal_lm-pytorch-4B_IT_QAT_GGUF-single_device-inference` in tt-forge-models.

**Fix 2 — tt-xla:** Added `clamp_out_of_range_slice_starts(gm)` FX pass in `python_package/tt_torch/backend/passes.py`. The pass iterates all `aten.slice.Tensor` nodes, reads `node.args[0].meta['val'].shape` to determine `dim_size`, and clamps any negative `start` that satisfies `start < -dim_size` to `-dim_size`. Called from `torch_pass_pipeline` in `backend.py` after `bypass_assert_tensor_metadata`. Branch: `remediation/gemma3_qat_gguf-causal_lm-pytorch-4B_IT_QAT_GGUF-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    396.11s (0:06:36)
- Tier A attempts: 1

## Files changed
- `tt-xla/python_package/tt_torch/backend/passes.py` — added `clamp_out_of_range_slice_starts`
- `tt-xla/python_package/tt_torch/backend/backend.py` — import and call new pass
- `tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1afb966ca004be6594d127a66b02a0410e93b1b5 |
| tt-forge-models | 75820233026f9d7581370663780d036247e660f9 |
