# Remediation Summary: lmstudio_gemma_3_4b_qat_gguf-causal_lm-pytorch-4B_IT_QAT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_gemma_3_4b_qat_gguf/causal_lm/pytorch-4B_IT_QAT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

Then after fixing:
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_37, 2, -1023, 9223372036854775807), kwargs = {})

## Root cause
**Bug 1 (loader):** 26 GGUF loaders patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a function that only accepts `(gguf_path, return_tensors=False)`. Transformers 5.x added a `model_to_load=` keyword to `load_gguf_checkpoint`. When pytest discovers all loaders during collection, one of these broken patches becomes active. `modeling_utils.py` imports `load_gguf_checkpoint` lazily (inside `from_pretrained`) and therefore gets the patched version, which raises `TypeError` when transformers passes `model_to_load=dummy_model`.

**Bug 2 (tt-xla):** Gemma 3's sliding-window attention (window=1024) caches KV states with `full_value_states[:, :, -sliding_window+1:, :]`. With `max_length=128` the actual token count is 23, so the start index is `-1023`, which is outside `[-23, 22]`. PyTorch's CPU slice silently clamps such indices; XLA's slice kernel validates bounds strictly and raises `RuntimeError`. This happens during `partition_fx_graph_for_cpu_fallback` graph replay before compilation.

## Fix
**Fix 1 (tt-forge-models):** In 26 loaders whose `_patched_load_gguf_checkpoint` had the fixed signature `(gguf_path, return_tensors=False)`, added `model_to_load=None, **kwargs` and forwarded `model_to_load` to the original function. Branch: `remediation/lmstudio_gemma_3_4b_qat_gguf-causal_lm-pytorch-4B_IT_QAT_GGUF-single_device-inference` in tt-forge-models.

**Fix 2 (tt-xla):** Added `clamp_out_of_range_slice_starts` FX pass in `python_package/tt_torch/backend/passes.py`. The pass walks the graph and replaces any `aten.slice.Tensor` start argument that is below `-dim_size` with `-dim_size`, matching PyTorch's clamping semantics without changing the computed result. Applied in `torch_pass_pipeline` in `backend.py`. Branch: `remediation/lmstudio_gemma_3_4b_qat_gguf-causal_lm-pytorch-4B_IT_QAT_GGUF-single_device-inference` in tt-xla.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    415.54s (0:06:55)
- Tier A attempts: N/A

## Files changed
**tt-forge-models:**
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

**tt-xla:**
- `python_package/tt_torch/backend/passes.py` (add `clamp_out_of_range_slice_starts`)
- `python_package/tt_torch/backend/backend.py` (import and apply the new pass)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (add EXPECTED_PASSING)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5f6d4e341f62b2ce7144a02e57d0edf01bfb4ae4 |
| tt-forge-models | 308f44b8788ed8928d63018781a9c266140123fc |
