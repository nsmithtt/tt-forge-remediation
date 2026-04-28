# Remediation Summary: bartowski_smallthinker_3b_preview_gguf-causal_lm-pytorch-SmallThinker_3B_Preview_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_smallthinker_3b_preview_gguf/causal_lm/pytorch-SmallThinker_3B_Preview_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(CI symptom reported as: /bin/bash: line 1: python: command not found — this is a CI environment issue where `python` is not on PATH, distinct from the actual test failure which is the TypeError above.)

## Root cause
When pytest collects all GGUF loader modules during test discovery, 26 of them install a module-level monkey-patch on `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with the old 2-arg signature `(gguf_path, return_tensors=False)`. Transformers 5.x added a `model_to_load=None` keyword argument to this function. The last such patcher installed during collection remains active when the SmallThinker test runs. `AutoModelForCausalLM.from_pretrained` calls `load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)`, which hits the patcher and raises the TypeError. The SmallThinker loader itself does not patch this function — it is a victim of another loader's stale global patch.

## Fix
Updated all 26 GGUF loader files that had the broken `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature. Each was changed to:
1. `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):`
2. `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)`

Committed on `remediation/bartowski_smallthinker_3b_preview_gguf-causal_lm-pytorch-SmallThinker_3B_Preview_GGUF-single_device-inference` in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    328.07s (0:05:28)
- Tier A attempts: N/A

## Files changed
- 26 GGUF loader files in `tenstorrent/tt-forge-models`:
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

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | f6e2a7ef96fd948e6d6babcef77fa62ae1cd9748 |
