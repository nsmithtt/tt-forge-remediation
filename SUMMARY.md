# Remediation Summary: andycurrent-mistral-nemo-2407-12b-heretic-gguf-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[andycurrent_mistral_nemo_2407_12b_thinking_claude_gemini_gpt5_2_uncensored_heretic_gguf/causal_lm/pytorch-12B_GGUF-single_device-inference]

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
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
26 GGUF loaders in tt_forge_models (all Qwen3.5/gpt-oss-swallow variants) monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time to inject
architecture aliases (`qwen35` → `qwen3`). These patches used a narrow signature:
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`.

Transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`.
When pytest collects tests it imports all loader.py files; any of the 26 affected loaders
installs the narrow-signature patch into the global module, then when
`AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` is called for the andycurrent
model, transformers internally calls `load_gguf_checkpoint(gguf_path, return_tensors=True,
model_to_load=dummy_model)` — hitting the stale narrow-signature wrapper and raising
TypeError.

## Fix
Cherry-picked commit `d85fecea7a` from `origin/ip-172-31-30-236-tt-xla-dev/ubuntu/2026-04-27_02-32/hf-bringup-21`
onto a new branch `remediation/andycurrent-mistral-nemo-2407-12b-heretic-gguf-inference`
in tt-forge-models. The commit adds `model_to_load=None` to the `_patched_load_gguf_checkpoint`
signature in all 26 affected loaders and forwards the argument to `_orig_load_gguf_checkpoint`.

Files changed (26 loader.py files in tt-forge-models):
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

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    557.28s (0:09:17)
- Tier A attempts: N/A

## Files changed
- tt-forge-models: 26 × causal_lm/pytorch/loader.py (see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | dc94cef42ab77a7a44dc3503de019d6e4525b9a4 |
