# Remediation Summary: apriel_1_6_15b_thinker_heretic_i1_gguf-causal_lm-pytorch-Apriel_1_6_15b_Thinker_heretic_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[apriel_1_6_15b_thinker_heretic_i1_gguf/causal_lm/pytorch-Apriel_1_6_15b_Thinker_heretic_i1_GGUF-single_device-inference]

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

## Root cause
26 GGUF loader modules in tt_forge_models each define a `_patched_load_gguf_checkpoint` wrapper and monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. The wrappers had a narrow signature `(gguf_path, return_tensors=False)`. When pytest collects `test_models.py` it imports all loader modules, and the last-imported narrow-signature loader leaves its broken patch as the globally-active function. Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` inside `from_pretrained`, which then fails with TypeError because the patched replacement does not accept `model_to_load`. The apriel loader itself does no patching — it is a victim of a stale patch left by another loader.

## Fix
Changed the function signature of `_patched_load_gguf_checkpoint` in all 26 affected loaders from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)`, and updated the forwarding call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)`. This allows the transformers 5.x `model_to_load` kwarg to pass through to the original function regardless of which loader's patch is active.

Changed in repo: tt-forge-models
Branch: remediation/apriel_1_6_15b_thinker_heretic_i1_gguf-causal_lm-pytorch-Apriel_1_6_15b_Thinker_heretic_i1_GGUF-single_device-inference
Commit: fb9624acc74414fec3b65428c2c121e5913a5f95

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    627.34s (0:10:27)
- Tier A attempts: N/A

## Files changed
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
| tt-xla          | 95df27b512cc495abd6c9cab1fd43b4491b17ed8 |
| tt-forge-models | fb9624acc74414fec3b65428c2c121e5913a5f95 |
