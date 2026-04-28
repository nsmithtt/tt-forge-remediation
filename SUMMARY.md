# Remediation Summary: nora_4b_merge_v2_gguf-causal_lm-pytorch-4B_Merge_V2_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[nora_4b_merge_v2_gguf/causal_lm/pytorch-4B_Merge_V2_GGUF-single_device-inference]

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
transformers 5.2.0 added a `model_to_load=None` keyword argument to
`load_gguf_checkpoint`. Twenty-six GGUF loaders in tt_forge_models monkey-patch
this function at module import time using a fixed signature
`(gguf_path, return_tensors=False)` that does not accept `**kwargs`. When pytest
co-collects all tests, one of those loaders (alphabetically before nora_4b_merge_v2_gguf)
patches the global `gguf_utils.load_gguf_checkpoint` reference with the broken
wrapper. The nora test then hits `TypeError` because `modeling_utils.py:4016`
calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`.

## Fix
Updated all 26 broken loaders in `tt-xla/third_party/tt_forge_models` to use
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):`
and pass `**kwargs` to `_orig_load_gguf_checkpoint(...)`.

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

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    360.57s (0:06:00)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models (26 loader.py files, see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 45571cde816788c825f7b436570eaa2fd569d990 |
| tt-forge-models | 6bc3a81cbb27aecdba2bee8e93ae916cecf32058 |
