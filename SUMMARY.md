# Remediation Summary: deepseek_r1_distill_qwen_14b_gguf

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_14b_gguf/causal_lm/pytorch-DeepSeek_R1_Distill_Qwen_14B_GGUF-single_device-inference]

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
```
third_party/tt_forge_models/deepseek_r1_distill_qwen_14b_gguf/causal_lm/pytorch/loader.py:109: in load_model
    model = AutoModelForCausalLM.from_pretrained(
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
26 GGUF loaders on this branch monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time using a narrow signature `(gguf_path, return_tensors=False)`. transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. During pytest collection all model modules are imported, so whichever of these loaders is collected first installs the patched (narrow) function globally. When `DeepSeek_R1_Distill_Qwen_14B_GGUF` loader subsequently calls `from_pretrained`, transformers passes `model_to_load=dummy_model` to the globally-patched function, raising TypeError. The deepseek_r1_distill_qwen_14b_gguf loader itself does not patch `load_gguf_checkpoint`; it is a victim of the cross-test pollution.

## Fix
Updated all 26 GGUF loaders in `tt-forge-models` that had the narrow `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature to use `(*args, **kwargs)`, and updated the corresponding call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)`.

Files changed (all in `third_party/tt_forge_models/`, each at `<model>/causal_lm/pytorch/loader.py`):
- bartowski_coniccat_qwen3_5_27b_writer_gguf
- daniloreddy_qwen3_5_0_8b_gguf
- dmind_3_mini_i1_gguf
- gpt_oss_swallow_120b_rl_v0_1_gguf
- gpt_oss_swallow_20b_rl_v0_1_gguf
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf
- mradermacher_bartleby_qwen3_5_4b_gguf
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf
- mradermacher_qwen3_5_27b_gguf
- mradermacher_qwen3_5_27b_homebrew_gguf
- mradermacher_qwen3_5_27b_tainted_heresy_gguf
- mradermacher_qwen3_5_4b_abliterated_i1_gguf
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf
- mradermacher_qwen3_5_4b_gabliterated_gguf
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf
- mradermacher_qwen3_5_4b_unfiltered_gguf
- mradermacher_qwen3_5_4b_unredacted_max_gguf
- mradermacher_qwen3_5_9b_abliterated_i1_gguf
- mradermacher_vilm_0_8b_sft_gguf
- qwen_3_5_imatrix_gguf
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf
- unified_reward_flex_qwen35_27b_gguf

Committed on branch `remediation/deepseek_r1_distill_qwen_14b_gguf` in `tt-forge-models` at commit `ae9738ac418ca3457619e31fe48a8dddef6a39cc`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    613.14s (0:10:13)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 loader.py files updated (see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | ae9738ac418ca3457619e31fe48a8dddef6a39cc |
