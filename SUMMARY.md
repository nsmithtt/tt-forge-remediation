# Remediation Summary: den4iiks_l3_8b_stheno_v3_2_gguf_iq_imatrix-causal_lm-pytorch-Q4_K_M-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[den4iiks_l3_8b_stheno_v3_2_gguf_iq_imatrix/causal_lm/pytorch-Q4_K_M-single_device-inference]

## Result
SILICON_PASS — loader fix: update _patched_load_gguf_checkpoint to accept *args/**kwargs for transformers 5.x model_to_load compat

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Full context: `transformers/modeling_utils.py:4016` calls `load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)`, but a monkey-patch installed by another GGUF model loader during test collection had the old signature `(gguf_path, return_tensors=False)`.

## Root cause
During pytest test collection, `TorchDynamicLoader.setup_test_discovery()` imports every `loader.py` in `tt_forge_models`. Several loaders (for qwen3.5 and gpt-oss variants) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module-level with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrapper. Transformers 5.x introduced a `model_to_load` keyword argument in the `load_gguf_checkpoint` call path. Because the patch runs at collection time and persists for the whole session, when the den4iiks test later calls `AutoModelForCausalLM.from_pretrained()`, the local import inside `modeling_utils.py` picks up the patched (narrow-signature) version and raises `TypeError`.

## Fix
Updated all 26 GGUF loaders in `tt_forge_models` that had the narrow `(gguf_path, return_tensors=False)` signature on `_patched_load_gguf_checkpoint` to use `(*args, **kwargs)`, forwarding all arguments to the original. The fix already existed in `tt_forge_models` at commit `5d5c309654` on branch `arch-c-36-tt-xla-dev/nsmith/hf-bringup-12`. Updated the tt-xla submodule pointer to include this commit.

Files changed (in tt_forge_models):
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
- Duration:    441.71s (0:07:21)
- Tier A attempts: N/A

## Files changed
- tt_forge_models: 26 loader.py files (signature fix for _patched_load_gguf_checkpoint)
- tt-xla: third_party/tt_forge_models submodule pointer updated to 5d5c309654

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | dc33b4c13 (remediation branch) |
| tt-forge-models | 5d5c309654dced79fe59a7ddf07390a724760f76 |
