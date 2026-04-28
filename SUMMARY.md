# Remediation Summary: bartowski_lumimaid_magnum_v4_12b_gguf-causal_lm-pytorch-Lumimaid_Magnum_v4_12B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_lumimaid_magnum_v4_12b_gguf/causal_lm/pytorch-Lumimaid_Magnum_v4_12B_GGUF-single_device-inference]

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
At test collection time, pytest imports all loader modules to enumerate variants. Several Qwen3.5 and GPT-OSS GGUF loaders replace `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` function that doesn't accept `model_to_load`. transformers 5.2.0 added `model_to_load` as a third positional/keyword argument to `load_gguf_checkpoint`, and calls it from `modeling_utils.py` with `model_to_load=dummy_model`. Because the module-level patch persists in the Python process, the bartowski Lumimaid loader (which uses the standard unpatched path) inherits the broken patched function and raises `TypeError`.

## Fix
Updated 26 GGUF loaders in `tt_forge_models` that defined `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to use `(*args, **kwargs)` and pass them through to `_orig_load_gguf_checkpoint(*args, **kwargs)`. This makes the patch forward-compatible with the `model_to_load` kwarg added in transformers 5.2.0.

Files changed (all in `tt-xla/third_party/tt_forge_models`):
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    498.90s (0:08:18)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/` — 26 loader files (signature fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a04871580d8a8500547ee419e976e22af4abce1c |
| tt-forge-models | 88df9e929ab364ba98b7b476db922f0f48ae51e0 |
