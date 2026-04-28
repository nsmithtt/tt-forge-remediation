# Remediation Summary: aisingapore_llama_sea_lion_v3_5_8b_r_gguf-causal_lm-pytorch-Llama_SEA_LION_v3_5_8B_R_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[aisingapore_llama_sea_lion_v3_5_8b_r_gguf/causal_lm/pytorch-Llama_SEA_LION_v3_5_8B_R_GGUF-single_device-inference]

## Result
SILICON_PASS — loader bugs fixed: missing gguf requirement and model_to_load kwarg not forwarded

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
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Two loader-layer bugs:

1. **Missing gguf requirement**: `aisingapore_llama_sea_lion_v3_5_8b_r_gguf/causal_lm/pytorch/` had no `requirements.txt`, so `gguf>=0.10.0` was not declared as a dependency. On CI environments where gguf was not installed, `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` raised the ImportError before any model weights could be loaded.

2. **Narrow-signature GGUF patch (cross-contamination)**: 26 other GGUF loaders (Qwen3.5, Swallow, mradermacher, etc.) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module-import time with a narrow signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Since `setup_test_discovery` imports ALL loaders during pytest collection, one of these patches is installed before the aisingapore test runs. Transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`; calling the patched function with `model_to_load=dummy_model` raises `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

## Fix
1. Added `aisingapore_llama_sea_lion_v3_5_8b_r_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0` (new file, 1 line).

2. Cherry-picked commit `eaee402cd2` from tt-forge-models, which updates all 26 GGUF loaders with narrow-signature patches to use `_patched_load_gguf_checkpoint(*args, **kwargs)` and forward all arguments to `_orig_load_gguf_checkpoint`. This prevents the TypeError regardless of which GGUF loaders are imported during collection.

Both changes are in tt-forge-models on branch `remediation/aisingapore_llama_sea_lion_v3_5_8b_r_gguf-causal_lm-pytorch-Llama_SEA_LION_v3_5_8B_R_GGUF-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    465.47s (0:07:45)
- Tier A attempts: N/A

## Files changed
- `aisingapore_llama_sea_lion_v3_5_8b_r_gguf/causal_lm/pytorch/requirements.txt` (added)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `gpt_oss_swallow_20b_sft_v0_1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_qwen_3_5_27b_derestricted_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py` (model_to_load kwarg fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 03fe65e92bbed5ef29788f2c4503dfd03284a703 |
| tt-forge-models | 5ba2cada9a7dd2a158a8f5037dde40ee429bd6de |
