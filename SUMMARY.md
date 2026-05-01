# Remediation Summary: bartowski_phi_3_mini_4k_instruct_gguf-causal_lm-pytorch-Phi_3_Mini_4K_Instruct_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_phi_3_mini_4k_instruct_gguf/causal_lm/pytorch-Phi_3_Mini_4K_Instruct_GGUF-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed: missing requirements.txt and bad _patched_load_gguf_checkpoint signature in 26 loaders

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-missing-requirements-and-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

Locally (with gguf installed), the failure manifested as:
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

## Root cause
Two loader-layer bugs:

1. **Missing requirements.txt**: `bartowski_phi_3_mini_4k_instruct_gguf/causal_lm/pytorch/` had no `requirements.txt`. In CI environments where `gguf` is not pre-installed, `is_gguf_available()` returns False inside `load_gguf_checkpoint`, producing the ImportError. Fix: add `requirements.txt` with `gguf>=0.10.0`.

2. **Bad _patched_load_gguf_checkpoint signature in 26 GGUF loaders**: pytest imports all loader modules at collection time. 26 loaders globally patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signature that is missing `**kwargs`. Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which raises `TypeError` when the patched version is in place. The patches form a chain; every link must accept and forward `**kwargs`.

## Fix
Both fixes are in the `tt_forge_models` submodule on branch `remediation/bartowski_phi_3_mini_4k_instruct_gguf-causal_lm-pytorch-Phi_3_Mini_4K_Instruct_GGUF-single_device-inference`:

1. Added `bartowski_phi_3_mini_4k_instruct_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0`.

2. Fixed all 26 loaders that had `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` by adding `**kwargs` to the signature and forwarding it to the inner `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)` call. Affected files: `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `dmind_3_mini_i1_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `mradermacher_bartleby_qwen3_5_4b_gguf`, `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf`, `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`, `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf`, `mradermacher_luna_qwen3_5_27b_v5_i1_gguf`, `mradermacher_qwen3_5_27b_gguf`, `mradermacher_qwen3_5_27b_homebrew_gguf`, `mradermacher_qwen3_5_27b_tainted_heresy_gguf`, `mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_gabliterated_gguf`, `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf`, `mradermacher_qwen3_5_4b_unfiltered_gguf`, `mradermacher_qwen3_5_4b_unredacted_max_gguf`, `mradermacher_qwen3_5_9b_abliterated_i1_gguf`, `mradermacher_vilm_0_8b_sft_gguf`, `qwen_3_5_imatrix_gguf`, `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`, `unified_reward_flex_qwen35_27b_gguf` (all `/causal_lm/pytorch/loader.py`).

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    259.26s (0:04:19)
- Tier A attempts: N/A

## Files changed
- `bartowski_phi_3_mini_4k_instruct_gguf/causal_lm/pytorch/requirements.txt` (created)
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

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b23592443e52e66d80b94e6e95c46b07b1b38dc9 |
| tt-forge-models | bfa4b7e8174412e13555c23c0b7de454a187c075 |
