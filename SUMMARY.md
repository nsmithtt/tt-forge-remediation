# Remediation Summary: llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf-causal_lm-pytorch-Llama3_1_SuperNovaLite_HuatuoSkywork_o1_8B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf/causal_lm/pytorch-Llama3.1_SuperNovaLite_HuatuoSkywork_o1_8B_i1_GGUF-single_device-inference]

## Result
SILICON_PASS — loader fixes for missing requirements.txt and bad _patched_load_gguf_checkpoint signature

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
(CI reported: `ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.`)

## Root cause
Two loader-layer bugs:

1. **Missing requirements.txt**: The `llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf` loader has no `requirements.txt` listing `gguf>=0.10.0`. In CI environments where `gguf` is not pre-installed, `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` raises `ImportError` before reaching the model load. This is the error reported in CI.

2. **Bad _patched_load_gguf_checkpoint signature**: 26 other GGUF loaders in `tt_forge_models` (qwen3.5 and gpt-oss-swallow variants) monkey-patch `load_gguf_checkpoint` at import time with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):`. Transformers 5.x added `model_to_load=None` to the signature. When these loaders are imported during pytest collection alongside this test (even in a focused run), the bad patcher overwrites `load_gguf_checkpoint` and rejects the `model_to_load` kwarg with TypeError.

## Fix
Two changes in `tt_forge_models` on remediation branch `remediation/llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf-causal_lm-pytorch-Llama3_1_SuperNovaLite_HuatuoSkywork_o1_8B_i1_GGUF-single_device-inference`:

1. Added `llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.

2. Fixed 26 loader files by changing `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` / `result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `def _patched_load_gguf_checkpoint(*args, **kwargs):` / `result = _orig_load_gguf_checkpoint(*args, **kwargs)`. This forwards all current and future positional/keyword arguments through the patcher chain.

Files changed (in tt_forge_models):
- `llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
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

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    506.63s (0:08:26)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/llama3_1_supernovalite_huatuoskywork_o1_8b_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- 26 `tt_forge_models/*/causal_lm/pytorch/loader.py` files (patched patcher signature)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7a8016bbae3d941090a03c574fcb9b6b6a878f25 |
| tt-forge-models | 88ad1e6ba08bb9250c7573d75723a6fe5e04c460 |
