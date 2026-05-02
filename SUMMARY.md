# Remediation Summary: nvidia_llama_3_1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_gguf-causal_lm-pytorch-nvidia_llama_3.1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nvidia_llama_3_1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_gguf/causal_lm/pytorch-nvidia_llama_3.1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS — fixed narrow-sig `_patched_load_gguf_checkpoint` session contamination

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

Full traceback:
```
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

The pytest output ends with:
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

## Root cause
transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. 26 loaders in tt-forge-models patch `load_gguf_checkpoint` at module-level with a narrow-signature wrapper:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

Because these patches are applied at module import time (not inside `load_model`), and because the dynamic loader imports all loader modules at test-collection time, the narrow-sig patch is installed in the global `transformers` namespace before the nvidia_llama test runs. When `AutoModelForCausalLM.from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, the still-resident narrow-sig patch raises `TypeError`.

The nvidia_llama loader itself does not patch `load_gguf_checkpoint`; it is a victim of cross-loader session contamination.

## Fix
In `tt-forge-models`, updated all 26 loaders with narrow-sig patches to use `(*args, **kwargs)` and pass them through to the original function. The fix is in `remediation/nvidia_llama_3_1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_Q4_K_M_GGUF-single_device-inference` (commit `c56936e1a1`).

Files changed (26):
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
- Hardware:    blackhole-p150b
- Duration:    474.26s (0:07:54)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 loader files (see Fix section above)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c56936e1a1f53edf71dbcf08756d20c5e66ac531 |
