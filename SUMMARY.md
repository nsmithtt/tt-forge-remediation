# Remediation Summary: qwen3_8b_192k_josiefied_uncensored_neo_max_gguf-causal_lm-pytorch-8B_192K_Josiefied_Uncensored_NEO_Max_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen3_8b_192k_josiefied_uncensored_neo_max_gguf/causal_lm/pytorch-8B_192K_Josiefied_Uncensored_NEO_Max_Q4_K_M-single_device-inference]

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

The originally-reported failure was:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

## Root cause
Two loader bugs:

1. **Missing `requirements.txt`**: The model directory had no `requirements.txt`, so `gguf>=0.10.0` was not declared as a dependency. When `gguf` was absent or too old, transformers raised `ImportError` before attempting to load the checkpoint.

2. **Session contamination via narrow-sig `_patched_load_gguf_checkpoint`**: 26 other GGUF loaders (all qwen35/gpt-oss variants) define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — a narrow signature — and install it globally at module import time via `_gguf_utils.load_gguf_checkpoint = _patched_load_gguf_checkpoint`. During pytest collection, these loaders are imported to enumerate test variants, silently replacing the real `load_gguf_checkpoint`. When transformers 5.2.0 later calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`, the narrow-sig wrapper rejects the new `model_to_load` keyword argument with a `TypeError`.

## Fix
In `tt-forge-models` on branch `remediation/qwen3_8b_192k_josiefied_uncensored_neo_max_gguf-causal_lm-pytorch-8B_192K_Josiefied_Uncensored_NEO_Max_Q4_K_M-single_device-inference`:

**Commit 1** — `qwen3_8b_192k_josiefied_uncensored_neo_max_gguf/causal_lm/pytorch/requirements.txt` (new file):
```
gguf>=0.10.0
```

**Commit 2** — 26 loader files updated to pass `**kwargs` through:
```python
# Before
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, **kwargs):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)
```

Affected loaders: tvall43_qwen3_5_4b_heretic_v2_i1_gguf, tvall43_qwen3_5_2b_heretic_v3b_i1_gguf, unified_reward_flex_qwen35_27b_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf, gpt_oss_swallow_20b_rl_v0_1_gguf, gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf, qwen_3_5_imatrix_gguf, dmind_3_mini_i1_gguf, daniloreddy_qwen3_5_0_8b_gguf, bartowski_coniccat_qwen3_5_27b_writer_gguf, and 16 `mradermacher_*` loaders.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    439.65s (0:07:19)
- Tier A attempts: N/A

## Files changed
- `qwen3_8b_192k_josiefied_uncensored_neo_max_gguf/causal_lm/pytorch/requirements.txt` (new)
- 26 × `*/causal_lm/pytorch/loader.py` (signature fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9b38bda47987826df805c84d616b3fd2c2c38da7 |
| tt-forge-models | 7dba9e97e982650f8e5e55433dba2303dce51223 |
