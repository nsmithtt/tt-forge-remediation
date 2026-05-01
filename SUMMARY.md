# Remediation Summary: lmstudio_qwen2_5_14b_instruct_1m_gguf-causal_lm-pytorch-14B_Instruct_1M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_qwen2_5_14b_instruct_1m_gguf/causal_lm/pytorch-14B_Instruct_1M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-missing-requirements-and-model-to-load-cross-loader-clobber

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Second error (after gguf fix): `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

## Root cause
Two independent loader bugs:

1. **Missing requirements.txt**: The model loader directory had no `requirements.txt` containing `gguf>=0.10.0`. In CI environments where gguf is not pre-installed, `is_gguf_available()` returns False and transformers raises ImportError before loading begins.

2. **Cross-loader `model_to_load` clobbering**: During pytest collection, all loader modules are imported. Several other GGUF loaders (tvall43_qwen3_5_*, unified_reward_flex_qwen35_27b_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf, etc.) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow signature `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Transformers 5.2.0 calls `load_gguf_checkpoint` with `model_to_load=dummy_model`, which raises TypeError against the stale narrow-sig patch. This model loader had no own patch to counteract it.

## Fix
**Commit 1** (`d6f19a71ff` in tt-forge-models): Added `lmstudio_qwen2_5_14b_instruct_1m_gguf/causal_lm/pytorch/requirements.txt` with `gguf>=0.10.0`.

**Commit 2** (`89ab5b2164` in tt-forge-models): Added DFS walker `_get_real_load_gguf_checkpoint()` to the loader that traverses the monkey-patch chain via `__globals__` and `__closure__` to find the original transformers function (identified by `__qualname__ == "load_gguf_checkpoint"` and `__module__ == "transformers.modeling_gguf_pytorch_utils"`). The `load_model()` method saves/restores the patched function around `AutoModelForCausalLM.from_pretrained()` so the real function is used for this loader's call.

Files changed in tt-forge-models:
- `lmstudio_qwen2_5_14b_instruct_1m_gguf/causal_lm/pytorch/requirements.txt` (new)
- `lmstudio_qwen2_5_14b_instruct_1m_gguf/causal_lm/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    564.54s (0:09:24)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/lmstudio_qwen2_5_14b_instruct_1m_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt-forge-models/lmstudio_qwen2_5_14b_instruct_1m_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f8da840b5f5de5543c02ee0d496ed01ebcd468ea |
| tt-forge-models | 89ab5b2164f5b970b07c267367aa10330122f337 |
