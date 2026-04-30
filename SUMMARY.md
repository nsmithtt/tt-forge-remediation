# Remediation Summary: glm_z1_9b_0414_heretic_gguf-causal_lm-pytorch-Z1_9B_0414_heretic_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_z1_9b_0414_heretic_gguf/causal_lm/pytorch-Z1_9B_0414_heretic_GGUF-single_device-inference]

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
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
26 Qwen3.5 GGUF loaders define a `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` function that patches `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time. This signature is missing the `model_to_load=None` parameter added in transformers 5.x.

The mechanism that makes this affect the GLM-Z1 test:

1. `bartowski_glm_4_9b_chat_gguf` is the first good patcher — it registers `"chatglm"` in `GGUF_SUPPORTED_ARCHITECTURES` and installs a correct `patched_load_gguf_checkpoint(*args, **kwargs)`.
2. `daniloreddy_qwen3_5_0_8b_gguf` and `dmind_3_mini_i1_gguf` run after (alphabetically), each installing the bad `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` on `modeling_gguf_pytorch_utils.load_gguf_checkpoint`.
3. All subsequent good GLM loaders (`glm_4_32b_0414_gguf`, `glm_4_7_flash_gguf`, etc.) have an early-return guard: `if "chatglm" in GGUF_SUPPORTED_ARCHITECTURES: return`. Since "chatglm" was already registered, they skip re-patching.
4. When the GLM-Z1 test runs, `modeling_utils.py:4010` does a lazy import `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint`, retrieving the bad `_patched_load_gguf_checkpoint`.
5. `modeling_utils.py:4016` then calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`, which fails because the bad patcher doesn't accept `model_to_load`.

## Fix
In all 26 bad loaders, changed:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
```

Files changed in `tt-forge-models` on branch `remediation/glm_z1_9b_0414_heretic_gguf-causal_lm-pytorch-Z1_9B_0414_heretic_GGUF-single_device-inference`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    732.90s (0:12:12)
- Tier A attempts: N/A

## Files changed
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
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | bd350fe43b9a3df352e417df3ab1e5a757d76613 |
