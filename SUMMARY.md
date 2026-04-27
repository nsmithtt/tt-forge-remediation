# Remediation Summary: aura_7b_gguf/causal_lm/pytorch-7B_GGUF-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[aura_7b_gguf/causal_lm/pytorch-7B_GGUF-single_device-inference]

## Result
SILICON_PASS

## Failure
2026-04-23 23:40:11.555 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

## Root cause
Loader layer — two compounding issues in `aura_7b_gguf/causal_lm/pytorch`:

1. **Missing requirements**: `aura_7b_gguf` had no `requirements.txt`. With `transformers==5.5.1`,
   loading a GGUF file via `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)` requires
   `gguf>=0.10.0` (GGUF binary parser) and `accelerate` (weight loading). Without them,
   `from_pretrained` raised `ImportError`/`ValueError` before the device was initialised, which
   surfaced as a device-hang timeout in CI.

2. **Broken monkey-patch from co-loaded models**: When pytest collects all parametrised model tests
   in the same session, ~26 other GGUF loaders (Qwen3.5, gpt-oss-swallow, etc.) each replace
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with a
   `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that does not accept
   `model_to_load`. `transformers 5.x` added `model_to_load=None` to that function and now
   passes it via keyword from `modeling_utils.py:4078`. When the aura test runs after any of
   those loaders, `from_pretrained` hits the patched version and raises
   `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

## Fix
All changes are in `tt-forge-models` on branch `remediation/aura-7b-gguf-fix`.

**Fix 1** — `aura_7b_gguf/causal_lm/pytorch/requirements.txt` (new file):
```
gguf>=0.10.0
accelerate
```
Commit `7f6f7950529e25003df69f5cb879ec79e3423d1f`.

**Fix 2** — Updated `_patched_load_gguf_checkpoint` in 26 GGUF loader files from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
so that `model_to_load` (and any future kwargs) are forwarded to the original function.
Commit `e267452d13293bc4eacfd322890494547f48a5f0`.

Neither fix trims the model, offloads to CPU, changes input shapes, or lowers PCC thresholds.

## Verification
SILICON_PASS — pytest exited 0, 1 passed in 438.57s (7:18 wall clock), blackhole hardware
(hostname: bh-lb-13-tt-forge-remediation-1)

## Files changed
- `aura_7b_gguf/causal_lm/pytorch/requirements.txt` (new)
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
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
| tt-xla          | 93a3b963864119113484e8cc3b3494f93f2982df |
| tt-forge-models | 7f6f7950529e25003df69f5cb879ec79e3423d1f |
