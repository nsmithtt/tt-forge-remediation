# Remediation Summary: dolphin_2_1_mistral_7b_gguf-causal_lm-pytorch-2.1_Mistral_7B_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[dolphin_2_1_mistral_7b_gguf/causal_lm/pytorch-2.1_Mistral_7B_GGUF-single_device-inference]

## Result
SILICON_PASS — two loader fixes applied: _patched_load_gguf_checkpoint kwarg compat + apply_chat_template guard

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
2026-04-23 21:00:59.755 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

When reproduced on current branch, the failure presented as:
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

The original timeout was caused by the same root failure: the model never loaded
successfully due to the loader error, resulting in a device hang during cleanup.

## Root cause
Two loader-layer bugs:

1. **_patched_load_gguf_checkpoint signature** (26 loaders): Several GGUF loaders
   (gpt_oss_swallow_*, mradermacher_qwen3_5_*, daniloreddy_*, dmind_*, tvall43_*,
   qwen_3_5_imatrix, unified_reward_flex_qwen35_27b, bartowski_coniccat_qwen3_5_27b)
   monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at
   import time with a narrow signature:
   ```python
   def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
   ```
   transformers 5.2.0 calls this function with `model_to_load=dummy_model`, raising
   `TypeError`. Because the patch is applied at import time and persists for the entire
   pytest session, any GGUF model imported first during collection poisons subsequent
   GGUF model loads — including dolphin_2_1_mistral_7b_gguf.

2. **apply_chat_template without chat_template** (dolphin loader): The GGUF tokenizer
   from `TheBloke/dolphin-2.1-mistral-7B-GGUF` does not ship a `chat_template`.
   `load_inputs` unconditionally called `tokenizer.apply_chat_template()`, raising
   `ValueError: Cannot use apply_chat_template() because tokenizer.chat_template is
   not set`.

## Fix
Both fixes are in the `remediation/dolphin_2_1_mistral_7b_gguf-causal_lm-pytorch-2.1_Mistral_7B_GGUF-single_device-inference` branch of tt-forge-models (commit 6c94c63a24).

**Fix 1** (commit 00a512a63b) — 26 files in tt-forge-models:
Changed `_patched_load_gguf_checkpoint` from a narrow explicit signature to `*args, **kwargs`:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_arch_support()
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
    ...
```

**Fix 2** (commit 6c94c63a24) — `dolphin_2_1_mistral_7b_gguf/causal_lm/pytorch/loader.py`:
Guarded `apply_chat_template` with `if self.tokenizer.chat_template is not None`,
falling back to `self.sample_text` directly when no template is set.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    284.70s (0:04:44)
- Tier A attempts: N/A

## Files changed
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
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
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `dolphin_2_1_mistral_7b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 6c94c63a24dc343fa1510098c3b3b02a2efbb8a2 |
