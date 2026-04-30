# Remediation Summary: dpopenhermes_7b_i1_gguf-causal_lm-pytorch-DPOpenHermes_7B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dpopenhermes_7b_i1_gguf/causal_lm/pytorch-DPOpenHermes_7B_i1_GGUF-single_device-inference]

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
26 loaders (gpt_oss_swallow*, mradermacher_qwen3_5*, tvall43_*, and others) patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a
function `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that is missing the
`model_to_load` parameter added in transformers 5.x.

When pytest collects `test_models.py`, it calls `TorchDynamicLoader.setup_test_discovery()` which
imports all loader modules. Any of these 26 loaders, when imported, globally patches
`load_gguf_checkpoint`. The patch persists for the entire session. When the dpopenhermes test
then runs and calls `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, transformers
internally calls `load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model)`
which fails with TypeError because the patched version doesn't accept `model_to_load`.

## Fix
In `tt-forge-models` on branch `remediation/dpopenhermes_7b_i1_gguf-single_device-inference`:

Added `model_to_load=None` parameter to `_patched_load_gguf_checkpoint` and passed it through
to `_orig_load_gguf_checkpoint` in all 26 affected loaders:

- bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

Change in each file (26 files changed, 52 insertions(+), 52 deletions(-)):
```python
# Before
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)

# After
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, model_to_load=model_to_load)
```

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    284.12s (0:04:44)
- Tier A attempts: N/A

## Files changed
- tt-forge-models: 26 loader files (see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a6ef25f205e0eb6c31249a78102f99bb8795b0a6 |
| tt-forge-models | 04179c673f407877ab2dc9a75bf653aec7bf0f50 |
