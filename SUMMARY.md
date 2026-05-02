# Remediation Summary: mradermacher_next_14b_i1_gguf-causal_lm-pytorch-14B_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_next_14b_i1_gguf/causal_lm/pytorch-14B_I1_Q4_K_M_GGUF-single_device-inference]

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
Cross-loader clobbering in the loader layer. 26 GGUF loaders (Qwen3.5 variants,
gpt_oss_swallow variants, etc.) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
at module import time with a narrow signature `(gguf_path, return_tensors=False)`.
transformers 5.2.0 added a `model_to_load=` keyword argument to this function.
When pytest collects tests, one of the 26 narrow-sig loaders is imported and
installs the stale wrapper. When `mradermacher_next_14b_i1_gguf` later calls
`AutoModelForCausalLM.from_pretrained`, transformers re-imports
`load_gguf_checkpoint` inside `modeling_utils.py` and picks up the patched
version, which rejects the new kwarg with a TypeError.

## Fix
Updated all 26 GGUF loaders in `tt_forge_models` to use `(*args, **kwargs)` for
both the function signature and the inner call to `_orig_load_gguf_checkpoint`.
Changed from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    ...
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
To:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    ...
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
Commit `c43fcd48d2` in tt-forge-models on branch
`remediation/mradermacher_next_14b_i1_gguf-causal_lm-pytorch-14B_I1_Q4_K_M_GGUF-single_device-inference`.

Affected loaders fixed (26 total):
- unified_reward_flex_qwen35_27b_gguf, tvall43_qwen3_5_2b_heretic_v3b_i1_gguf,
  tvall43_qwen3_5_4b_heretic_v2_i1_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf,
  mradermacher_qwen3_5_9b_abliterated_i1_gguf, mradermacher_qwen3_5_4b_unredacted_max_gguf,
  mradermacher_qwen3_5_4b_gabliterated_gguf, mradermacher_qwen3_5_4b_unfiltered_gguf,
  mradermacher_vilm_0_8b_sft_gguf, mradermacher_qwen3_5_4b_abliterated_i1_gguf,
  mradermacher_qwen3_5_4b_ara_heresy_v2_gguf, mradermacher_qwen3_5_27b_gguf,
  mradermacher_qwen3_5_27b_homebrew_gguf, mradermacher_qwen3_5_27b_tainted_heresy_gguf,
  mradermacher_luna_qwen3_5_27b_v5_i1_gguf, mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf,
  mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf, mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf,
  mradermacher_bartleby_qwen3_5_4b_gguf, mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf,
  gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf, gpt_oss_swallow_20b_rl_v0_1_gguf,
  qwen_3_5_imatrix_gguf, dmind_3_mini_i1_gguf, daniloreddy_qwen3_5_0_8b_gguf,
  bartowski_coniccat_qwen3_5_27b_writer_gguf

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    526.52s (0:08:46)
- Tier A attempts: N/A

## Files changed
- tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c43fcd48d2bc4fc944bd1148cff061f1abb7ceb0 |
