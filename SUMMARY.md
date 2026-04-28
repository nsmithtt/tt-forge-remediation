# Remediation Summary: gemma_2b_it_gguf/causal_lm/pytorch-2B_IT_GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2b_it_gguf/causal_lm/pytorch-2B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-gemma-v1-arch-not-registered-transformers-5x

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9247366921559638. Required: pcc=0.95.

(Reproduction produced two underlying loader bugs before reaching PCC measurement:
1. ValueError: GGUF model with architecture gemma is not supported yet.
2. TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause
Two loader-layer bugs caused the test to fail:

**Bug 1 — Gemma v1 architecture missing from transformers 5.x GGUF support**

Transformers 5.x dropped `gemma` (v1) from `GGUF_CONFIG_MAPPING`; only `gemma2`
and `gemma3` remain. The `lmstudio-ai/gemma-2b-it-GGUF` file stores
`general.architecture = 'gemma'`, so `load_gguf_checkpoint` raised:
  `ValueError: GGUF model with architecture gemma is not supported yet.`

Additionally the GGUF v3 tokenizer for Gemma v1 embeds no `chat_template`,
so `apply_chat_template` raised `ValueError: tokenizer.chat_template is not set`.

**Bug 2 — 26 GGUF loaders patched `load_gguf_checkpoint` with a narrow signature**

Transformers 5.2.0 added `model_to_load` keyword arg to `load_gguf_checkpoint`.
26 other loaders in tt_forge_models (qwen35 and gpt-oss families) monkey-patched
this function at module import time with signature `(gguf_path, return_tensors=False)`.
During collection pytest imports all loaders, so the patched narrow-signature
function was the active one when the gemma test ran. Transformers passed
`model_to_load=dummy_model`, which the narrow-signature function rejected with
`TypeError`.

## Fix
All fixes are in `tt_forge_models` on branch
`remediation/gemma_2b_it_gguf-causal_lm-pytorch-2B_IT_GGUF-single_device-inference`.

**Fix 1** (`gemma_2b_it_gguf/causal_lm/pytorch/loader.py`):
- Added `_patch_gemma_v1_support()` called at module import time. This registers
  `gemma` into `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with the correct field
  mappings (matching the actual GGUF metadata), adds `gemma` to
  `GGUF_SUPPORTED_ARCHITECTURES`, registers `Gemma2TensorProcessor` for `gemma`
  in `TENSOR_PROCESSORS` (needed for the -1 norm weight adjustment), and registers
  `GGUFGemmaConverter` for `gemma` in `GGUF_TO_FAST_CONVERTERS`.
- Fixed `load_inputs` to fall back to raw `sample_text` when
  `tokenizer.chat_template is None`.

**Fix 2** (26 files across qwen35/gpt-oss GGUF loaders):
- Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` to
  `def _patched_load_gguf_checkpoint(*args, **kwargs):` and updated the inner call
  to `_orig_load_gguf_checkpoint(*args, **kwargs)` in all 26 affected loaders.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    348.07s (0:05:48)
- Tier A attempts: N/A

## Files changed
- `third_party/tt_forge_models/gemma_2b_it_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1bfdc312a813c41331c4f313a15c7d97df348849 |
| tt-forge-models | 22a8f5527934d904b1fe81d97b284fdecd610c7a |
