# Remediation Summary: mungert_mirothinker_1_7_mini_gguf-causal_lm-pytorch-MiroThinker_1_7_mini_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mungert_mirothinker_1_7_mini_gguf/causal_lm/pytorch-MiroThinker_1_7_mini_Q4_K_M_GGUF-single_device-inference]

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
Two failure modes in the same test:

1. In CI at hf-bringup-38 (gguf not pre-installed):
   `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")`

2. When other GGUF loaders (Qwen3.5 variants) are imported in the same session with the narrow-sig patch at commit `0f7b734348`:
   `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

## Root cause
Two related loader bugs:

1. **Missing requirements.txt**: `mungert_mirothinker_1_7_mini_gguf/causal_lm/pytorch/requirements.txt`
   did not exist, so the test runner did not install `gguf>=0.10.0`. In CI environments where
   gguf is not pre-installed in the base venv, transformers raises an ImportError when trying
   to call `load_gguf_checkpoint`.

2. **Session contamination from narrow-sig `_patched_load_gguf_checkpoint`**: Multiple
   Qwen3.5/GPT-OSS GGUF loaders (26 total) patched
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time using
   the narrow signature `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
   without `**kwargs`. Since `transformers/modeling_utils.py:4010` does a lazy
   `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint` inside `from_pretrained`,
   the narrow-sig patched function is resolved at call time and receives
   `model_to_load=dummy_model` (added in transformers 5.2.0), causing TypeError. The
   mungert_mirothinker loader itself has no patching, but it is the victim of cross-loader
   contamination during test collection.

## Fix
Two changes in `tt-forge-models` on branch
`remediation/mungert_mirothinker_1_7_mini_gguf-causal_lm-pytorch-MiroThinker_1_7_mini_Q4_K_M_GGUF-single_device-inference`
(commit `e139c005af`):

1. Added `mungert_mirothinker_1_7_mini_gguf/causal_lm/pytorch/requirements.txt` with
   `gguf>=0.10.0` so the RequirementsManager installs gguf before running the test.

2. Changed all 26 narrow-sig `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`
   definitions to `_patched_load_gguf_checkpoint(*args, **kwargs)` with corresponding
   forwarding to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Affected loaders:
   bartowski_coniccat_qwen3_5_27b_writer_gguf, daniloreddy_qwen3_5_0_8b_gguf,
   dmind_3_mini_i1_gguf, gpt_oss_swallow_120b_rl_v0_1_gguf,
   gpt_oss_swallow_20b_rl_v0_1_gguf, gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf,
   mradermacher_bartleby_qwen3_5_4b_gguf,
   mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf,
   mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf,
   mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf,
   mradermacher_luna_qwen3_5_27b_v5_i1_gguf, mradermacher_qwen3_5_27b_gguf,
   mradermacher_qwen3_5_27b_homebrew_gguf, mradermacher_qwen3_5_27b_tainted_heresy_gguf,
   mradermacher_qwen3_5_4b_abliterated_i1_gguf, mradermacher_qwen3_5_4b_ara_heresy_v2_gguf,
   mradermacher_qwen3_5_4b_gabliterated_gguf, mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf,
   mradermacher_qwen3_5_4b_unfiltered_gguf, mradermacher_qwen3_5_4b_unredacted_max_gguf,
   mradermacher_qwen3_5_9b_abliterated_i1_gguf, mradermacher_vilm_0_8b_sft_gguf,
   qwen_3_5_imatrix_gguf, tvall43_qwen3_5_2b_heretic_v3b_i1_gguf,
   tvall43_qwen3_5_4b_heretic_v2_i1_gguf, unified_reward_flex_qwen35_27b_gguf.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    11m12s (0:11:12)
- Tier A attempts: N/A

## Files changed
- `mungert_mirothinker_1_7_mini_gguf/causal_lm/pytorch/requirements.txt` (new file)
- 26 GGUF loader `loader.py` files: narrow-sig `_patched_load_gguf_checkpoint` → `*args, **kwargs`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | e139c005afa007b706a23895d828376b6976724a |
