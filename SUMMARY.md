# Remediation Summary: outetts_gguf-text_to_speech-pytorch-Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[outetts_gguf/text_to_speech/pytorch-Q8_0-single_device-inference]

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

(The failure message given was `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` but the actual failure reproduced was the TypeError above, because gguf>=0.10.0 was already installed.)

## Root cause
Session contamination: 26 other GGUF model loaders (qwen35 and gpt-oss variants)
monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at
module import time with a narrow-signature wrapper
`def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. During
pytest collection, all loaders are imported, so these patches are applied before
the outetts test runs. transformers 5.2.0 added a `model_to_load` keyword
argument to `load_gguf_checkpoint`, which the narrow-sig wrappers reject.

The outetts_gguf loader itself has no patches; it is a victim of contamination
from other loaders collected in the same session.

Additionally, `requirements.txt` was missing for the outetts_gguf loader, so
`gguf>=0.10.0` was not declared as a dependency (though it was already installed
in the test environment).

## Fix
Two changes, both in the `tt_forge_models` submodule
(`remediation/outetts_gguf-text_to_speech-pytorch-Q8_0-single_device-inference`):

1. **26 narrow-sig GGUF patchers** — changed signature from
   `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` and updated the
   inner call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)`
   to `_orig_load_gguf_checkpoint(*args, **kwargs)`. Affected files:
   tvall43_qwen3_5_*, unified_reward_flex_qwen35_27b, gpt_oss_swallow_*, and all
   mradermacher_qwen3_5_* loaders (26 total).

2. **`outetts_gguf/text_to_speech/pytorch/requirements.txt`** — added
   `gguf>=0.10.0` to declare the dependency explicitly.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    264.11s (0:04:24)
- Tier A attempts: N/A

## Files changed
- `outetts_gguf/text_to_speech/pytorch/requirements.txt` (new)
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a0a5afcb2ec911a09bd6857a54f1e1b5370cf0a9 |
| tt-forge-models | 0e889eae824e607c2918d4eee0ff509f56d978ff |
