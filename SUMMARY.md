# Remediation Summary: mradermacher_blossom_v6_7b_i1_gguf-causal_lm-pytorch-V6_7B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mradermacher_blossom_v6_7b_i1_gguf/causal_lm/pytorch-V6_7B_i1_GGUF-single_device-inference]

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
Loader-layer bug. During pytest collection, all loader modules are imported. 26
Qwen3.5 GGUF loaders (bartowski_coniccat_qwen3_5_27b_writer_gguf, dmind_3_mini_i1_gguf,
daniloreddy_qwen3_5_0_8b_gguf, and 23 others) install a narrow-signature monkey-patch of
transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint with signature
(gguf_path, return_tensors=False) at import time. Transformers 5.2.0 now calls
load_gguf_checkpoint(path, return_tensors=True, model_to_load=dummy_model) inside
from_pretrained(). When mradermacher_blossom_v6_7b_i1_gguf's loader runs (it has no
own patch), it picks up whichever narrow-sig wrapper was last installed during
collection and that wrapper rejects the model_to_load kwarg with TypeError.

## Fix
1. Widened all 26 narrow-sig _patched_load_gguf_checkpoint functions from
   (gguf_path, return_tensors=False) to (*args, **kwargs), and updated the inner
   _orig_load_gguf_checkpoint call to match, so model_to_load passes through.
2. Added requirements.txt with gguf>=0.10.0 to the mradermacher_blossom_v6_7b_i1_gguf
   loader directory (conventional pattern for GGUF models).

All changes in tt-xla/third_party/tt_forge_models (tt-forge-models repo),
remediation branch: remediation/mradermacher_blossom_v6_7b_i1_gguf-causal_lm-pytorch-V6_7B_i1_GGUF-single_device-inference.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    382.26s (0:06:22)
- Tier A attempts: N/A

## Files changed
- tt-xla/third_party/tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_blossom_v6_7b_i1_gguf/causal_lm/pytorch/requirements.txt (new)
- tt-xla/third_party/tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py
- tt-xla/third_party/tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 49154132b3abac51dc293c8f5199722ba2f7e5be |
| tt-forge-models | 953640a0db8c497ac8f7de945973706c469b3339 |
