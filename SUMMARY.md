# Remediation Summary: glotmax_101_14b_gguf-causal_lm-pytorch-101-14B-GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[glotmax_101_14b_gguf/causal_lm/pytorch-101-14B-GGUF-single_device-inference]

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

(Original report listed "Fabric Router Sync: Timeout" but that was a different run; the actual error on this branch is the TypeError above.)

## Root cause
Three GGUF loader modules alphabetically before `glotmax_101_14b_gguf`
(`bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`,
`dmind_3_mini_i1_gguf`) monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import
time with a narrow signature `(gguf_path, return_tensors=False)`.
transformers 5.2.0 now calls `load_gguf_checkpoint` with
`model_to_load=dummy_model`, which raises `TypeError` via the stale
narrow-signature patch when the glotmax model loader later calls
`AutoModelForCausalLM.from_pretrained`. This is a loader-layer bug in
those three Qwen3.5 GGUF loaders (and 24 others not affecting this test).

## Fix
Cherry-picked commit `45866b7a1b` from
`tt_forge_models/remediation/bartowski-huihui-gpt-oss-20b-bf16-abliterated-gguf`
onto a new branch
`remediation/glotmax_101_14b_gguf-causal_lm-pytorch-101-14B-GGUF-single_device-inference`
in `tt-forge-models`.

The fix changes all 27 affected `_patched_load_gguf_checkpoint` definitions
from the narrow `(gguf_path, return_tensors=False)` signature to
`(gguf_path, return_tensors=False, model_to_load=None)` and passes
`model_to_load` through to the original function.

Files changed (27 loader.py files in tt-forge-models):
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_gguf/causal_lm/pytorch/loader.py`
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
- `mradermacher_qwen_3_5_27b_derestricted_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    564.48s (0:09:24)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py` (and 26 more — see Fix section)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | d5f3666e96d2b326de147684839ee16697e5eb2e |
| tt-forge-models | 5f7eef293d9e9289b5314907f0273810cecae044 |
