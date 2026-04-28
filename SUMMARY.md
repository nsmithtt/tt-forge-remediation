# Remediation Summary: 14b_qwen2_5_kunou_v1_gguf/causal_lm/pytorch-14B_Q4_K_M-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[14b_qwen2_5_kunou_v1_gguf/causal_lm/pytorch-14B_Q4_K_M-single_device-inference]

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
26 GGUF loader files in tt_forge_models each install a module-level monkey-patch on
`transformers.modeling_utils.load_gguf_checkpoint` (and related modules) at import time.
These patchers are collected during pytest session startup (all model loaders are imported
to enumerate test variants). Their `_patched_load_gguf_checkpoint` wrappers were defined
with the signature `(gguf_path, return_tensors=False)`, omitting the `model_to_load`
parameter added in transformers 5.x. When the `14b_qwen2_5_kunou_v1_gguf` loader later
calls `AutoModelForCausalLM.from_pretrained(...)`, transformers 5.2.0 internally calls
`load_gguf_checkpoint(..., model_to_load=dummy_model)`, which hits the now-global patched
version and raises TypeError.

## Fix
Updated all 26 affected loader files in tt_forge_models to add `model_to_load=None` to
the `_patched_load_gguf_checkpoint` signature and pass it through to the original
`_orig_load_gguf_checkpoint` call.

Files changed (all in `third_party/tt_forge_models/`):
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

Commit: `b255704d3bb75b464b06fa4d02322067616d9e3d` on
`tenstorrent/tt-forge-models` branch
`remediation/14b-qwen2-5-kunou-v1-gguf-q4-k-m-single-device-inference`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    547.29s (0:09:07)
- Tier A attempts: N/A

## Files changed
- 26 files in `tt-xla/third_party/tt_forge_models/*/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 80f105e2b46bfb5cb40f6c7f3d142f428ee3eaab |
| tt-forge-models | b255704d3bb75b464b06fa4d02322067616d9e3d |
