# Remediation Summary: g_health_gguf/causal_lm/pytorch-14B_Base_GGUF-single_device-inference

## Skill version
2

## Test
`tests/runner/test_models.py::test_all_models_torch[g_health_gguf/causal_lm/pytorch-14B_Base_GGUF-single_device-inference]`

## Result
SILICON_PASS

## Failure
```
2026-04-24 05:56:46.165 | critical |          Always | TT_THROW: Fabric Router Sync: Timeout after 10000 ms. Device 2: Expected status 0xa2b2c2d2, got 0xa1b1c1d1 (assert.hpp:104)
```

## Root cause
Loader layer — `transformers` was updated to 5.2.0, which added a `model_to_load`
keyword argument to `load_gguf_checkpoint` in `modeling_gguf_pytorch_utils.py`.
Twenty-six other GGUF model loaders in `tt_forge_models` monkey-patch
`transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` (and related
module-level bindings) with a `_patched_load_gguf_checkpoint` function that only
accepted `(gguf_path, return_tensors=False)`.  When pytest collects all tests, these
import-time patches run first and replace the real function with the broken wrapper.
When the g_health_gguf loader then calls `AutoModelForCausalLM.from_pretrained(…,
gguf_file=…)`, transformers internally calls
`load_gguf_checkpoint(…, model_to_load=dummy_model)`, which raises
`TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.
The model never reached the device, so the original runtime timeout could not be
reproduced in this environment.

## Fix
Updated all 26 `_patched_load_gguf_checkpoint` function signatures in
`tt_forge_models` from `(gguf_path, return_tensors=False)` to
`(gguf_path, return_tensors=False, **kwargs)`, and forwarded `**kwargs` to
`_orig_load_gguf_checkpoint(…, **kwargs)` so the new `model_to_load` argument
passes through correctly.

This is not a forbidden workaround — it restores the correct semantics of the
monkey-patched function to match the updated transformers API; no model trimming,
CPU offload, shape change, or PCC threshold adjustment is involved.

Repository: `tt_forge_models`
Branch: `remediation/g_health_gguf-causal_lm-pytorch-14B_Base_GGUF-single_device-inference`

## Verification
pytest exit status: PASSED
Wall-clock duration: 644.82 s (0:10:44)
Hardware: p150b (Blackhole 150W, single device)

## Files changed
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

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3d189cfd7021b2145e30a695e6d8dccec61a8351 |
| tt-forge-models | 99907be8ba5d85f02e1fb60ecb6d7e29844e5da8 |
