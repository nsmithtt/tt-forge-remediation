# Remediation Summary: llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf-causal_lm-pytorch-Llama_3_2_3B_Instruct_heretic_ablitered_uncensored_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf/causal_lm/pytorch-Llama_3_2_3B_Instruct_heretic_ablitered_uncensored_i1_GGUF-single_device-inference]

## Result
SILICON_PASS — loader fix for model_to_load kwarg missing in 26 module-level GGUF checkpoint patchers

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
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Full traceback:
```
third_party/tt_forge_models/llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf/causal_lm/pytorch/loader.py:105: in load_model
    model = AutoModelForCausalLM.from_pretrained(
venv/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py:374: in from_pretrained
    return model_class.from_pretrained(
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
26 loaders in tt_forge_models patch `load_gguf_checkpoint` at **module level** (not inside a function). When pytest collects all tests, `TorchDynamicLoader.setup_test_discovery()` imports every loader.py file. Any of these 26 loaders that gets imported installs its `_patched_load_gguf_checkpoint` globally in `transformers.modeling_gguf_pytorch_utils` and related modules.

The patched functions all had the signature `(gguf_path, return_tensors=False)`, which is missing the `model_to_load` parameter added in transformers 5.x. The real `load_gguf_checkpoint` signature in transformers 5.2.0 is `(gguf_checkpoint_path, return_tensors=False, model_to_load=None)`.

When `llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf` loader subsequently calls `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`, transformers internally calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` — which hits the already-installed patched version and raises TypeError.

The Llama model's own loader does not patch `load_gguf_checkpoint`; it is purely a victim of pollution from the 26 other loaders imported during collection.

## Fix
Added `model_to_load=None` to the function signature of `_patched_load_gguf_checkpoint` in all 26 affected loader files, and passed `model_to_load=model_to_load` through to the original `_orig_load_gguf_checkpoint` call.

**Repository:** tt_forge_models  
**Branch:** `remediation/llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf-causal_lm-pytorch-Llama_3_2_3B_Instruct_heretic_ablitered_uncensored_i1_GGUF-single_device-inference`  
**Commit:** `63dc4a7e961d3d47ebe4dfce64ea7d3c5d656249`

Files changed (all in `<loader>/causal_lm/pytorch/loader.py`):
- bartowski_coniccat_qwen3_5_27b_writer_gguf
- daniloreddy_qwen3_5_0_8b_gguf
- dmind_3_mini_i1_gguf
- gpt_oss_swallow_120b_rl_v0_1_gguf
- gpt_oss_swallow_20b_rl_v0_1_gguf
- gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf
- mradermacher_bartleby_qwen3_5_4b_gguf
- mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf
- mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf
- mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf
- mradermacher_luna_qwen3_5_27b_v5_i1_gguf
- mradermacher_qwen3_5_27b_gguf
- mradermacher_qwen3_5_27b_homebrew_gguf
- mradermacher_qwen3_5_27b_tainted_heresy_gguf
- mradermacher_qwen3_5_4b_abliterated_i1_gguf
- mradermacher_qwen3_5_4b_ara_heresy_v2_gguf
- mradermacher_qwen3_5_4b_gabliterated_gguf
- mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf
- mradermacher_qwen3_5_4b_unfiltered_gguf
- mradermacher_qwen3_5_4b_unredacted_max_gguf
- mradermacher_qwen3_5_9b_abliterated_i1_gguf
- mradermacher_vilm_0_8b_sft_gguf
- qwen_3_5_imatrix_gguf
- tvall43_qwen3_5_2b_heretic_v3b_i1_gguf
- tvall43_qwen3_5_4b_heretic_v2_i1_gguf
- unified_reward_flex_qwen35_27b_gguf

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    439.13s (0:07:19)
- Tier A attempts: N/A

## Files changed
- 26 × `<loader>/causal_lm/pytorch/loader.py` in tt_forge_models (see list above)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 63dc4a7e961d3d47ebe4dfce64ea7d3c5d656249 |
