# Remediation Summary: mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf-causal_lm-pytorch-12B_Chimera_v1.1_heretic_uncensored_abliterated_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf/causal_lm/pytorch-12B_Chimera_v1.1_heretic_uncensored_abliterated_i1_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-requirements-txt-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

When gguf was available (from the global venv), the test then hit a second error from global state pollution by other loaders:

TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

This error comes from 26 GGUF loaders that monkey-patch `transformers.modeling_utils.load_gguf_checkpoint` at import time using a function with signature `(gguf_path, return_tensors=False)` — missing the `model_to_load` kwarg added in transformers 5.x. During pytest collection all loaders are imported, so these patches are applied before any test runs. The last loader alphabetically (`unified_reward_flex_qwen35_27b_gguf`) sets the active patch when the chimera test executes.

## Root cause
Two loader-layer bugs:

1. Missing `requirements.txt` in `mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf/causal_lm/pytorch/`. Without it the test framework does not install `gguf>=0.10.0`, so importing any GGUF utility in transformers raises ImportError.

2. 26 GGUF loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and install it at module level as a global replacement for `transformers.modeling_utils.load_gguf_checkpoint`. In transformers 5.x, `from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`. The narrow signature raises TypeError. The loader imported last alphabetically (`unified_reward_flex_qwen35_27b_gguf`) is the one whose bad patch survives into the test run.

## Fix
Three changes in `tt_forge_models` (branch `remediation/mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf-causal_lm-pytorch-12B_Chimera_v1.1_heretic_uncensored_abliterated_i1_GGUF-single_device-inference`):

1. **Added `requirements.txt`** (`mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt`) containing `gguf>=0.10.0` so the test framework installs it before the test runs.

2. **Guarded `apply_chat_template`** in the chimera loader's `load_inputs` with `if self.tokenizer.chat_template is not None:` to handle GGUF files that do not embed chat template metadata.

3. **Fixed `_patched_load_gguf_checkpoint` signature** in 26 loaders: changed `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, **kwargs)` and propagated `**kwargs` to `_orig_load_gguf_checkpoint(...)` so that `model_to_load` passes through correctly.

## Verification
- pytest exit: PASS
- Hardware: blackhole-p150b
- Duration: 513.32s (0:08:33)
- Tier A attempts: N/A

## Files changed
- `mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `mn_captainerisnebula_12b_chimera_v1_1_heretic_uncensored_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
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
| tt-xla          | bde3e48d4e1ba329215f4f900dc3c763a3aedae4 |
| tt-forge-models | 4de3ee7e275850b942d996362d764e7ac832a490 |
