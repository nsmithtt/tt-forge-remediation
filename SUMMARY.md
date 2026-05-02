# Remediation Summary: nizami_1_7b_i1_gguf-causal_lm-pytorch-1_7B_I1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[nizami_1_7b_i1_gguf/causal_lm/pytorch-1_7B_I1_GGUF-single_device-inference]

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
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

(In CI, where `gguf` is not pre-installed, the failure surfaced one step earlier as:
`ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.`)

## Root cause
Two cascading loader bugs prevented the test from running:

1. **Missing `requirements.txt`**: The nizami loader had no `requirements.txt` listing
   `gguf>=0.10.0`.  In a CI environment where `gguf` is absent,
   `is_gguf_available()` returns False and transformers raises the ImportError
   immediately.

2. **`model_to_load` kwarg missing from 26 GGUF patchers**: 26 loaders patch
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module
   import time using the old fixed signature
   `(gguf_path, return_tensors=False)`.  Transformers 5.x added a
   `model_to_load` keyword that is passed at every call site in
   `modeling_utils.py`.  Because `setup_test_discovery` imports ALL loaders
   during pytest collection (even for a single-test run), these patchers are
   already in place when the nizami test runs.  The patched function does not
   accept `model_to_load`, so transformers raises TypeError.

## Fix
Both fixes are in `tenstorrent/tt-forge-models` on branch
`remediation/nizami_1_7b_i1_gguf-causal_lm-pytorch-1_7B_I1_GGUF-single_device-inference`
(commit `1ddb02881b`):

1. **`nizami_1_7b_i1_gguf/causal_lm/pytorch/requirements.txt`** (new file):
   ```
   gguf>=0.10.0
   ```

2. **26 loader files** (modified): all `_patched_load_gguf_checkpoint` functions
   with signature `(gguf_path, return_tensors=False)` changed to
   `(*args, **kwargs)` and the inner call updated to
   `_orig_load_gguf_checkpoint(*args, **kwargs)`, so `model_to_load` (and any
   future kwargs) pass through correctly.

The `tt-xla` remediation branch
(`remediation/nizami_1_7b_i1_gguf-causal_lm-pytorch-1_7B_I1_GGUF-single_device-inference`,
commit `c09011b33b`) advances the `third_party/tt_forge_models` pointer to
`1ddb02881b`.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    293.03s (0:04:53)
- Tier A attempts: N/A

## Files changed
- `nizami_1_7b_i1_gguf/causal_lm/pytorch/requirements.txt` (new file)
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
| Submodule       | Commit                                     |
|-----------------|--------------------------------------------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc   |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee   |
| tt-xla          | c09011b33b0e876ee9a03f93589c5eb749e386ab   |
| tt-forge-models | 1ddb02881b2dfd2b9fd5f767e68e101726d36d6e   |
