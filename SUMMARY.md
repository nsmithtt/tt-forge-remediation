# Remediation Summary: gemma_3_12b_it_heretic_gguf-causal_lm-pytorch-12B_IT_HERETIC_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_3_12b_it_heretic_gguf/causal_lm/pytorch-12B_IT_HERETIC_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_51, 2, -1023, 9223372036854775807), kwargs = {})
  File "transformers/cache_utils.py", line 792, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause

Two bugs, both needed to clear the failure.

**Bug 1 (loader, pre-existing cross-test clobbering):** 26 GGUF loaders in
`tt_forge_models` monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
at module import time with a narrow signature:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
```
Transformers 5.2.0 added a `model_to_load` keyword argument to
`load_gguf_checkpoint`. Any test run in the same pytest session after one of
those 26 loaders has been imported will hit `TypeError: got an unexpected
keyword argument 'model_to_load'` when `from_pretrained` calls the patched
function. The `gemma_3_12b_it_heretic_gguf` loader itself does not have a
patch, but it is affected by the stale wrapper installed by other loaders.

**Bug 2 (tt-xla, Tier A):** Gemma3's `SlidingWindowCache.update` computes:
```python
self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```
With `sliding_window = 1024` and `full_value_states.shape[2] = 23` (the
context length is shorter than the window), the start index is `-1023`.
PyTorch eager silently clamps this to `0` and returns the full tensor.
The XLA lazy backend instead raises `RuntimeError: Value out of range
(expected to be in range of [-23, 22], but got -1023)`.

## Fix

**Fix 1 (tt_forge_models):** Changed all 26 narrow-signature wrappers from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    return _orig_load_gguf_checkpoint(*args, **kwargs)
```
Files: `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`,
`dmind_3_mini_i1_gguf`, `gpt_oss_swallow_120b_rl_v0_1_gguf`,
`gpt_oss_swallow_20b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`,
and 20 more `mradermacher_*`, `qwen_3_5_*`, `tvall43_*`, `unified_reward_*` loaders
(26 total, all `causal_lm/pytorch/loader.py`).

**Fix 2 (tt-xla):** Added pre-clamping in `TorchFunctionOverride.__torch_function__`
in `python_package/tt_torch/torch_overrides.py`. When the intercepted function is
`torch.ops.aten.slice.Tensor` and the start/end index is an integer smaller than
`-dim_size` for a statically-known dimension, the index is clamped to `-dim_size`
before forwarding to XLA, matching PyTorch eager semantics.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    749.60s (0:12:29)
- Tier A attempts: 1

## Files changed
- `tt_forge_models/bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_bartleby_qwen3_5_4b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_crow_4b_opus_4_6_distill_heretic_qwen3_5_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_huihui_qwen3_5_27b_claude_4_6_opus_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_luna_qwen3_5_27b_v5_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_homebrew_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_27b_tainted_heresy_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_ara_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_gabliterated_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_sompoa_heresy_v2_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unfiltered_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_4b_unredacted_max_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_qwen3_5_9b_abliterated_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/mradermacher_vilm_0_8b_sft_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/qwen_3_5_imatrix_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_2b_heretic_v3b_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/tvall43_qwen3_5_4b_heretic_v2_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/unified_reward_flex_qwen35_27b_gguf/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 659b05cb960a88e88c55935951d7ec167923a7ad |
| tt-forge-models | 69707086276c6188730199cb3bd6c5ddb13fd7fb |
