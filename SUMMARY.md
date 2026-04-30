# Remediation Summary: google_gemma3_gguf-causal_lm-pytorch-google_gemma_3_4B_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[google_gemma3_gguf/causal_lm/pytorch-google_gemma_3_4B_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

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

While executing: AutoModelForCausalLM.from_pretrained(...)
  transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
E   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After fixing the loader bug, a second error appeared:
```
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
  (args = (%cat_37, 2, -1023, 9223372036854775807))
Original traceback:
  transformers/cache_utils.py:214: in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

## Root cause

**Bug 1 — loader (gguf-load-checkpoint-model-to-load-kwarg):**
26 GGUF model loaders replace `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
at module import time with a 2-argument wrapper `_patched_load_gguf_checkpoint(gguf_path,
return_tensors=False)`. In transformers 5.x the function gained a third parameter
`model_to_load=None`. When the test suite collects all models (via `os.walk`), these
loaders are imported and chain their patches globally. When `google_gemma3_gguf`'s
`from_pretrained` later calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`,
it hits the chained bad patches and raises TypeError. The `google_gemma3_gguf` loader
itself does not patch anything — it is a victim.

**Bug 2 — tt-xla frontend (aten-slice-tensor-out-of-bounds-start):**
Gemma 3 uses sliding-window attention with `sliding_window=1024`. The KV cache update
slices with `full_value_states[:, :, -sliding_window+1:, :]` = `[:, :, -1023:, :]`.
With seq_len=23, the start index -1023 is outside the valid range [-23, 22]. PyTorch
CPU silently clamps such out-of-range starts; XLA validates strictly and raises
`RuntimeError: Value out of range`. The clamping fix in `TorchFunctionOverride` was
not yet present on this branch.

## Fix

**Fix 1 — tt_forge_models (26 loaders):**
Added `model_to_load=None` to `_patched_load_gguf_checkpoint` and passed it through
to the original function for all 26 affected loaders:
- `bartowski_coniccat_qwen3_5_27b_writer_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`,
  `dmind_3_mini_i1_gguf` (qwen35 patch pattern)
- `gpt_oss_swallow_120b_rl_v0_1_gguf`, `gpt_oss_swallow_20b_rl_v0_1_gguf`,
  `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf`, `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf`
  (gpt-oss patch pattern)
- 19 more qwen35 loaders: mradermacher_bartleby/crow/gpt_oss/huihui/luna/qwen3_5_27b*/
  qwen3_5_4b*/qwen3_5_9b*/vilm, qwen_3_5_imatrix, tvall43_qwen3_5_2b*/4b*,
  unified_reward_flex_qwen35_27b

Branch: `remediation/google_gemma3_gguf-causal_lm-pytorch-google_gemma_3_4B_IT_GGUF-single_device-inference`
in `tenstorrent/tt-forge-models`

**Fix 2 — tt-xla `python_package/tt_torch/torch_overrides.py`:**
Added a clamp in `TorchFunctionOverride.__torch_function__` for
`func is torch.ops.aten.slice.Tensor`: when the start index is less than `-dim_size`,
clamp it to `-dim_size` before dispatch. This fires before the FX graph is built,
matching PyTorch CPU's silent-clamp semantics.

Branch: `remediation/google_gemma3_gguf-causal_lm-pytorch-google_gemma_3_4B_IT_GGUF-single_device-inference`
in `tenstorrent/tt-xla`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    410.87s (0:06:50)
- Tier A attempts: 1

## Files changed
- `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`
- `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`
- `dmind_3_mini_i1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- `gpt_oss_swallow_20b_sft_v0_1_mxfp4_moe_gguf/causal_lm/pytorch/loader.py`
- `mradermacher_gpt_oss_swallow_120b_rl_v0_1_gguf/causal_lm/pytorch/loader.py`
- 19 more qwen35 loaders in tt-forge-models (mradermacher_*, qwen_3_5_imatrix_gguf, tvall43_*, unified_*)
- `python_package/tt_torch/torch_overrides.py` (in tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550e3a3ec0b34c7a01ae9c2e64b23e9a |
| tt-mlir         | 553c0632b5eab5ac671df93c0c7d059cfb2deba9 |
| tt-xla          | 5ef71b8e7 (remediation branch) |
| tt-forge-models | b3ea5ef3ec (remediation branch) |
