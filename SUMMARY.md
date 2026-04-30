# Remediation Summary: gemma_sea_guard_12b_2602_gguf-causal_lm-pytorch-SEA_Guard_12B_2602_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_sea_guard_12b_2602_gguf/causal_lm/pytorch-SEA_Guard_12B_2602_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-missing-requirements-txt, gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Three cascading loader and compiler-frontend bugs:

1. **Missing requirements.txt** (loader): `gemma_sea_guard_12b_2602_gguf/causal_lm/pytorch/` had no `requirements.txt`, so the test framework never installed `gguf>=0.10.0`. When transformers calls `load_gguf_checkpoint`, it first checks for the `gguf` package and raises `ImportError` if absent.

2. **`_patched_load_gguf_checkpoint` missing `model_to_load` kwarg** (loader): 28 qwen3.5 GGUF loaders (beginning with `bartowski_coniccat_qwen3_5_27b_writer_gguf`, alphabetically before `gemma_sea_guard`) globally patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time. Their patched function only accepted `(gguf_path, return_tensors=False)`, but transformers 5.2.0 added a `model_to_load` keyword argument. When the gemma_sea_guard test ran, the qwen3.5 loaders had already been imported during test discovery and the corrupted global patch was in effect, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`.

3. **XLA lazy slice out-of-bounds negative start** (tt-xla): Gemma3-based models use a sliding window cache with `sliding_window=1024`. During inference with short sequences (23 tokens), `transformers/cache_utils.py` computes `full_value_states[:, :, -self.sliding_window + 1 :, :]` = `[:, :, -1023:, :]`. The tensor only has 23 elements in dim 2 so `-1023 < -23` is out of range. The XLA lazy backend raises `RuntimeError: Value out of range` rather than clamping as standard Python/PyTorch would.

## Fix
**Fix 1 — loader (tt-forge-models)**:
Added `gemma_sea_guard_12b_2602_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0`.

**Fix 2 — loader (tt-forge-models)**:
Updated `_patched_load_gguf_checkpoint` in 28 qwen3.5 GGUF loaders to accept `**kwargs` and pass them through to `_orig_load_gguf_checkpoint`. Changed signature from `(gguf_path, return_tensors=False)` to `(gguf_path, return_tensors=False, **kwargs)` and call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors, **kwargs)`.

**Fix 3 — tt-xla (Tier A)**:
Added pre-clamping for `aten.slice.Tensor` in `TorchFunctionOverride.__torch_function__` in `python_package/tt_torch/torch_overrides.py`. When the slice start or end is a negative integer less than `-size` for the given dimension, it is clamped to `-size` before dispatching, matching Python/PyTorch eager semantics.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    687.59s (0:11:27)
- Tier A attempts: 1

## Files changed
**tt-forge-models** (`remediation/gemma_sea_guard_12b_2602_gguf-causal_lm-pytorch-SEA_Guard_12B_2602_GGUF-single_device-inference`):
- `gemma_sea_guard_12b_2602_gguf/causal_lm/pytorch/requirements.txt` (new)
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

**tt-xla** (`remediation/gemma_sea_guard_12b_2602_gguf-causal_lm-pytorch-SEA_Guard_12B_2602_GGUF-single_device-inference`):
- `python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | eba47edbd86984bbe960148444659a92de03c598 |
| tt-forge-models | b93a57b883ac74e7467ed2e241799a8fc34021e1 |
