# Remediation Summary: gemma3_27b_qat_gguf-causal_lm-pytorch-stduhpf_27B_IT_QAT_Q4_0_GGUF_SMALL-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_27b_qat_gguf/causal_lm/pytorch-stduhpf_27B_IT_QAT_Q4_0_GGUF_SMALL-single_device-inference]

## Result
XFAIL — Gemma 3 27B QAT Q4_0 GGUF dequantizes to ~54 GB BF16, exceeding single-device DRAM on all supported hardware (p150b 32 GB, n150 12 GB)

## Stack layer
loader, tt-xla, hardware-class

## Tier
A

## Bug fingerprint
hardware-class-dram-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure:
```
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
```

Actual first failure on reproduction:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Second failure after loader fix:
```
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -1023)
While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_65, 2, -1023, 9223372036854775807), kwargs = {})
Original traceback: cache_utils.py:214: self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]
```

Terminal failure after both fixes:
```
RuntimeError: TT_FATAL @ bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 231211008 B DRAM buffer across 8 banks,
where each bank needs to store 28901376 B, but bank size is 4273390016 B
(allocated: 4225128768 B, free: 48261248 B, largest free block: 13855040 B)
```

## Root cause
Three stacked issues were found:

**1. Loader bug (gguf-load-checkpoint-model-to-load-kwarg):** transformers 5.2.0 added a `model_to_load` keyword argument to `load_gguf_checkpoint`. 26 GGUF loaders in tt_forge_models monkey-patch this function at module import time with a narrow signature `(gguf_path, return_tensors=False)` that rejects the new kwarg. Since pytest collects all loaders before running the target test, any collected GGUF loader poisons the global function for the gemma3_27b_qat_gguf test.

**2. Compiler frontend bug (aten-slice-tensor-out-of-bounds-start, Tier A):** Gemma 3 uses sliding-window attention with `sliding_window=1024`. The cache update code does `full_value_states[:, :, -self.sliding_window + 1:, :]` = `[:, :, -1023:, :]`. For a short test input (max_length=128 → 24 cached tokens), the XLA/TT backend raises "Value out of range" because -1023 < -24, whereas PyTorch eager would silently clamp. Fixed by adding a `clamp_out_of_range_slice_starts` FX pass in `tt-xla/python_package/tt_torch/backend/passes.py`.

**3. Hardware capacity (terminal):** After both fixes, the model allocates 33.8 GB of DRAM on p150b (8 banks × 4.27 GB = 34.2 GB total) before failing to allocate 231 MB for an activation. A 27B parameter model dequantized to BF16 requires ~54 GB, which exceeds any supported single device.

## Fix
**Loader fix — tt_forge_models** (`remediation/gemma3_27b_qat_gguf-causal_lm-pytorch-stduhpf_27B_IT_QAT_Q4_0_GGUF_SMALL-single_device-inference`, commit `4ac9e0b582`):
- Changed `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` → `def _patched_load_gguf_checkpoint(*args, **kwargs):` in 26 loaders
- Changed corresponding `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` → `_orig_load_gguf_checkpoint(*args, **kwargs)`
- Files: `bartowski_coniccat_qwen3_5_27b_writer_gguf/causal_lm/pytorch/loader.py`, `daniloreddy_qwen3_5_0_8b_gguf/causal_lm/pytorch/loader.py`, and 24 other GGUF loaders

**Compiler fix — tt-xla** (`remediation/gemma3_27b_qat_gguf-causal_lm-pytorch-stduhpf_27B_IT_QAT_Q4_0_GGUF_SMALL-single_device-inference`, commits `e63cec579`, `5463e3a96`):
- Added `clamp_out_of_range_slice_starts(gm)` FX pass to `python_package/tt_torch/backend/passes.py`: iterates `aten.slice.Tensor` nodes, reads `input_node.meta["val"].shape[dim]`, clamps `start` to `-dim_size` if `start < -dim_size`
- Added import and call to the pass in `python_package/tt_torch/backend/backend.py` after `bypass_assert_tensor_metadata`
- Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Verification
- pytest exit: FAIL (DRAM OOM after both fixes)
- Hardware:    blackhole-p150b
- Duration:    809.81s (0:13:29) for second run
- Tier A attempts: 1

## Files changed
**tt_forge_models** (26 files):
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

**tt-xla**:
- `python_package/tt_torch/backend/passes.py`
- `python_package/tt_torch/backend/backend.py`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5463e3a96fe8754818c7d1306f0c8eb45e8196be |
| tt-forge-models | 4ac9e0b5821495f21a763c162656ce44eba69335 |
