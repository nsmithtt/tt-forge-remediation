# Remediation Summary: gpt_oss_20b_heretic_ara_gguf-causal_lm-pytorch-20B_Heretic_Ara_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_20b_heretic_ara_gguf/causal_lm/pytorch-20B_Heretic_Ara_GGUF-single_device-inference]

## Result
XFAIL — GPT-OSS 20B BF16 weights (~33 GB) leave <1 GB free for inference activations on p150b (34 GB DRAM); hardware capacity ceiling

## Stack layer
loader, tt-xla, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, aten-slice-tensor-out-of-bounds-start, hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original: `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`

After loader fix: `RuntimeError: Value out of range (expected to be in range of [-100, 99], but got -127) While executing %slice_6 : call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_3, 2, -127, 9223372036854775807))`

After slice fix: `RuntimeError: TT_FATAL @ bank_manager.cpp:439: Out of Memory: Not enough space to allocate 1061683200 B DRAM buffer across 8 banks (allocated: 4111190976 B, free: 162199040 B)`

## Root cause
Three bugs fixed in sequence:

**1. Loader layer — GGUF global patch missing `model_to_load`**: 26 loaders install `_patched_load_gguf_checkpoint` at import time with signature `(gguf_path, return_tensors=False)`. Transformers 5.x added a third positional kwarg `model_to_load=dummy_model` to `load_gguf_checkpoint`. During pytest collection all model modules are imported; the last patcher installed (alphabetically `unified_reward_flex_qwen35`) overwrites the module attribute without `model_to_load` support. When `from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, it hits the broken patch → TypeError.

The `gpt_oss_20b_heretic_ara_gguf` loader also needed its own gpt-oss GGUF registration (GGUF arch `gpt-oss` → HF `model_type: gpt_oss`) to survive in a pytest session where other patchers may map "gpt-oss" to "qwen3_moe" incorrectly.

**2. Compiler frontend (tt-xla) — aten.slice.Tensor OOB start**: GPT-OSS uses `SlidingWindowCache` with `sliding_window=128`. The cache update does `full_value_states[:, :, -self.sliding_window + 1:, :]` = start=-127. With seq_len=100, XLA's strict bounds check raises `ValueError: Value out of range [-100, 99], got -127`. PyTorch CPU silently clamps; XLA does not. Fix: clamp the start index to `max(-dim_size, start)` in `TorchFunctionOverride.__torch_function__`.

**3. Hardware capacity**: After the above fixes the model compiled and began executing on TT silicon. GPT-OSS 20B dequantizes to BF16 weighing ~33 GB, consuming 97% of p150b's 34 GB DRAM. An intermediate activation tensor of 1.06 GB cannot be allocated → OOM. The base `gpt_oss/pytorch-20B` is already EXCLUDE_MODEL for this reason; the GGUF variant has the same constraint.

## Fix
**Loader fixes (tt-forge-models branch):**
- `eb2c17a51c`: Added `**kwargs` to `_patched_load_gguf_checkpoint` in 26 GGUF loaders so `model_to_load` is forwarded to the original function.
- `92c6c66905` through `a87c820cc3`: Six commits registering the `gpt-oss` GGUF architecture in the heretic_ara loader, fixing `GptOssTensorProcessor` expert weight packing, patching `get_gguf_hf_weights_map` for model_type underscore/hyphen mismatch, patching `load_gguf_checkpoint` with file-header arch detection to override swallow-loader interference, and registering `GGUF_TO_FAST_CONVERTERS["gpt_oss"]` for tokenizer lookup.

**tt-xla compiler fix:**
- `6d45f45c5`: Clamped out-of-range `aten.slice.Tensor` start index in `TorchFunctionOverride.__torch_function__` in `tt-xla/python_package/tt_torch/torch_overrides.py`.

**Test config XFAIL:**
- `17d489e9d`: Added `KNOWN_FAILURE_XFAIL` entry for this test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Tier B justification (FAIL with Tier=B only — omit otherwise)

## Verification
- pytest exit: FAIL (OOM — hardware capacity)
- Hardware:    blackhole-p150b
- Duration:    1083.65s (0:18:03) for the final OOM run; 1119.29s for the preceding slice-OOM run
- Tier A attempts: 1 (slice clamp fix, confirmed past the slice OOB error; OOM is hardware-class)

## Files changed
**tt-forge-models:**
- `gpt_oss_20b_heretic_ara_gguf/causal_lm/pytorch/loader.py` — comprehensive gpt-oss GGUF registration + patching
- 26 other `*/causal_lm/pytorch/loader.py` files — added `**kwargs` to `_patched_load_gguf_checkpoint`

**tt-xla:**
- `python_package/tt_torch/torch_overrides.py` — slice start clamp in `TorchFunctionOverride`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 17d489e9d6638115c2d8ad948518bff29d634855 |
| tt-forge-models | a87c820cc31e35aaa1e47e03e5f94ebcf8295050 |
