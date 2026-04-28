# Remediation Summary: bartowski_gryphe_codex_24b_small_3_2_gguf-causal_lm-pytorch-Gryphe_Codex_24B_Small_3_2_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_gryphe_codex_24b_small_3_2_gguf/causal_lm/pytorch-Gryphe_Codex_24B_Small_3_2_GGUF-single_device-inference]

## Result
XFAIL — 24B model exhausts single-device DRAM capacity on p150b

## Stack layer
loader, hardware-class

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
Original failure (pre-fix): raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

After gguf was added to global requirements (cd8104788 in tt-xla), the test advanced to:
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

After kwargs fix, the test advanced to:
ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!

After chat template guard fix, the test ran to completion and failed with:
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks, where each bank needs to store 41943040 B, but bank size is 4273390016 B (allocated: 4196976832 B, free: 76413184 B, largest free block: 37030336 B)

## Root cause
Two loader bugs cascaded before the hardware ceiling was reached:

1. **kwargs compat**: 26 GGUF loaders in tt-forge-models patch `load_gguf_checkpoint` at module import time with `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)`. Transformers 5.x added `model_to_load=None` to this call, causing TypeError when running in a full pytest session where all loaders are imported during collection.

2. **Chat template guard**: The loader's `load_inputs()` unconditionally called `tokenizer.apply_chat_template()`. The Gryphe Codex-24B GGUF tokenizer has no chat template, causing ValueError in transformers 5.x which tightened the error vs returning silently.

3. **Hardware capacity**: After both loader bugs are fixed, the 24B bfloat16 model exhausts the p150b single-device DRAM. The device has ~34.2 GB total DRAM (8 banks × 4.27 GB) and the model consumes ~33.6 GB, leaving insufficient space for activation tensors.

## Fix
**Loader fix (tt-forge-models)**:
- `bartowski_gryphe_codex_24b_small_3_2_gguf/causal_lm/pytorch/loader.py`: Guard `apply_chat_template` with `if self.tokenizer.chat_template is not None`
- 26 other GGUF loader files: Added `**kwargs` to `_patched_load_gguf_checkpoint` signature and forwarded to `_orig_load_gguf_checkpoint` call

**Test config (tt-xla)**:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this test with the OOM reason

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    637.29s (0:10:37)
- Tier A attempts: N/A

## Files changed
**tt-forge-models (remediation/bartowski_gryphe_codex_24b_small_3_2_gguf-causal_lm-pytorch-Gryphe_Codex_24B_Small_3_2_GGUF-single_device-inference)**:
- bartowski_gryphe_codex_24b_small_3_2_gguf/causal_lm/pytorch/loader.py
- 26 other GGUF loader files with `_patched_load_gguf_checkpoint` pattern

**tt-xla (remediation/bartowski_gryphe_codex_24b_small_3_2_gguf-causal_lm-pytorch-Gryphe_Codex_24B_Small_3_2_GGUF-single_device-inference)**:
- tests/runner/test_config/torch/test_config_inference_single_device.yaml

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 30bf49d265ba4324e995f376353f136104739ab6 |
| tt-forge-models | 769604415f5854a8aaa1478174e457dab9794a45 |
