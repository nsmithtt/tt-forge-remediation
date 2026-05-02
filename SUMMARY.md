# Remediation Summary: mozilla_ai_gemma_2_27b_it_llamafile-causal_lm-pytorch-gemma-2-27b-it-llamafile-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mozilla_ai_gemma_2_27b_it_llamafile/causal_lm/pytorch-gemma-2-27b-it-llamafile-single_device-inference]

## Result
XFAIL — 27B model dequantized to BF16 (~54 GB) exceeds p150b single-device DRAM (~34 GB)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-gemma2-27b-bf16-oom-p150b

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 339738624 B DRAM buffer across 8 banks, where each bank needs to store 42467328 B, but bank size is 4273390016 B (allocated: 4211366208 B, free: 62023808 B, largest free block: 31353152 B)
```

## Root cause

The `mozilla-ai/gemma-2-27b-it-llamafile` HuggingFace repo contains only llamafile executables (GGUF wrapped in a shell script), not standard HF model files. The original loader pointed at this repo and would fail with a tokenizer loading error once the repo no longer had stale cached HF files.

The loader was rewritten to use `bartowski/gemma-2-27b-it-GGUF` with `gemma-2-27b-it-Q4_K_M.gguf`. Transformers dequantizes the GGUF to BF16 before running on device, yielding a ~54 GB BF16 model (27B parameters × 2 bytes/param). The p150b has 8 GDDR6 banks × ~4.27 GB = ~34 GB of device DRAM. 54 GB > 34 GB: the model cannot fit on a single p150b.

Additionally, during the loader fix work, two cross-cutting GGUF loader bugs were identified and fixed:
1. 28 loaders patched `load_gguf_checkpoint` at import time with a signature missing `**kwargs`, causing `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` in transformers 5.x.
2. The same 26 loaders dropped `**kwargs` when forwarding to the original function, causing `AttributeError: 'NoneType' object has no attribute 'config'` in downstream patchers.

These were fixed but the OOM is the terminal failure, which is hardware-class.

## Fix

**tt_forge_models** — branch `remediation/mozilla_ai_gemma_2_27b_it_llamafile-causal_lm-pytorch-gemma-2-27b-it-llamafile-single_device-inference`:
- `mozilla_ai_gemma_2_27b_it_llamafile/causal_lm/pytorch/loader.py`: rewrote to use `bartowski/gemma-2-27b-it-GGUF` with `gemma-2-27b-it-Q4_K_M.gguf` via GGUF loading
- 26 loaders: added `**kwargs` to `_patched_load_gguf_checkpoint` signature
- 26 loaders: forwarded `**kwargs` through call to `_orig_load_gguf_checkpoint`

**tt-xla** — branch `remediation/mozilla_ai_gemma_2_27b_it_llamafile-causal_lm-pytorch-gemma-2-27b-it-llamafile-single_device-inference`:
- `python_package/tt_torch/torch_overrides.py`: added aten.slice OOB clamping for Gemma 2 sliding-window attention (start indices below `-dim_size` clamped to `-dim_size`)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: marked `KNOWN_FAILURE_XFAIL` with OOM reason

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling)
- Hardware: p150b
- Duration: 751.62s (0:12:31)
- Tier A attempts: N/A

## Files changed
- `tt-xla/python_package/tt_torch/torch_overrides.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)
- `tt_forge_models/mozilla_ai_gemma_2_27b_it_llamafile/causal_lm/pytorch/loader.py`
- 26 × `tt_forge_models/*/causal_lm/pytorch/loader.py` (kwargs fixes)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 276d588f8f9dd41bb58e4a842b091c8b022cf902 |
| tt-forge-models | ccf5f95fdd74282ea6205ef746631af9b98b5c16 |
