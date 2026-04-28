# Remediation Summary: qwen_3_32b_f16_gguf-causal_lm-pytorch-32B_F16_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_32b_f16_gguf/causal_lm/pytorch-32B_F16_GGUF-single_device-inference]

## Result
XFAIL — Qwen3-32B bfloat16 model (~64 GB) exceeds single-device DRAM (~32 GB on Blackhole n150)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-32b-bf16-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: TT_FATAL @ /home/nsmith/tt-forge-remediation/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:439: false
info:
Out of Memory: Not enough space to allocate 262144000 B DRAM buffer across 8 banks, where each bank needs to store 32768000 B, but bank size is 4273390016 B (allocated: 4202914112 B, free: 70475904 B, largest free block: 22282240 B)
```

The original CI failure `TT_THROW @ silicon_sysmem_manager.cpp:326` was the same root cause: the model exceeds device DRAM. Locally, the test also surfaced a loader-layer TypeError (see Root cause below), which was fixed before the OOM became visible.

## Root cause

Two issues were present:

**1. Loader-layer (fixed): TypeError in _patched_load_gguf_checkpoint**

Multiple GGUF loaders in `tt_forge_models` monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.x changed the call-site to pass `model_to_load=dummy_model` as a keyword argument (via a local import at `modeling_utils.py:4016`), causing a `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. This is a loader-layer bug.

Commit `8f228cff5e` on the `arch-c-36-tt-xla-dev/nsmith/hf-bringup-range-2000-3000-15` branch of `tt_forge_models` already contained the fix (changing the wrapper signature to `(*args, **kwargs)`). The tt-xla submodule pointer was advanced from `0f7b734348` to `8f228cff5e`.

**2. Hardware capacity ceiling (XFAIL): OOM on device**

The Qwen3-32B F16 GGUF model is loaded as bfloat16 (~64 GB of parameters). The Blackhole device has ~31.8 GB of DRAM (8 banks × ~3.98 GB). By the time the first inference kernel fires, 4202914112 B (~3.91 GB/bank, ~31.3 GB total) is already allocated to model weights, leaving only ~67 MB/bank free — far short of the 250 MB/bank needed for the next activation buffer. This is a genuine hardware capacity ceiling, not an allocator bug.

## Fix

- `tt_forge_models` (commit `8f228cff5e`): Changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to `(*args, **kwargs)` in loaders that monkey-patch `load_gguf_checkpoint`. Pushed as `remediation/qwen3-32b-f16-gguf-patched-load-model-to-load` on `tenstorrent/tt-forge-models`.
- `tt-xla` (commit `47d47ac5d`): Added `KNOWN_FAILURE_XFAIL` entry for `qwen_3_32b_f16_gguf/causal_lm/pytorch-32B_F16_GGUF-single_device-inference` to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. Advanced `third_party/tt_forge_models` submodule pointer from `0f7b734348` to `8f228cff5e`. Pushed as `remediation/qwen3-32b-f16-gguf-causal-lm-single-device-inference` on `tenstorrent/tt-xla`.

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling confirmed)
- Hardware:    n150
- Duration:    ~14 minutes (model loading + one inference attempt before OOM)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry
- `tt-xla/third_party/tt_forge_models` — submodule pointer advanced to `8f228cff5e`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 47d47ac5dac73523b5d31da38073196cd1b010a7 |
| tt-forge-models | 8f228cff5e1840aa02ec82974f6b0dc05892c033 |
