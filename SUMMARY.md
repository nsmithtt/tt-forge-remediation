# Remediation Summary: mox_small_1_i1_gguf-causal_lm-pytorch-mox_small_1_i1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mox_small_1_i1_gguf/causal_lm/pytorch-mox_small_1_i1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — mox-small-1 is a 32.23B OLMo2 model (~64 GB BF16) exceeding p150b 32 GB DRAM capacity

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-olmo2-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The original CI failure reported `/home/ttuser/hf-bringup/tt-xla/.local_venv/bin/python: No module named pytest` — a CI environment issue. The actual model failure on the configured branch is:

```
ValueError: GGUF model with architecture olmo2 is not supported yet.
```

Followed by session contamination:

```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

After both fixes, execution completes but crashes with OOM in `AllocatorImpl::allocate_buffer` during tilize — the model is too large for device DRAM.

## Root cause
Three stacked issues:

1. **Loader (olmo2 arch missing)**: `transformers 5.x` does not include `olmo2` in `GGUF_CONFIG_MAPPING` or `GGUF_TO_FAST_CONVERTERS`. The mox-small-1-i1 GGUF declares `general.architecture = olmo2`, so `load_gguf_checkpoint` raises `ValueError: GGUF model with architecture olmo2 is not supported yet`.

2. **Loader (narrow-sig session contamination)**: 26 GGUF loaders for qwen35/gpt-oss variants define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — a narrow signature that predates the `model_to_load` kwarg added in `transformers 5.2.0`. When any of these loaders is imported in the same pytest session (which happens during test collection), their narrow-sig wrapper replaces the global `load_gguf_checkpoint`, causing `TypeError` in any subsequent GGUF loading.

3. **Hardware capacity**: mox-small-1 is a 32.23B parameter OLMo2 model. At BF16 it occupies ~64 GB, which exceeds the capacity of every supported single-device configuration (n150: 12 GB, p150b: 32 GB). The Q4_K_M GGUF file itself is 19.5 GB. Even with quantized weights, inference requires significantly more DRAM than available.

## Fix
**Fix 1 — olmo2 GGUF arch registration** (`tt_forge_models/mox_small_1_i1_gguf/causal_lm/pytorch/loader.py`):

Added `_patch_olmo2_gguf_support()` at module import time that:
- Injects an `olmo2` entry into `GGUF_TO_TRANSFORMERS_MAPPING["config"]` with the correct GGUF→HF config key mappings (`block_count`→`num_hidden_layers`, `embedding_length`→`hidden_size`, `rope.freq_base`→`rope_theta`, etc.)
- Appends `"olmo2"` to `GGUF_SUPPORTED_ARCHITECTURES`
- Adds `olmo2 → GGUFGPTConverter` to `GGUF_TO_FAST_CONVERTERS` (OLMo2 uses a GPT2-BPE tokenizer)

**Fix 2 — narrow-sig contamination** (26 loader files across `tt_forge_models`):

Changed all 26 narrow-sig `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` definitions to `(*args, **kwargs)` and updated their internal `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` calls to `_orig_load_gguf_checkpoint(*args, **kwargs)`. This makes all wrappers forward `model_to_load` correctly.

**Fix 3 — hardware XFAIL** (`tt_xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):

Added `KNOWN_FAILURE_XFAIL` entry for this test with explanation that the 32.23B OLMo2 model exceeds single-device DRAM capacity.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
N/A

## Verification
- pytest exit: FAIL (OOM on device after ~16 min; hardware-class)
- Hardware:    blackhole-p150b
- Duration:    981.07s (loader fixes verified; OOM confirmed as hardware ceiling)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mox_small_1_i1_gguf/causal_lm/pytorch/loader.py` — olmo2 GGUF arch registration + GGUFGPTConverter
- 26 × `tt_forge_models/*/causal_lm/pytorch/loader.py` — narrow-sig fix for `_patched_load_gguf_checkpoint`
- `tt_xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — KNOWN_FAILURE_XFAIL

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 493c7675bf8c11adc284b672846ff515b42d9ab7 |
| tt-forge-models | 42009dbf1420fab4fa875aeca152c0bd5d18037e |
