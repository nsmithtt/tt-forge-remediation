# Remediation Summary: alice_32b_i1_gguf-causal_lm-pytorch-32B_i1_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[alice_32b_i1_gguf/causal_lm/pytorch-32B_i1_GGUF-single_device-inference]

## Result
XFAIL — Alice-32B (Qwen3-32B architecture) BF16 weights consume ~33.6 GB of available single-device DRAM (~34 GB), leaving no room for inference activations

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
dram-capacity-32b-model-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: TT_FATAL @ tt_metal/impl/allocator/bank_manager.cpp:439: false
info: Out of Memory: Not enough space to allocate 262144000 B DRAM buffer across 8 banks, where each bank needs to store 32768000 B, but bank size is 4273390016 B (allocated: 4202914112 B, free: 70475904 B, largest free block: 22282240 B)

The failure was reached after fixing two loader bugs:
1. `ValueError: GGUF model with architecture qwen3vl is not supported yet.` — the Alice-32B GGUF header declares `general.architecture = "qwen3vl"` but the loader was a plain causal-LM loader with no GGUF architecture registration patch.
2. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` — multiple other GGUF loaders had installed fixed-signature wrappers `(gguf_path, return_tensors=False)` that dropped the `model_to_load` kwarg added in transformers 5.2.0.

## Root cause
Two distinct issues:

**Loader bugs (fixed):**
1. `qwen3vl` architecture not registered: Alice-32B is a fine-tune of Qwen3-32B and its GGUF file uses `general.architecture = "qwen3vl"`. transformers 5.x does not list `qwen3vl` in `GGUF_SUPPORTED_ARCHITECTURES`, so `load_gguf_checkpoint` raises `ValueError`. Fix: register `qwen3vl` as an alias for `qwen3` in `GGUF_TO_TRANSFORMERS_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`, and remap `model_type` from `"qwen3vl"` to `"qwen3"` post-load so `AutoModelForCausalLM` resolves to `Qwen3ForCausalLM`.
2. `model_to_load` kwarg: 26 other GGUF loaders installed monkey-patches with fixed signature `(gguf_path, return_tensors=False)`. Each patch captured the previous as `_orig_load_gguf_checkpoint`, forming a chain. When the weight-loading path in transformers 5.2.0 calls with `model_to_load=dummy_model`, the chain raises `TypeError`. Fixed by cherry-picking commits `7643c93cc7` and `eaee402cd2` from tt-forge-models (which updated all 26 affected loaders to `(*args, **kwargs)`).

**Hardware capacity ceiling:**
After the loader fixes the test reaches actual compilation and execution. At execution time, the full Qwen3-32B architecture in BF16 fills the single-device DRAM: 8 banks × 4.27 GB = ~34 GB available, ~33.6 GB consumed by model weights (Qwen3-32B ~16B unique parameters × 2 bytes ≈ 32–34 GB), leaving only ~400 MB free (largest contiguous block ~22 MB per bank). The tilize operation for the first weight tensor (gate_proj: 131M elements × 2B = 262 MB) cannot be allocated. This is a genuine hardware capacity ceiling: the model weights leave no room for inference activation buffers on a single p150b device.

## Fix
**tt-forge-models** (`remediation/alice_32b_i1_gguf-causal_lm-pytorch-32B_i1_GGUF-single_device-inference`):
- `alice_32b_i1_gguf/causal_lm/pytorch/loader.py`: Added `_patch_qwen3vl_causal_lm_support()` which registers `qwen3vl` as an alias for `qwen3` in `GGUF_TO_TRANSFORMERS_MAPPING` and `GGUF_SUPPORTED_ARCHITECTURES`, and a `_patched_load_gguf_checkpoint(*args, **kwargs)` wrapper that fixes `model_type` from `"qwen3vl"` to `"qwen3"` post-load.
- Cherry-picked commits `7643c93cc7` and `eaee402cd2` to fix 26 other GGUF loaders that used narrow-signature wrappers.

**tt-xla** (`remediation/alice_32b_i1_gguf-causal_lm-pytorch-32B_i1_GGUF-single_device-inference`):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `alice_32b_i1_gguf/causal_lm/pytorch-32B_i1_GGUF-single_device-inference` as `KNOWN_FAILURE_XFAIL` documenting the DRAM capacity ceiling.

## Verification
- pytest exit: FAIL (OOM, hardware capacity — expected for XFAIL)
- Hardware:    p150b
- Duration:    864.94s (0:14:24) to reach OOM
- Tier A attempts: N/A

## Files changed
- `alice_32b_i1_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)
- 26 other GGUF loader files updated via cherry-pick (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | 9d881f7fa |
| tt-forge-models | c487f28391 |
