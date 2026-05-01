# Remediation Summary: mirxa2_mirxa_3_5_35b_a3b_uncensored_aggressive_gguf/causal_lm/pytorch-3_5_35B_A3B_Uncensored_Aggressive_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mirxa2_mirxa_3_5_35b_a3b_uncensored_aggressive_gguf/causal_lm/pytorch-3_5_35B_A3B_Uncensored_Aggressive_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — Model is 35B MoE (21B dequantized params, ~42.2 GB BF16) exceeds p150b 32 GB DRAM — hardware capacity ceiling

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
Underlying crash chain:
1. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` — cross-loader clobbering of `load_gguf_checkpoint` with narrow-sig patcher
2. `KeyError: 'blk.0.ffn_gate_exps'` — cross-loader clobbering of `get_gguf_hf_weights_map` stripped the ffn_gate_exps/ffn_up_exps alias logic
3. Fatal segfault after 3 StableHLO compilations — OOM from 42.2 GB model on 32 GB DRAM

## Root cause
**Loader bugs (fixed):**

Bug 1: transformers 5.2.0 added `model_to_load` parameter to `load_gguf_checkpoint`. The `tvall43_qwen3_5_4b_heretic_v2_i1_gguf` loader installs a narrow-sig wrapper `(gguf_path, return_tensors=False)` at module import time, overwriting the wide-sig wrapper installed by mirxa2. When transformers calls `load_gguf_checkpoint(..., model_to_load=dummy)`, the narrow-sig wrapper raises `TypeError`.

Bug 2: `qwen_3_5_claude_distilled_gguf` guards its `_patch_qwen35_support()` on `if "qwen35" in GGUF_SUPPORTED_ARCHITECTURES` (not `"qwen35moe"`), so it runs even after mirxa2 has patched, and reinstalls `get_gguf_hf_weights_map` with a version that lacks the `ffn_gate_exps`/`ffn_up_exps` alias logic. The mirxa2 GGUF stores separate `ffn_gate_exps` and `ffn_up_exps` tensors; without the alias entries in `tensor_key_mapping`, `GGUFMoEProcessor.process()` raises `KeyError`.

**Hardware ceiling (XFAIL):**

Qwen3.5-35B-A3B is a 35B MoE model with 21B active parameters. Dequantized from Q4_K_M (~20 GB GGUF) to BF16 weights requires ~42.2 GB DRAM. The p150b device provides 32 GB. After all 733 GGUF tensors load and 3 StableHLO graph compilations complete, the inference forward pass triggers a segfault in the C++ runtime — the OOM symptom for this hardware class.

## Fix
**Loader layer (tt_forge_models):**

All fixes in `tt-xla/third_party/tt_forge_models/mirxa2_mirxa_3_5_35b_a3b_uncensored_aggressive_gguf/causal_lm/pytorch/loader.py` on branch `remediation/mirxa2-mirxa-3-5-35b-a3b-uncensored-aggressive-gguf-single-device-inference` in `tenstorrent/tt-forge-models`.

- Added `_find_real_load_gguf_checkpoint()`: traverses the patcher chain via `__globals__['_orig_load_gguf_checkpoint']` and `__closure__` cells, terminating when `fn.__module__ == "transformers.modeling_gguf_pytorch_utils"`. Returns the original function, bypassing all cross-loader clobbers.
- Added `_make_qwen35moe_gguf_checkpoint_wrapper(real_fn)`: wraps the real function with `(*args, **kwargs)` signature (compatible with transformers 5.2.0 `model_to_load` param). Converts `model_type="qwen35moe"` → `"qwen3_5_moe_text"` and builds `layer_types` from `full_attention_interval`.
- Added `_find_real_get_gguf_hf_weights_map()`: same closure-traversal pattern for `get_gguf_hf_weights_map`.
- Added `_make_qwen35moe_weights_map_wrapper(real_fn)`: extends the GGUF→HF name map with `ffn_gate_exps` and `ffn_up_exps` no-suffix aliases pointing to the fused `gate_up_proj` HF parameter name.
- In `load_model()`, re-applies both wrappers immediately before `AutoModelForCausalLM.from_pretrained()` to defeat any clobbering that happens between module load and model load.

**Test config (tt-xla):**

Added `KNOWN_FAILURE_XFAIL` entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml` on branch `remediation/mirxa2-mirxa-3-5-35b-a3b-uncensored-aggressive-gguf-single-device-inference` in `tenstorrent/tt-xla`.

## Verification
- pytest exit: FAIL (segfault/OOM after successful GGUF load and StableHLO compilation)
- Hardware:    p150b
- Duration:    ~30 minutes (GGUF download + compilation + OOM crash)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mirxa2_mirxa_3_5_35b_a3b_uncensored_aggressive_gguf/causal_lm/pytorch/loader.py` (loader bugs)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` (XFAIL entry)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | da9ccdfbbf5bd1ec9bb64bb6efc7cedeb350b460 |
| tt-forge-models | 0fc2c4e32664fc142fc4ab4d1f9660e997d8c314 |
