# Remediation Summary: gpt_oss_sft_s1k_i1_gguf-causal_lm-pytorch-GPT_OSS_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_sft_s1k_i1_gguf/causal_lm/pytorch-GPT_OSS_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — GPT-oss-sft-s1K is a ~20B Qwen3-MoE model (~41.8 GB BF16) exceeding single-device DRAM capacity (32 GB on p150b); loader bugs fixed, OOM confirmed on silicon

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-gpt-oss-arch-not-registered

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

With gguf installed, the test then failed with:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

And after fixing the narrow-sig issue, with:
```
ValueError: GGUF model with architecture gpt-oss is not supported yet.
```

## Root cause
Three loader bugs in sequence:

1. **Missing gguf>=0.10.0 dependency** — `requirements.txt` absent; transformers raises ImportError when trying to load GGUF without gguf package.

2. **Cross-loader narrow-sig contamination** — 26 other GGUF loaders (Qwen3.5 variants) installed their own `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` wrapper globally. When `gpt_oss_sft_s1k_i1_gguf` runs in the same session, it calls the contaminated global which rejects the `model_to_load` kwarg added in transformers 5.x.

3. **`gpt-oss` architecture not registered** — The GGUF file declares `general.architecture = gpt-oss` (OpenAI's branding for a Qwen3-MoE architecture). This key is absent from `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_TRANSFORMERS_MAPPING`, causing `ValueError: GGUF model with architecture gpt-oss is not supported yet.`

After all loader bugs are fixed, the model loads as `Qwen3MoeForCausalLM` with 24 layers, 32 experts (~20.91B params, ~41.8 GB BF16) and OOMs on p150b (32 GB DRAM).

## Fix
Three fixes in `tt_forge_models` on branch `remediation/gpt_oss_sft_s1k_i1_gguf-causal_lm-pytorch-GPT_OSS_SFT_S1K_I1_Q4_K_M_GGUF-single_device-inference`:

1. `gpt_oss_sft_s1k_i1_gguf/causal_lm/pytorch/requirements.txt` — added `gguf>=0.10.0`

2. 26 Qwen3.5 GGUF loaders — changed `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` to `(*args, **kwargs)` to stop contaminating the session with a narrow-sig wrapper

3. `gpt_oss_sft_s1k_i1_gguf/causal_lm/pytorch/loader.py` — added:
   - `_patch_gpt_oss_support()`: registers `gpt-oss` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING` (aliased to `qwen3_moe` keys with `expert_feed_forward_length` and `attention.sliding_window` extras), and `GGUF_TO_FAST_CONVERTERS`
   - `_patched_load_gguf_checkpoint(*args, **kwargs)`: applies the patch and fixes `model_type` from `gpt-oss` to `qwen3_moe`
   - Installs the patched function at all four `load_gguf_checkpoint` binding sites
   - `model.config._experts_implementation = "batched_mm"` after load_model to avoid `histc`-on-Int failure in grouped_mm on XLA
   - Chat template guard in `load_inputs`

In `tt-xla`:
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `KNOWN_FAILURE_XFAIL` for the single_device inference variant

## Verification
- pytest exit: XFAIL
- Hardware:    blackhole-p150b
- Duration:    1204.90s (0:20:04)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/gpt_oss_sft_s1k_i1_gguf/causal_lm/pytorch/requirements.txt` (new)
- `tt_forge_models/gpt_oss_sft_s1k_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/<26 Qwen3.5 GGUF loaders>/loader.py` (narrow-sig fix)
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 747bb7b745e40945ac1d89a9870b487d4c2448a4 |
| tt-forge-models | 07cf6a12737383e8ed48b2e9cbc8a0b0199a7f28 |
