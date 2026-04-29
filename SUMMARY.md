# Remediation Summary: arthurcollet_qwen3_5_27b_mlx_nvfp4-causal_lm-pytorch-Qwen3_5_27B_mlx_nvfp4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[arthurcollet_qwen3_5_27b_mlx_nvfp4/causal_lm/pytorch-Qwen3_5_27B_mlx_nvfp4-single_device-inference]

## Result
XFAIL — 27.3B BF16 model (~54 GB) exceeds single-device DRAM capacity (~24 GB p150b); MLX FP4 checkpoint format also requires custom dequantization infrastructure

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
mlx-affine-quantized-model-incompatible-format

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
The test hung indefinitely. The reported failure message (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`) is a Python 3.12 SWIG warning unrelated to the hang.

Root hang: `AutoModelForCausalLM.from_pretrained('arthurcollet/Qwen3.5-27B-mlx-nvfp4', torch_dtype=torch.bfloat16)` permanently blocked inside `_initialize_missing_keys()`.

## Root cause

Two loader-layer bugs conspired to produce the hang, and a hardware-capacity ceiling means the test cannot pass on single-device silicon even with both fixed.

**Bug 1 — key prefix mismatch (loader)**

The checkpoint's `config.json` declares `"architectures": ["Qwen3_5ForConditionalGeneration"]` (the VLM wrapper), whose parameter namespace is `model.language_model.*`.  All 1 349 safetensors keys use the prefix `language_model.model.*`.  `rename_source_key()` (with `base_model_prefix = "model"`) cannot bridge this gap, so `param_name_to_load` ends up empty.  `_initialize_missing_keys()` then randomly initialises all 27.3 B parameters (~54 GB BF16) — causing the indefinite hang.

**Bug 2 — unsupported quantization format (loader)**

The checkpoint stores weights as `torch.uint32` (8 packed E2M1 fp4 nibbles per 32-bit word) with `torch.uint8` group scales (one fp8 E4M3fn scale per 16 fp4 values, `group_size=16`).  `quantization_config` has no `quant_method` field, so `AutoHfQuantizer.supports_quant_method()` raises `ValueError` and `pre_quantized` is set to `False`.  There is no transformers handler for this MLX-native format; the weights must be dequantized manually.

**Hardware capacity ceiling**

After dequantizing to BF16 the model occupies ≈54.6 GB (27.3 B params × 2 bytes).  Tenstorrent p150b has ≈24 GB DRAM, so the model cannot fit on a single device.  This is a genuine hardware-class ceiling, not a compiler bug.

## Fix

**Loader fix (committed to `remediation/arthurcollet_qwen3_5_27b_mlx_nvfp4-causal_lm-pytorch-Qwen3_5_27B_mlx_nvfp4-single_device-inference` in tt_forge_models):**

`arthurcollet_qwen3_5_27b_mlx_nvfp4/causal_lm/pytorch/loader.py` was rewritten to:
1. Load `full_config.text_config` and instantiate `Qwen3_5ForCausalLM(text_config)` directly, bypassing the VLM wrapper.
2. Read shards via `safetensors.safe_open`, keeping only `language_model.*` keys and stripping the prefix to produce `Qwen3_5ForCausalLM`-compatible state-dict keys.
3. Implement `_dequantize_nvfp4()`: unpack E2M1 fp4 nibbles from `uint32` words (low nibble first), apply fp8 E4M3fn group scales from `uint8`, return `bfloat16`.
4. Load with `strict=False, assign=True`; raise `RuntimeError` if any keys remain missing after the load.
5. Update `load_shard_spec()` to branch on `layer.layer_type` (`"full_attention"` / `"linear_attention"`) since the model is an SSM-attention hybrid with 64 layers (48 `Qwen3_5GatedDeltaNet` + 16 standard attention).

**Test config (committed to `remediation/arthurcollet_qwen3_5_27b_mlx_nvfp4-causal_lm-pytorch-Qwen3_5_27B_mlx_nvfp4-single_device-inference` in tt-xla):**

Added `KNOWN_FAILURE_XFAIL` entry for this test in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

## Verification
- pytest exit: not-run (hardware-class XFAIL; no silicon run attempted)
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/arthurcollet_qwen3_5_27b_mlx_nvfp4/causal_lm/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 5d2853a14e41d2b7e82023c43e4b3254a7a4e77c |
| tt-forge-models | 7e991f5435eae3df60c7e9a93151b0315a7db902 |
