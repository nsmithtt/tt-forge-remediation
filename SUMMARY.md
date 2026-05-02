# Remediation Summary: mistral_small_3_1_pytorch_single_device_inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/mistral_small_3_1/pytorch-unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit-single_device-inference]

## Result
XFAIL — 24B BF16 model (48 GB after BnB dequantization) exceeds single p150b DRAM (~32 GB)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
bnb-4bit-missing-requirements-and-dequantize

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

## Root cause
Two loader bugs:

1. **Missing requirements.txt**: The `mistral/mistral_small_3_1/pytorch/` directory had no `requirements.txt`, so `bitsandbytes` was never installed before loading. Transformers raises `ImportError` from `quantizer_bnb_4bit.py:validate_environment` when a BnB 4-bit model is loaded without `bitsandbytes` installed.

2. **No BnB dequantization**: TT hardware has no CUDA BnB kernels. `Linear4bit` modules must be replaced with plain `nn.Linear` (bf16). Unsloth's BnB 4-bit format stores weights in bf16 in the safetensors files even when the module structure uses `Linear4bit` (due to `quantization_config`), so `module.weight` is a plain `Parameter` without `quant_state`. The dequantization function handles both cases: if `quant_state` is present, use `bnb.functional.dequantize_4bit`; otherwise cast directly to bf16.

After these loader fixes, the model compiles and executes on device, but OOMs: the 24B parameter model at bf16 requires ~48 GB, exceeding the p150b DRAM capacity (~32 GB, 8 banks × 4 GB/bank). This is a hardware capacity ceiling, not a compiler bug.

Additional fixes in the same commit:
- `load_shard_spec`: corrected attribute paths from `model.language_model.layers` → `model.model.language_model.layers` and `model.vision_tower.vision_model.encoder.layers` → `model.model.vision_tower.transformer.layers`, with correct sub-layer names (`attention.q_proj`, `feed_forward.{gate,up,down}_proj`).
- `get_image_features` patch: compute `split_sizes` on CPU to avoid TT int64 → bfloat16 rounding (2310 → 2320) that would break `torch.split`.

## Fix
**loader** (`tt_forge_models/mistral/mistral_small_3_1/pytorch/`):
- Added `requirements.txt` with `bitsandbytes>=0.46.1`.
- Added `_dequantize_bnb4_to_bf16()` called after `from_pretrained`; handles both `Params4bit` (standard BnB) and plain `Parameter` (unsloth bf16-stored) weights.
- Fixed `load_shard_spec` attribute paths for `Mistral3ForConditionalGeneration`.
- Patched `model.model.get_image_features` to compute `split_sizes` on CPU.

**test config** (`tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`):
- Added `KNOWN_FAILURE_XFAIL` for this test with OOM reason.

## Verification
- pytest exit: PASS (xfail)
- Hardware:    blackhole-p150b
- Duration:    363.04s (0:06:03)
- Tier A attempts: N/A

## Files changed
- `mistral/mistral_small_3_1/pytorch/loader.py` (tt_forge_models)
- `mistral/mistral_small_3_1/pytorch/requirements.txt` (tt_forge_models, new)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f13058706bad557785f26695e3fbd12c313add23 |
| tt-forge-models | 2e89d41b8a3f7330ab962d672988cd9fb65acb1b |
