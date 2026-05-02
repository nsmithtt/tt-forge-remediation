# Remediation Summary: mistral_small_3_2_24b_instruct_2506_awq_sym-pytorch-24B_Instruct_2506_AWQ_Sym-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_small_3_2_24b_instruct_2506_awq_sym/pytorch-24B_Instruct_2506_AWQ_Sym-single_device-inference]

## Result
XFAIL — 24B AWQ model decompresses to ~48 GB BF16 on device, exceeding single p150b DRAM (~34 GB); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-24b-bf16-oom-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original reported failure: `raise AttributeError(`

Full failure sequence encountered during debugging:
1. `ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization` — missing requirements.txt with compressed-tensors dependency
2. `AttributeError: 'Mistral3ForConditionalGeneration' object has no attribute 'language_model'` — load_shard_spec used wrong model attribute path
3. Terminal: `TT_FATAL: Out of Memory: Not enough space to allocate 61194895360 B DRAM buffer across 8 banks, where each bank needs to store 7649361920 B, but bank size is 4273390016 B` → surfaces as `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

## Root cause
Two loader bugs preceded the hardware ceiling:

1. **Missing compressed-tensors dependency**: The `jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym` checkpoint uses the compressed-tensors quantization format, which requires the `compressed-tensors` Python package. No `requirements.txt` existed in the loader directory.

2. **Wrong attribute paths in load_shard_spec**: The standalone loader used `model.language_model.layers` and `model.vision_tower.vision_model.encoder.layers`. The actual structure is `model.model.language_model.layers` (inner Mistral3Model) and `model.model.vision_tower.transformer.layers` (PixtralTransformer).

After fixing the loader bugs, the test progresses to model compilation and hits hardware OOM. The compressed-tensors AWQ format stores weights in INT4 on disk (~12 GB for 24B), but the TT device allocates BF16 buffers during compilation: ~48 GB needed vs ~34 GB available (8 banks x 4.27 GB). This is a genuine hardware capacity ceiling.

Also patched `get_image_features` on `model.model` to compute `split_sizes` on CPU (same fix as the `mistral/mistral_small_3_2` AWQ variant) to avoid TT int64->bfloat16 precision loss in prod(). The OOM occurs before this code path is exercised.

## Fix
Loader fixes in `tt-xla/third_party/tt_forge_models/mistral_small_3_2_24b_instruct_2506_awq_sym/pytorch/` (committed to `remediation/mistral-small-3-2-24b-awq-sym` branch in tt-forge-models):
- `requirements.txt`: Added `compressed-tensors` dependency
- `loader.py`:
  - Added `import types` and `import torch`
  - Added `_patch_get_image_features_cpu_split()`: patches `model.model.get_image_features` to compute split_sizes on CPU, with explicit `return_dict=None` param to avoid kwargs conflict with `@can_return_tuple` decorator
  - Fixed `load_shard_spec` to use `model.model.language_model.layers` and `model.model.vision_tower.transformer.layers` (PixtralAttention/PixtralMLP structure)
  - Calls `_patch_get_image_features_cpu_split(model)` after `model.eval()`

Test config update in `tt-xla` (committed to `remediation/mistral-small-3-2-24b-awq-sym` branch in tt-xla):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` status entry with verbatim OOM reason

## Verification
- pytest exit: FAIL (OOM — hardware capacity ceiling)
- Hardware:    blackhole-p150b
- Duration:    ~300s (OOM during compilation)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/mistral_small_3_2_24b_instruct_2506_awq_sym/pytorch/requirements.txt` (new)
- `tt-xla/third_party/tt_forge_models/mistral_small_3_2_24b_instruct_2506_awq_sym/pytorch/loader.py`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 285e49ba122d67f61fb6590ced7e3cf37e2fd6f6 |
| tt-forge-models | 3b3a1b039cf943f495d844b5a6b12e9bc025e346 |
