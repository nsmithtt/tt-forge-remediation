# Remediation Summary: glm_4_6v_fp8-conditional_generation-pytorch-glm_4_6v_fp8-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_6v_fp8/conditional_generation/pytorch-glm_4_6v_fp8-single_device-inference]

## Result
XFAIL — model too large for n150 (128 experts × 46 MoE layers ≈ 100 GB FP8, 12 GB DRAM limit)

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-glm-4-6v-fp8-n150

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
The image processor of type `Glm46VImageProcessor` is now loaded as a fast processor by default,
even if the model checkpoint was saved with a slow processor. This is a breaking change and may
produce slightly different outputs. To continue using the slow processor, instantiate this class
with `use_fast=False`.
```

After adding `use_fast=False`, the test revealed `compressed_tensors` not installed:
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization.
```

After installing `compressed_tensors`, the test revealed FP8 grouped_mm unsupported:
```
RuntimeError: Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Float8_e4m3fn
```
in `transformers/integrations/moe.py:184: torch._grouped_mm(input.to(weight.dtype), weight, offs=offs)`

## Root cause
Three loader bugs were found and fixed:

1. **Transformers 5.x image processor breaking change**: `AutoProcessor.from_pretrained` for
   `Glm46VImageProcessor` now defaults to the fast processor. Fixed by passing `use_fast=False`.

2. **Missing `compressed-tensors` requirement**: The FP8 checkpoint requires `compressed-tensors`
   for quantization config parsing. Fixed by adding `requirements.txt`.

3. **FP8 expert weights unsupported by torch._grouped_mm on CPU/TT**: `Glm4vMoeTextNaiveMoe`
   stores `gate_up_proj` and `down_proj` as stacked `nn.Parameter` tensors in float8_e4m3fn.
   `compressed-tensors` wraps `nn.Linear` layers with dequantization logic but cannot wrap
   stacked `nn.Parameter`s. PyTorch 2.9's `_grouped_mm_experts_forward` calls
   `torch._grouped_mm(input.to(weight.dtype), weight, offs)` which fails on CPU/TT when
   weight.dtype is float8_e4m3fn. Fixed by post-load dequantization: iterate all
   `Glm4vMoeTextNaiveMoe` modules and cast `gate_up_proj.data` and `down_proj.data` to bfloat16.

**Hardware capacity ceiling**: The model has 128 routed experts × 46 MoE layers ×
(2 × 1408 × 4096 + 4096 × 1408) parameters ≈ 102 billion expert parameters. In FP8
(1 byte/param) this is ~102 GB; in BF16 (after our dequantization) ~204 GB. The n150 device
has ~12 GB of single-device DRAM. This model cannot run on n150 under any quantization that
the current TT stack supports. The same hardware-class ceiling applies as GLM-4.5 Air (glm4moe).

## Fix
Three fixes in `tt_forge_models` remediation branch, one test config entry in `tt-xla`:

**tt-forge-models** (`remediation/glm_4_6v_fp8-conditional_generation-pytorch-glm_4_6v_fp8-single_device-inference`):
- `glm_4_6v_fp8/conditional_generation/pytorch/loader.py`:
  - Added `use_fast=False` to `AutoProcessor.from_pretrained`
  - Added `_dequantize_fp8_experts()` post-load helper that casts FP8 `gate_up_proj` and
    `down_proj` Parameters to bfloat16 in all `Glm4vMoeTextNaiveMoe` modules
- `glm_4_6v_fp8/conditional_generation/pytorch/requirements.txt`: added `compressed-tensors`

**tt-xla** (`remediation/glm_4_6v_fp8-conditional_generation-pytorch-glm_4_6v_fp8-single_device-inference`):
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: added
  `KNOWN_FAILURE_XFAIL` entry with hardware capacity reason

## Verification
- pytest exit: not-run (hardware-class XFAIL — model ~100 GB FP8, cannot load onto n150 12 GB DRAM)
- Hardware:    n150
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- `tt-xla/third_party/tt_forge_models` (submodule pointer)
- `tt_forge_models/glm_4_6v_fp8/conditional_generation/pytorch/loader.py`
- `tt_forge_models/glm_4_6v_fp8/conditional_generation/pytorch/requirements.txt`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7f79d521f9ca70eb6134b4c9c4caa9405b5dc4dd |
| tt-forge-models | aebd1d02c799a496b30a18b61826e555239475f5 |
