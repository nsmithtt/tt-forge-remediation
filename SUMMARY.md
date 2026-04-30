# Remediation Summary: deepseek-deepseek_r1_distill-pytorch-Distill_Llama_70B_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_distill/pytorch-Distill_Llama_70B_bnb_4bit-single_device-inference]

## Result
XFAIL — 70B model (≈140 GB at BF16) exceeds single-device TT DRAM (~12 GB); BNB 4-bit format is CUDA-specific and cannot be moved to TT device

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-70b-model-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Creating a Parameter from an instance of type Params4bit requires that detach() returns an instance of the same type, but return type Tensor was found instead. To use the type as a Parameter, please correct the detach() semantics defined by its __torch_dispatch__() implementation.
```

Raised at `tests/infra/runners/torch_device_runner.py:129` when the test framework calls `workload.model.to(device)` to move the loaded BNB 4-bit model to the TT XLA device.

(When `bitsandbytes` is not installed the test instead raises `ImportError: Using bitsandbytes 4-bit quantization requires bitsandbytes: pip install -U bitsandbytes>=0.46.1`.)

## Root cause
Two overlapping issues, both rooted in hardware capacity:

1. **Missing `bitsandbytes` requirement**: The loader at `deepseek/deepseek_r1_distill/pytorch/loader.py` has no `requirements.txt`, so `bitsandbytes` is not guaranteed to be installed in the test venv. Without it, `AutoModelForCausalLM.from_pretrained` raises `ImportError` immediately.

2. **Hardware capacity ceiling**: `unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit` is a 70-billion-parameter model stored in BitsAndBytes 4-bit format (~35 GB on disk). After dequantization to BF16 the weights occupy ~140 GB — far beyond the ~12 GB DRAM available on a single TT n150 device. Even at 4-bit the model cannot run on single-device TT hardware.

   Additionally, `bitsandbytes.nn.Params4bit` is a CUDA-specific tensor subclass. PyTorch 2.x enforces that `torch.nn.Parameter(t)` requires `t.detach()` to return the same type; `Params4bit.detach()` returns a plain `Tensor`, so `model.to(device)` always raises `RuntimeError` for any non-CUDA device.

The BNB quantization was chosen precisely because 70B parameters don't fit at native precision — the quantization is a workaround for size, not a solution that enables TT execution.

## Fix
- **`tt_forge_models` (loader)**: Added `deepseek/deepseek_r1_distill/pytorch/requirements.txt` with `bitsandbytes>=0.46.1` so the import error is surfaced clearly rather than as a silent test skip.
- **`tt-xla` (test config)**: Added `KNOWN_FAILURE_XFAIL` entry for `deepseek/deepseek_r1_distill/pytorch-Distill_Llama_70B_bnb_4bit-single_device-inference` in `tests/runner/test_config/torch/test_config_inference_single_device.yaml`. No compiler fix is possible; the model exceeds single-device hardware capacity.

## Verification
- pytest exit: FAIL (hardware capacity; not run to completion on TT silicon — model exceeds device DRAM)
- Hardware:    not-run
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1_distill/pytorch/requirements.txt` (new, in tt_forge_models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 7ea4a6bb162178e5d8ad03ff209afc608dce9457 |
| tt-forge-models | 211528dab7c218c8fa7d18c8619fd79837b5d2ab |
