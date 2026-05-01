# Remediation Summary: deepseek-deepseek_r1_0528_qwen3-pytorch-Qwen3_8B_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_r1_0528_qwen3/pytorch-Qwen3_8B_bnb_4bit-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

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
E   ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

## Root cause
The loader for `deepseek/deepseek_r1_0528_qwen3/pytorch` was missing a `requirements.txt` declaring `bitsandbytes>=0.46.1`. When `transformers` tries to load the BnB-quantized checkpoint, `quantizer_bnb_4bit.py:validate_environment` raises `ImportError` before any weights are loaded. Additionally, even if bitsandbytes were installed, the model would contain `bnb.nn.Linear4bit` layers that TT hardware cannot execute (no CUDA kernels), so dequantization to `nn.Linear` (bfloat16) is required before running on device.

## Fix
Two changes in `deepseek/deepseek_r1_0528_qwen3/pytorch/`:

1. **`requirements.txt`** (new file): declares `bitsandbytes>=0.46.1`.
2. **`loader.py`**: Added `_dequantize_bnb4_to_bf16(model)` static method that walks `model.named_modules()`, finds every `bnb.nn.Linear4bit`, dequantizes its weight via `bnb.functional.dequantize_4bit`, and replaces it with a standard `nn.Linear(dtype=torch.bfloat16)`. Called immediately after `AutoModelForCausalLM.from_pretrained`. The `bitsandbytes` import is lazy (inside the function) so the loader can be collected before `requirements.txt` installs the package.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    183.59s
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_r1_0528_qwen3/pytorch/requirements.txt` (new)
- `deepseek/deepseek_r1_0528_qwen3/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | b2fda151f6cba9474a30c649668c18248820f036 |
