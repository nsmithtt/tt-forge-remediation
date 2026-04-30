# Remediation Summary: inferencerlabs_qwen3_5_27b_mlx_9bit-image_to_text-pytorch-27B_MLX_9bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[inferencerlabs_qwen3_5_27b_mlx_9bit/image_to_text/pytorch-27B_MLX_9bit-single_device-inference]

## Result
XFAIL — Qwen3.5-27B at BF16 (~54 GB) exceeds n150 (12 GB) and p150b (32 GB) DRAM; additionally the loader is classified as image_to_text but the HF repo contains no vision encoder and MLX affine quantization is unsupported by transformers

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-qwen35-27b-exceeds-single-device-dram

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

Reproduced locally as:
```
OSError: Can't load image processor for 'inferencerlabs/Qwen3.5-27B-MLX-9bit'. If you were
trying to load it from 'https://huggingface.co/models', make sure you don't have a local
directory with the same name. Otherwise, make sure 'inferencerlabs/Qwen3.5-27B-MLX-9bit' is
the correct path to a directory containing a preprocessor_config.json file
```

## Root cause
`inferencerlabs/Qwen3.5-27B-MLX-9bit` is an Apple MLX-quantized text-only language model
(base: `Qwen/Qwen3.5-27B`, 27B parameters, 64 layers, hidden_size=5120). Despite the HuggingFace
`config.json` listing `architectures: ['Qwen3_5ForConditionalGeneration']` (a multimodal class in
transformers 5.7.0), the model repository contains **no vision tensors and no preprocessor_config.json**.
The safetensors weights (3 shards, 30.26 GB total) are stored in MLX affine-quantized format
(uint8 weights + float32 scale/bias per group-of-32), which is not supported by standard
HuggingFace transformers quantization backends.

The loader (`image_to_text/pytorch/loader.py`) makes two wrong calls:
1. `AutoProcessor.from_pretrained(...)` — fails immediately with OSError because there is no
   `preprocessor_config.json` (no image processor) in the repo.
2. `AutoModelForImageTextToText.from_pretrained(...)` — wrong class for a text-only model.

Even if the loader were corrected to use `AutoModelForCausalLM` + `AutoTokenizer`:
- MLX affine quantization (`mode: affine`) is not a recognized HuggingFace quantization format;
  transformers would load the weights as raw uint8 tensors with mismatched parameter names.
- Loading the model in bfloat16 (required for TT computation) would consume ~54 GB of device DRAM
  (27B params × 2 bytes), exceeding both n150 (12 GB) and p150b (32 GB) single-device DRAM.

The CI timeout was caused by the test attempting to download 30 GB of safetensors from HuggingFace
(not cached in CI), hitting the configured timeout before any failure could be reported.

## Fix
Added `KNOWN_FAILURE_XFAIL` entry to
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` in the tt-xla repo
(branch `remediation/inferencerlabs_qwen3_5_27b_mlx_9bit-image_to_text-pytorch-27B_MLX_9bit-single_device-inference`,
commit `71bb0ea57`):

```yaml
inferencerlabs_qwen3_5_27b_mlx_9bit/image_to_text/pytorch-27B_MLX_9bit-single_device-inference:
  status: KNOWN_FAILURE_XFAIL
  reason: "Hardware capacity: Qwen3.5-27B MLX 9-bit (27B params) dequantizes to ~54 GB BF16,
    exceeding n150 (12 GB) and p150b (32 GB) DRAM. Also: loader uses
    AutoModelForImageTextToText for a text-only model (no preprocessor_config.json in HF repo),
    and MLX affine quantization (mode=affine) is not supported by transformers."
```

## Verification
- pytest exit: not-run (hardware capacity ceiling; model cannot load on any available device)
- Hardware:    p150b
- Duration:    N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 71bb0ea578174b1b98ce3d71c8365e0edfdd73b4 |
| tt-forge-models | 4b4e081580e6d368682c5ac26e360388489e7638 |
