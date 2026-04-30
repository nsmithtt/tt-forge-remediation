# Remediation Summary: gpt_oss_120b_uncensored_bf16_gguf-causal_lm-pytorch-GPT_OSS_120B_UNCENSORED_BF16_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gpt_oss_120b_uncensored_bf16_gguf/causal_lm/pytorch-GPT_OSS_120B_UNCENSORED_BF16_Q4_K_M_GGUF-single_device-inference]

## Result
XFAIL — 120B model at Q4_K_M (~67 GB) far exceeds n150 DRAM (12 GB); hardware capacity ceiling

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
gguf-single-file-reorganized-to-shards

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
OSError: bartowski/huizimao_gpt-oss-120b-uncensored-bf16-GGUF does not appear to have a file named huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M.gguf. Checkout 'https://huggingface.co/bartowski/huizimao_gpt-oss-120b-uncensored-bf16-GGUF/tree/main' for available files.

(The "sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute" in the failure message is a harmless swig warning printed after the actual OSError.)

## Root cause
The HuggingFace repository `bartowski/huizimao_gpt-oss-120b-uncensored-bf16-GGUF` reorganized its GGUF files from a single flat file at the repo root (`huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M.gguf`) into per-quantization subdirectories with multi-shard naming (`huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M/huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M-00001-of-00002.gguf`). The loader's `GGUF_FILE` constant still referenced the old flat path, causing `AutoTokenizer.from_pretrained` to fail with a 404.

Additionally, the full 120B model at Q4_K_M quantization is approximately 67 GB — far beyond the n150's ~12 GB DRAM capacity. Even with the corrected file path, the model cannot run on this hardware.

## Fix
Two changes in `tt_forge_models` (branch `remediation/gpt_oss_120b_uncensored_bf16_gguf-causal_lm-pytorch-GPT_OSS_120B_UNCENSORED_BF16_Q4_K_M_GGUF-single_device-inference`):

1. **Loader fix** (`gpt_oss_120b_uncensored_bf16_gguf/causal_lm/pytorch/loader.py`): Updated `GGUF_FILE` from `"huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M.gguf"` to `"huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M/huizimao_gpt-oss-120b-uncensored-bf16-Q4_K_M-00001-of-00002.gguf"` to reference the first shard in the new directory layout.

2. **Requirements** (`gpt_oss_120b_uncensored_bf16_gguf/causal_lm/pytorch/requirements.txt`): Added `gguf>=0.10.0`.

One change in `tt-xla` (branch `remediation/gpt_oss_120b_uncensored_bf16_gguf-causal_lm-pytorch-GPT_OSS_120B_UNCENSORED_BF16_Q4_K_M_GGUF-single_device-inference`):

3. **Test config** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`): Added `KNOWN_FAILURE_XFAIL` entry with reason explaining the hardware capacity ceiling.

## Verification
- pytest exit: XFAIL (1 xfailed)
- Hardware:    n150
- Duration:    18.76s
- Tier A attempts: N/A

## Files changed
- `gpt_oss_120b_uncensored_bf16_gguf/causal_lm/pytorch/loader.py` (tt_forge_models)
- `gpt_oss_120b_uncensored_bf16_gguf/causal_lm/pytorch/requirements.txt` (tt_forge_models, new)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 5b85073695682d062a0ac7fe5888bfb5b410853d |
| tt-xla          | 078f77bd5553101363708bcf2026db32c0909a5b |
| tt-forge-models | 49ee296f79c9fec7af222dc69e81f98c298e416b |
