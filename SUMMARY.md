# Remediation Summary: deepseek-deepseek_v3_0324_nvfp4-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek/deepseek_v3_0324_nvfp4/pytorch-single_device-inference]

## Result
XFAIL — DeepSeek-V3-0324 is 671B parameters; the NVFP4 variant requires 8x NVIDIA B200 GPUs with TensorRT-LLM and cannot run on single-device n150 (12 GB DRAM)

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-deepseek-v3-671b-exceeds-n150

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'. Did you mean: 'get_seq_length'?

In modeling_deepseek.py:1427 (and lines 797, 928):
    past_key_values_length = past_key_values.get_usable_length(seq_length)

## Root cause
Two distinct issues:

1. **Loader bug (loader layer)**: `DynamicCache.get_usable_length` was removed in
   transformers 5.x. The remote model code (`modeling_deepseek.py`, fetched from
   `deepseek-ai/DeepSeek-V3-0324` via `trust_remote_code=True`) calls
   `past_key_values.get_usable_length()` in three places. The current
   `DynamicCache` only exposes `get_seq_length(layer_idx)`. This causes
   `AttributeError` before any silicon execution.

2. **Hardware capacity (hardware-class)**: DeepSeek-V3-0324 has 671B parameters
   (hidden_size=7168, 61 layers, 256 MoE experts). Even with NVFP4 (4-bit)
   quantization the full model requires 8x NVIDIA B200 GPUs with TensorRT-LLM.
   Single-device n150 has 12 GB DRAM — the model cannot fit. The existing loader
   also contains pre-existing model trimming (hidden_size 7168→1024, 61→6 layers)
   which is a forbidden workaround that was masking the hardware capacity failure.

## Fix
**Loader fix** (`deepseek/deepseek_v3_0324_nvfp4/pytorch/loader.py` in
`tt-forge-models` on branch
`remediation/deepseek-deepseek_v3_0324_nvfp4-pytorch-single_device-inference`):
Added `config.use_cache = False` to the model config before calling
`AutoModelForCausalLM.from_config()`. This prevents instantiation of `DynamicCache`
and avoids the removed `get_usable_length` API entirely, consistent with the
pattern used in `tiny_random_minicpm/causal_lm/pytorch/loader.py`.

**Test config** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
in `tt-xla` on branch
`remediation/deepseek-deepseek_v3_0324_nvfp4-pytorch-single_device-inference`):
Added `KNOWN_FAILURE_XFAIL` entry explaining the hardware capacity ceiling.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    163.71s (original failing run)
- Tier A attempts: N/A

## Files changed
- `deepseek/deepseek_v3_0324_nvfp4/pytorch/loader.py` (tt-forge-models): add `config.use_cache = False`
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla): add `KNOWN_FAILURE_XFAIL` entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 59f70e2557b65a81f9033f598506e59f53e4aa2e |
| tt-forge-models | 8127928bc9a4f710e709627ec3123f18776cdb94 |
