# Remediation Summary: llama2_70b_oasst_sft_v10_gptq-causal_lm-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama2_70b_oasst_sft_v10_gptq/causal_lm/pytorch-single_device-inference]

## Result
XFAIL — Llama2-70B GPTQ dequantized to BF16 requires ~140 GB, exceeding p150b 32 GB DRAM

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gptq-missing-optimum-dep

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_gptq.py:48: in __init__
    raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
E   ImportError: Loading a GPTQ quantized model requires optimum (`pip install optimum`)
```

## Root cause
Two problems compound here. First, the loader bug: `transformers` 5.x GPTQ quantizer requires `optimum>=1.24.0` (and for CPU support, `gptqmodel>=1.4.3`), neither of which is installed in the tt-xla environment (which uses torch 2.9.1+cpu). The loader has no requirements.txt and makes no attempt to strip the `quantization_config` before calling `from_pretrained`, so transformers immediately raises `ImportError`.

Second, the hardware-class ceiling: the model is TheBloke/Llama2-70B-OASST-SFT-v10-GPTQ, a 70B-parameter 4-bit GPTQ model. Even at 4-bit, the raw weights are ~35 GB (exceeding the p150b's 32 GB DRAM). After the necessary dequantization to BF16 for TT execution, the model requires ~140 GB DRAM — 4.4× the device capacity. There is no path to run this model on a single p150b.

## Fix
**Loader fix** (tt-forge-models `remediation/llama2_70b_oasst_sft_v10_gptq-causal_lm-pytorch-single_device-inference`):

`llama2_70b_oasst_sft_v10_gptq/causal_lm/pytorch/loader.py`: Added `_dequantize_gptq_weights()` helper that strips `quantization_config` from the `AutoConfig` before `from_pretrained` (so transformers doesn't invoke the GPTQ quantizer), then manually reads GPTQ int4-packed weights from the safetensors shards and injects dequantized BF16 weights into the model. This is the same pattern used for `llama_3_2_3b_gptq_4bit_128g`.

**Test config** (tt-xla `remediation/llama2_70b_oasst_sft_v10_gptq-causal_lm-pytorch-single_device-inference-v2`):

`tests/runner/test_config/torch/test_config_inference_single_device.yaml`: Added `KNOWN_FAILURE_XFAIL` entry for this test with reason explaining the hardware capacity ceiling.

## Verification
- pytest exit: XFAIL (confirmed by prior run: exit code 0; 1 xfailed)
- Hardware:    blackhole-p150b
- Duration:    2152.21s (0:35:52)
- Tier A attempts: N/A
- OOM: TT_FATAL: Out of Memory: Not enough space to allocate 469762048 B DRAM buffer across 8 banks (allocated: 4113318080 B ~30.6 GB total, free: 160071936 B)

## Files changed
- `llama2_70b_oasst_sft_v10_gptq/causal_lm/pytorch/loader.py` (tt-forge-models)
- `tests/runner/test_config/torch/test_config_inference_single_device.yaml` (tt-xla)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 23073ba0a8f9c4cf1915fd31889939b2a3205a1b |
| tt-forge-models | 8752efa44a73c8af0d95426e2a3cd68e5aeb557e |
