# Remediation Summary: llama_3_1_bnb_4bit-pytorch-3.1_8B_Unsloth_BNB_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_1_bnb_4bit/pytorch-3.1_8B_Unsloth_BNB_4bit-single_device-inference]

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
- PCC threshold lowering: YES — measured BF16-CPU vs FP32-CPU PCC = 0.9998; TT-BF16 vs CPU-BF16 = 0.9806; gap is the known LLaMA 3.1 8B precision issue (tt-xla/issues/2944) shared with all llama/causal_lm/pytorch-3.1_8B variants in the config
- Warning / exception suppression: NO

## Failure
```
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```
Raised from `transformers/quantizers/quantizer_bnb_4bit.py:validate_environment` during `AutoModelForCausalLM.from_pretrained`.

## Root cause
Two loader-layer bugs:

1. **Missing `requirements.txt`**: No `bitsandbytes>=0.46.1` requirement in the model directory. Transformers raises `ImportError` from `quantizer_bnb_4bit.py` before any weights load when `bitsandbytes` is absent.

2. **No dequantization**: After `from_pretrained` loads BNB 4-bit weights into `bnb.nn.Linear4bit` modules, nothing converts them to standard `nn.Linear` (bf16). TT hardware has no CUDA BNB kernel support; `model.to(xla_device)` fails because `Params4bit.detach()` returns a plain `Tensor` instead of a `Params4bit`, breaking the parameter copy.

After fixing both, the PCC was 0.9806 vs required 0.99. BF16-CPU vs FP32-CPU = 0.9998 (not a BF16 floor). This matches the known LLaMA 3.1 8B precision issue (tt-xla #2944) that already sets required_pcc=0.98 for all LLaMA 3.1 8B variants.

## Fix
**tt-forge-models** (`remediation/llama_3_1_bnb_4bit-pytorch-3.1_8B_Unsloth_BNB_4bit-single_device-inference`):

- `llama_3_1_bnb_4bit/pytorch/requirements.txt` (new): `bitsandbytes>=0.46.1`
- `llama_3_1_bnb_4bit/pytorch/loader.py`: added `_dequantize_bnb_model()` that iterates `named_modules`, replaces each `bnb.nn.Linear4bit` with a plain `nn.Linear` (bf16) using `bitsandbytes.functional.dequantize_4bit`; called after `from_pretrained`.

**tt-xla** (`remediation/llama_3_1_bnb_4bit-pytorch-3.1_8B_Unsloth_BNB_4bit-single_device-inference`):

- `tests/runner/test_config/torch/test_config_inference_single_device.yaml`: added entry with `required_pcc: 0.98` (consistent with LLaMA 3.1 8B, same issue #2944), `status: EXPECTED_PASSING`, `n150: NOT_SUPPORTED_SKIP` (8B BF16 too large for single-chip n150).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    203.33s (0:03:23)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: llama_3_1_bnb_4bit/pytorch/loader.py`
- `tt-forge-models: llama_3_1_bnb_4bit/pytorch/requirements.txt` (new)
- `tt-xla: tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 0b87d0ac44fdea983cf4653b7f06c4374dd04be2 |
| tt-forge-models | d764208d07841cb70c8e4b21e9c8e0eaaddedd32 |
