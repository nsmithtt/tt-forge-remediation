# Remediation Summary: deepseek_r1_medical_cot/causal_lm/pytorch-DeepSeek_R1_Medical_COT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_medical_cot/causal_lm/pytorch-DeepSeek_R1_Medical_COT-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
peft-adapter-redirects-to-bnb-quantized-base

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
The HuggingFace repo `rwibawa/DeepSeek-R1-Medical-COT` ships both an
`adapter_config.json` (a LoRA adapter trained on top of
`unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit`) and full merged
model weights (`pytorch_model-*.bin`). Transformers 5.x detects the PEFT
adapter in two independent places:

1. `auto_factory.py` (`AutoModelForCausalLM.from_pretrained`) reads
   `adapter_config.json`, extracts `base_model_name_or_path`, and changes
   the effective model path to `unsloth/deepseek-r1-distill-llama-8b-unsloth-bnb-4bit`.
2. `modeling_utils.py` (`PreTrainedModel.from_pretrained`) calls
   `maybe_load_adapters` which also detects the adapter.

Both redirect loading to the 4-bit quantized base model whose `config.json`
contains `quantization_config: {load_in_4bit: True}`. This triggers the
bitsandbytes validation check, which fails because bitsandbytes is not
installed. The merged weights in the repo are never used.

## Fix
`deepseek_r1_medical_cot/causal_lm/pytorch/loader.py` in tt-forge-models:

- Import `LlamaForCausalLM` directly (bypasses `auto_factory.py` PEFT
  detection, which only fires inside the Auto-class dispatch loop).
- Temporarily set `transformers.integrations.peft.is_peft_available = lambda: False`
  to suppress the `maybe_load_adapters` PEFT detection in `modeling_utils.py`.
- Load config from the original repo (which has no `quantization_config`),
  then call `LlamaForCausalLM.from_pretrained` with that config so the
  merged weights are loaded as bfloat16 without bitsandbytes.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    103.98s (1:43)
- Tier A attempts: N/A

## Files changed
- `deepseek_r1_medical_cot/causal_lm/pytorch/loader.py` (tt-forge-models, remediation/deepseek-r1-medical-cot)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 07e3c6869ffb5784bd3f320f8a9fbe39622ea0b8 |
| tt-forge-models | b69b2878f97245e0402076c7c84fe33f8c054309 |
