# Remediation Summary: openai_oss_20b_evo-causal_lm-pytorch-Default-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[openai_oss_20b_evo/causal_lm/pytorch-Default-single_device-inference]

## Result
XFAIL — 20B model (~40 GB BF16) exceeds p150b 32 GB single-device DRAM capacity

## Stack layer
hardware-class

## Tier
N/A

## Bug fingerprint
hardware-capacity-gptoss-20b-single-device

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: You set `ignore_mismatched_sizes` to `False`, thus raising an error. For details look at the above report!
```

From `transformers.utils.loading_report.log_state_dict_report` triggered in
`loader.py:95` during `AutoModelForCausalLM.from_pretrained("Cyborg-AI/openai_oss_20b_evo")`.

The loading report showed:
- MISMATCH: `self_attn.{q,k,v,o}_proj.weight` — checkpoint stores NF4-packed uint8 flat
  tensors (e.g. `[737280,1]` for a `[512,2880]` weight) because the base model was
  saved with BitsAndBytes NF4 quantization.
- MISSING: all MLP expert parameters (`experts.gate_up_proj`, `experts.down_proj`,
  router weights) — the checkpoint (saved with transformers 4.55) stores each expert
  as an individual `Linear` module (`experts.gate_up_projs.{i}.weight`) while
  transformers 5.2 expects batched `nn.Parameter` tensors.

## Root cause
`Cyborg-AI/openai_oss_20b_evo` is a PEFT LoRA adapter repo. Its `adapter_config.json`
references base model `unsloth/gpt-oss-20b-unsloth-bnb-4bit`, which stores attention
projection weights in BitsAndBytes NF4 quantized format. When loaded via
`AutoModelForCausalLM.from_pretrained`, transformers 5.x auto-detects `adapter_config.json`
and loads the base model; without bitsandbytes installed, it reads the NF4-packed tensors
as raw safetensors data, producing shape mismatches against the expected BF16 Linear
parameter shapes, raising RuntimeError.

A secondary issue (not causing the RuntimeError): the base model checkpoint was saved with
transformers 4.55 which used individual per-expert `Linear` modules
(`mlp.experts.gate_up_projs.{i}`) while the current transformers 5.2 `GptOssExperts` class
uses batched `nn.Parameter` tensors. These show as MISSING in the loading report.

After the loader fix, the model still cannot run on single-device hardware: the full
model at BF16 is ~40 GB (20B × 2 bytes), which exceeds the p150b 32 GB DRAM limit.

## Fix
**Loader fix** (`openai_oss_20b_evo/causal_lm/pytorch/loader.py`, `requirements.txt`
in `tt-forge-models` on branch
`remediation/openai_oss_20b_evo-causal_lm-pytorch-Default-single_device-inference`):

- Import `BitsAndBytesConfig` from transformers.
- Add `_dequantize_bnb_model()` helper that iterates all `bnb.nn.Linear4bit` modules,
  dequantizes via `bitsandbytes.functional.dequantize_4bit`, and replaces them with
  standard `nn.Linear`.
- In `load_model`: add `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True)` and `device_map="cpu"` to `from_pretrained` kwargs;
  call `_dequantize_bnb_model` after loading.
- Add `bitsandbytes>=0.46.1` to `requirements.txt`.

**Test config XFAIL** (`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
in `tt-xla`):

- Added `openai_oss_20b_evo/causal_lm/pytorch-Default-single_device-inference:
  status: KNOWN_FAILURE_XFAIL` with hardware capacity reason.

## Verification
- pytest exit: FAIL (not run on silicon — hardware capacity XFAIL)
- Hardware:    blackhole-p150b
- Duration:    not-run (loader fix cannot be verified due to hardware capacity)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/openai_oss_20b_evo/causal_lm/pytorch/loader.py`
- `tt-xla/third_party/tt_forge_models/openai_oss_20b_evo/causal_lm/pytorch/requirements.txt`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f508ae5c2c6f58bea7382ca390da6ba27252024b |
| tt-forge-models | e77cf1723dd97160145296c6fb841f314cd69b16 |
