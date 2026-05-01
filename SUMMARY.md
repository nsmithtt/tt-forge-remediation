# Remediation Summary: ministral_3_8b-pytorch-3_8B_Instruct_2512_bnb_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral/ministral_3_8b/pytorch-3_8B_Instruct_2512_bnb_4bit-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
bnb-4bit-params4bit-detach-returns-tensor

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: Unrecognized configuration class <class 'transformers.models.mistral3.configuration_mistral3.Mistral3Config'> for this kind of AutoModel: AutoModelForCausalLM.
Model type should be one of AfmoeConfig, ..., MinistralConfig, Ministral3Config, MistralConfig, ...
```

After fixing the model class:

```
AttributeError: 'Parameter' object has no attribute 'quant_state'
```

## Root cause
Two loader bugs:

1. **Wrong model class.** The Ministral-3-8B model uses `model_type="mistral3"` (i.e., `Mistral3Config` from `transformers.models.mistral3`). `AutoModelForCausalLM` does not support `Mistral3Config` — it supports a different `Ministral3Config` from `transformers.models.ministral3`. The correct class for loading `Mistral3Config` checkpoints is `Mistral3ForConditionalGeneration`. Both the BF16 and BNB 4-bit variants are affected because both HuggingFace repos (`mistralai/Ministral-3-8B-Instruct-2512-BF16` and `unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit`) report `model_type="mistral3"`.

2. **BnB Params4bit not compatible with XLA device transfer.** After loading with `device_map="cpu"`, bitsandbytes stores weights as `Params4bit` objects. When the test runner calls `model.to(xla_device)`, `torch.nn.Module._apply()` invokes `param.detach()` on each parameter and wraps the result in `nn.Parameter`. However, `Params4bit.detach()` returns a plain `Tensor` (not `Params4bit`), causing `Parameter.__new__` to raise `RuntimeError`. The fix is to dequantize all `Linear4bit` layers to standard bfloat16 `nn.Linear` layers before device transfer. Some CPU-loaded `Linear4bit` modules may already have their weight materialized as a plain `Parameter` (no `quant_state`) — these are handled by direct dtype casting.

3. **Missing `bitsandbytes>=0.46.1` in `requirements.txt`.** The quantized checkpoint requires bitsandbytes to load.

## Fix
In `tt_forge_models`, on branch `remediation/ministral_3_8b-pytorch-3_8B_Instruct_2512_bnb_4bit-single_device-inference`:

1. **`mistral/ministral_3_8b/pytorch/loader.py`**:
   - Changed `AutoModelForCausalLM.from_pretrained()` to `Mistral3ForConditionalGeneration.from_pretrained()` for all variants.
   - Added `_dequantize_bnb4_to_bf16()` static method: iterates over all `bnb.nn.Linear4bit` modules, dequantizes each with `bitsandbytes.functional.dequantize_4bit` (or casts directly if `quant_state` is None), and replaces with a standard `nn.Linear` holding bfloat16 weights. Called in `load_model()` after `from_pretrained()` for the BNB variant.
   - Updated `get_mesh_config()` to use `config.text_config.num_attention_heads` (previously `config.num_attention_heads` which does not exist on `Mistral3Config`).
   - Updated `load_shard_spec()` to use `model.model.language_model.layers` (the correct path for the text layers in `Mistral3ForConditionalGeneration`).

2. **`mistral/ministral_3_8b/pytorch/requirements.txt`** (new file): Added `bitsandbytes>=0.46.1`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    191.15s
- Tier A attempts: N/A

## Files changed
- `mistral/ministral_3_8b/pytorch/loader.py`
- `mistral/ministral_3_8b/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f03ffc07e365306a0d20bcae0511c1e98200683c |
| tt-forge-models | ed30765358c157999fe8bfc689d3133ad3a879a3 |
