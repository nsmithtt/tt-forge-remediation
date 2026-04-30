# Remediation Summary: gemma_2_awq-causal_lm-pytorch-9B_IT_AWQ_INT4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_awq/causal_lm/pytorch-9B_IT_AWQ_INT4-single_device-inference]

## Result
SILICON_PASS — added gptqmodel dependency and dequantized AWQ layers before TT compilation

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
awq-gptqmodel-missing-dep-and-torchaten-awq-linear-not-dequantized

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`
```

## Root cause
Two related loader bugs:

1. **Missing `gptqmodel` dependency**: transformers 5.x's AWQ quantizer (`transformers/quantizers/quantizer_awq.py`) requires `gptqmodel` to be installed. The `gemma_2_awq` loader had no `requirements.txt`, so the package was absent and `AutoModelForCausalLM.from_pretrained` raised `ImportError` before the model could load.

2. **AWQ layers not dequantized**: After fixing the import error, a second loader bug appeared: `load_shard_spec` and ultimately the TT device runner called the model with `TorchAtenAwqLinear` layers still in place. gptqmodel's `TorchAtenAwqLinear._fused_op_forward` uses a CPU-only kernel (`torch.ops.aten._weight_int4pack_mm_for_cpu`) and raises `NotImplementedError` when x.device != 'cpu'. The TT device run fails because TT tensors are not CPU tensors. The immediate symptom on the first run was `AttributeError: 'TorchAtenAwqLinear' object has no attribute 'weight'` in `load_shard_spec`, because AWQ layers use `.qweight` not `.weight`.

Fix: dequantize all AWQ quantized linear layers to standard `nn.Linear` (bfloat16) immediately after model load, before any forward pass.

## Fix
Two changes in `tt-forge-models` on branch `remediation/gemma_2_awq-causal_lm-pytorch-9B_IT_AWQ_INT4-single_device-inference` (commit `dedc61ed61`):

1. **`gemma_2_awq/causal_lm/pytorch/requirements.txt`** (new file): adds `gptqmodel` so the RequirementsManager installs it before the test runs.

2. **`gemma_2_awq/causal_lm/pytorch/loader.py`**: adds `_dequantize_awq_layers()` helper (same pattern as the gemma3 AWQ remediation) and calls it in `load_model()` after `from_pretrained`. The function walks `model.named_modules()`, finds layers with `awq_weight_dequantize` (all `TorchAtenAwqLinear` instances), dequantizes each to bfloat16, and replaces it with a standard `nn.Linear`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    275.62s (0:04:35)
- Tier A attempts: N/A

## Files changed
- `gemma_2_awq/causal_lm/pytorch/requirements.txt` (new)
- `gemma_2_awq/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 421baf2194172e53fac2bb86204858741d7b60ff |
| tt-forge-models | dedc61ed615336e23d391bbe39fa6d246156a374 |
