# Remediation Summary: flux_gguf-pytorch-Q8_0-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_gguf/pytorch-Q8_0-single_device-inference]

## Result
SILICON_PASS — fixed two loader bugs in flux_gguf: invalid dtype cast on quantized model and GGUFParameter __torch_function__ infinite recursion during torch.compile tracing

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
flux-gguf-dequantize-before-compile

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`
```

After fixing the ValueError, a second failure was exposed:
```
torch._dynamo.exc.InternalTorchDynamoError: RecursionError: maximum recursion depth exceeded
...
diffusers/quantizers/gguf/utils.py, line 564, in __torch_function__
    result = super().__torch_function__(func, types, args, kwargs)
[Previous line repeated 158 more times]
```

## Root cause
Two bugs in `flux_gguf/pytorch/loader.py`:

1. The original loader called `self.transformer.to(dtype_override)` after `FluxTransformer2DModel.from_single_file(...)`. The diffusers `to()` override blocks dtype casting on quantized models (raises `ValueError`). This was unnecessary since `torch_dtype=compute_dtype` was already passed to `from_single_file`.

2. When `GGUFParameter` weights remain in the model and torch.compile (TorchDynamo) traces through the forward pass, `GGUFParameter.__torch_function__` is invoked. Its implementation wraps all results back into `GGUFParameter` via `super().__torch_function__()`, which triggers `__torch_function__` again on the wrapped result — causing infinite recursion. The model must be dequantized to plain tensors before compilation.

Secondary complication: `model.dequantize()` removes `hf_quantizer` but leaves `model.is_quantized = True`, so the diffusers `to()` override still blocks dtype casting. Using `torch.nn.Module.to()` directly bypasses this check. Additionally, Q8_0 dequantization internally uses float16 for its scale factors, so some weights come out as float16 even when bfloat16 was requested — requiring an explicit cast after dequantizing.

## Fix
File: `tt_forge_models/flux_gguf/pytorch/loader.py`

Replaced the guard-blocked `self.transformer.to(dtype_override)` with:

```python
self.transformer = self.transformer.dequantize()
torch.nn.Module.to(self.transformer, compute_dtype)
```

This converts all `GGUFLinear` layers to plain `nn.Linear` with regular tensor weights, then casts all parameters to `compute_dtype` (bfloat16) via the base Module's `.to()` method to bypass the diffusers quantization guard.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    518 s (8:38)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/flux_gguf/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4fb064e1aef4baa8a00d5c0efb33e2b7e2f58e5e |
| tt-forge-models | 5e450d8c50ccc3485505758ec460c205c67c9ee0 |
