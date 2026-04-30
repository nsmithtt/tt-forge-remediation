# Remediation Summary: joycaption-pytorch-Alpha_Two_HF_LLaVA_FP8_Dynamic-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[joycaption/pytorch-Alpha_Two_HF_LLaVA_FP8_Dynamic-single_device-inference]

## Result
FAIL — `compressed_tensors` FP8 quantized forward triggers nested `TorchFunctionMode` compilation; `fused_0.xla_args` never set, `extract_graph_helper` raises AttributeError

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
compressed-tensors-fp8-torch-function-mode-nested-compile

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
AttributeError: 'fused_0' object has no attribute 'xla_args'
```

Full traceback (key frames):
```
venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1964: in __getattr__
    raise AttributeError(
  File "venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:348"
      xla_args = xla_model.xla_args
  File "venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:513"
      → extract_internal(fused_module)
  File "venv/lib/python3.12/site-packages/torch_xla/_dynamo/dynamo_bridge.py:804"
      → partition_fx_graph_for_cpu_fallback
  File "python_package/tt_torch/backend/backend.py:225"
      → _call_experimental_compile → bridge.extract_compiled_graph
  File "venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py:1044"
      _fn (dynamo compiled function)
  File "python_package/tt_torch/torch_overrides.py:9"
      TorchFunctionOverride.__torch_function__
  File "venv/lib/python3.12/site-packages/compressed_tensors/quantization/lifecycle/forward.py:273"
      weight_data = weight.data   ← inside quantized_forward (monkey-patched nn.Linear.forward)
  File "venv/lib/python3.12/site-packages/transformers/models/siglip/modeling_siglip.py:288"
      → SigLIP linear layer
  File "venv/lib/python3.12/site-packages/transformers/models/llava/modeling_llava.py:370"
      → LlavaForConditionalGeneration forward
```

The `minicpm_*` loader `patched_getattr` frames in the traceback are incidental — those loaders globally monkey-patch `nn.Module.__getattr__`, so they appear as the chain that propagates the `AttributeError`.

## Root cause

The model `JKCHSTR/llama-joycaption-alpha-two-hf-llava-FP8-Dynamic` is loaded via `compressed_tensors`, which after loading calls `set_forward_quantized()` on every `nn.Linear`. This replaces each module's `forward` with `quantized_forward`, a closure that:
1. Accesses `weight.data` on the (XLA lazy) weight tensor.
2. Calls `fake_quantize` / `forward_quantize`, invoking `torch.clamp`, `torch.round`, and related ops.

`tt_torch/torch_overrides.py` installs a global `TorchFunctionMode` (`TorchFunctionOverride`) at import time. This mode intercepts **all** torch function calls. When `quantized_forward` runs inside a `torch.compile("tt")` traced forward pass, the torch ops inside it (or the `.data` property access itself on the XLA lazy tensor) go through `TorchFunctionOverride.__torch_function__`, which re-enters the TT backend compilation path (`eval_frame._fn → backend.py:_call_experimental_compile → bridge.extract_compiled_graph`).

This nested compilation invokes `partition_fx_graph_for_cpu_fallback`, which creates a `fused_0` GraphModule. `InputCollector` should set `fused_0.xla_args` by running `call_module("fused_0", ...)` through the interpreter — but in this nested context, the interpreter does not reach that call before `extract_internal(fused_0)` is called, leaving `fused_0.xla_args` unset. `extract_graph_helper` then raises `AttributeError`.

The root cause is in `tt-xla`: the `TorchFunctionMode` global interceptor is not guarded against re-entry when the torch ops it intercepts originate from inside a patched-forward (like `compressed_tensors.quantized_forward`) that itself runs within a `torch.compile` tracing context.

## Fix

The fix lives in `tt-xla` (`python_package/tt_torch/torch_overrides.py` or `python_package/tt_torch/backend/backend.py`).

**Proposed approach**: In `TorchFunctionOverride.__torch_function__`, check `torch._dynamo.is_compiling()` before re-entering the TT backend. If we are already inside a dynamo compile context, skip the TT interception and let the call pass through unmodified:

```python
def __torch_function__(self, func, types, args=(), kwargs=None):
    if torch._dynamo.is_compiling():
        return func(*args, **(kwargs or {}))
    # existing dispatch logic ...
```

This would prevent the nested compilation attempt and allow `quantized_forward` to run its ops without re-entering the compilation pipeline. However, this change affects the behavior of `TorchFunctionMode` for all quantized models; regression testing (beyond this one test) would be needed before merging.

## Tier B justification
**cross-cutting**: Changing `TorchFunctionOverride` to guard against nested compilation is a cross-cutting change — it affects how the global `TorchFunctionMode` handles every torch function call during any `torch.compile("tt")` forward pass, not just this model. The fix would need to be validated across all models that use the TT backend to ensure no regressions. Additionally, the exact mechanism by which `weight.data` on an XLA lazy tensor routes through `TorchFunctionMode` is not fully characterized; diagnosis must precede a production fix.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    120.17s (0:02:00) to AttributeError
- Tier A attempts: N/A

## Files changed

**tt-xla** (`remediation/joycaption-pytorch-Alpha_Two_HF_LLaVA_FP8_Dynamic-single_device-inference`):
- `tests/runner/utils/dynamic_loader.py` — removed `sys.path.insert(0, models_root)` that caused `tt_forge_models/spacy/` to shadow the real `spacy` package as a namespace package, breaking `datasets._dill`

**tt-forge-models** (`remediation/joycaption-pytorch-Alpha_Two_HF_LLaVA_FP8_Dynamic-single_device-inference`):
- `joycaption/pytorch/requirements.txt` (new) — added `compressed-tensors` dependency required by FP8 Dynamic quantized model variant
- `joycaption/pytorch/loader.py` — fixed `apply_chat_template` content format: changed from `[{"type": "image"}, {"type": "text", ...}]` list to plain string (`self.sample_text`); the Alpha Two Jinja template calls `.replace()` on `message['content']` and requires a string
- `joycaption/pytorch/loader.py` — added `use_fast=False` to `AutoProcessor.from_pretrained()`; transformers 5.x switched `SiglipImageProcessor` default to fast mode which uses lanczos interpolation via torchvision, but `F.interpolate` does not support lanczos

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 07136523bcd4d36bd483d7241a1b5560fe098387 |
| tt-forge-models | fb061234f3079a3388116170df77a757bd88cc1b |
