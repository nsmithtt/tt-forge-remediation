# Remediation Summary: einstein_bnb_4bit-causal_lm-pytorch-v6.1_Llama3_8B_BnB_4bit-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[einstein_bnb_4bit/causal_lm/pytorch-v6.1_Llama3_8B_BnB_4bit-single_device-inference]

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
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

After adding bitsandbytes:

```
RuntimeError: Creating a Parameter from an instance of type Params4bit requires that detach() returns an instance of the same type, but return type Tensor was found instead. To use the type as a Parameter, please correct the detach() semantics defined by its __torch_dispatch__() implementation.
```

## Root cause
Two loader bugs:

1. `bitsandbytes>=0.46.1` was missing from `requirements.txt`. The `PrunaAI/Einstein-v6.1-Llama3-8B-bnb-4bit-smashed` model requires bitsandbytes for loading its 4-bit quantized weights.

2. After loading, the test runner calls `model.to(xla_device)` to move it to TT hardware. `torch.nn.Module._apply()` calls `param.detach()` on every parameter and then wraps the result in `nn.Parameter`. However, bitsandbytes `Params4bit.detach()` returns a plain `Tensor` (not a `Params4bit`), so `Parameter.__new__` raises a `RuntimeError` because the type invariant is violated. TT hardware has no bitsandbytes/CUDA 4-bit dequantization kernels, so the model must be dequantized to bfloat16 before device transfer.

Additionally, `padding="max_length"` in `load_inputs` is a known pattern that pads short inputs to a fixed length, which degrades PCC on TT hardware; removed in the same fix.

## Fix
In `tt_forge_models`, on branch `remediation/einstein_bnb_4bit-causal_lm-pytorch-v6.1_Llama3_8B_BnB_4bit-single_device-inference`:

1. **`einstein_bnb_4bit/causal_lm/pytorch/requirements.txt`** (new file): Added `bitsandbytes>=0.46.1`.

2. **`einstein_bnb_4bit/causal_lm/pytorch/loader.py`**: Added `_dequantize_bnb4_to_bf16()` which iterates over all `bnb.nn.Linear4bit` modules, dequantizes each with `bitsandbytes.functional.dequantize_4bit`, and replaces with standard `nn.Linear` holding bfloat16 weights. Called immediately after `from_pretrained`. The bitsandbytes import is lazy (inside the function body) so the loader module can be imported during test collection before bitsandbytes is installed. Also removed `padding="max_length"` from `load_inputs`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    162.79s
- Tier A attempts: N/A

## Files changed
- `einstein_bnb_4bit/causal_lm/pytorch/requirements.txt` (new)
- `einstein_bnb_4bit/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9ca9599e0b8315f8b36bb8b5cb76a3b5122f3b23 |
| tt-forge-models | df35665a4128bb0719f6f6eaa55d9516ae46646b |
