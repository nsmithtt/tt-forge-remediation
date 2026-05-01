# Remediation Summary: meta_llama_3_1_8b_instruct_quantized_w8a16-causal_lm-pytorch-Meta_Llama_3_1_8B_Instruct_Quantized_W8A16-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[meta_llama_3_1_8b_instruct_quantized_w8a16/causal_lm/pytorch-Meta_Llama_3_1_8B_Instruct_Quantized_W8A16-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
compressed-tensors-missing-dep-and-w8a16-dequant

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`.
```

After installing compressed-tensors, a second error appeared:
```
ValueError: Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
```

## Root cause
Three loader bugs:

1. **Missing dependency**: `compressed-tensors` was not listed in the model's `requirements.txt`. The model `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16` uses compressed-tensors W8A16 quantization format; transformers 5.x raises `ImportError` during `from_pretrained` if the package is absent.

2. **Missing pad_token**: The Llama tokenizer has no pad token by default. The loader calls the tokenizer with `padding="max_length"`, which requires a pad token. Fix: set `tokenizer.pad_token = tokenizer.eos_token` after loading.

3. **INT8 weights not dequantized**: compressed-tensors 0.15.x keeps weights as INT8 by default (`run_compressed=True`). TT silicon does not support the compressed-tensors INT8 matmul path. Setting `run_compressed=False` in the quantization config before `from_pretrained` causes weights to be dequantized to bfloat16 during load. Additionally, compressed-tensors 0.15.x attaches an instance-level `quantized_forward` method to each quantized Linear module that accesses `weight.data` unconditionally; this conflicts with TT-XLA's `__torch_function__` during `torch.compile`. Removing the instance-level `forward` from each module's `__dict__` restores the class-level forward and fixes the conflict.

## Fix
All fixes are in `tt_forge_models/meta_llama_3_1_8b_instruct_quantized_w8a16/causal_lm/pytorch/`:

- `requirements.txt` (new file): add `compressed-tensors`
- `loader.py`:
  - In `_load_tokenizer`: set `self.tokenizer.pad_token = self.tokenizer.eos_token` when `pad_token is None`
  - In `load_model`: load `AutoConfig`, set `run_compressed=False` on the quantization config before calling `from_pretrained`, and after load remove any instance-level `forward` from each module

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    318.05s (0:05:18)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/meta_llama_3_1_8b_instruct_quantized_w8a16/causal_lm/pytorch/requirements.txt` (new)
- `tt_forge_models/meta_llama_3_1_8b_instruct_quantized_w8a16/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 131fa1404ef306a3c12635c22ca32fb1bd16ba12 |
| tt-forge-models | bbcf8001917525fef43930e791e2563e412c11ec |
