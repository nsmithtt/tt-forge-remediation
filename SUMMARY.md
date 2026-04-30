# Remediation Summary: gemma_2_bnb-causal_lm-pytorch-2_9B_IT_BNB_4BIT-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma_2_bnb/causal_lm/pytorch-2_9B_IT_BNB_4BIT-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
bnb-4bit-missing-requirements-and-dequantize, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`

## Root cause
Two bugs:

1. **Loader**: The gemma_2_bnb loader had no `requirements.txt`, so `bitsandbytes` was never installed. transformers' `quantizer_bnb_4bit.py:validate_environment` raises `ImportError` before any weights load. Additionally, once bitsandbytes is installed, the `Linear4bit` layers in the loaded model are incompatible with TT XLA (no CUDA kernels), so the model must be dequantized to standard `nn.Linear` (bfloat16) layers before inference.

2. **tt-xla (TorchFunctionOverride)**: After fixing the loader, `SlidingWindowCache.update()` in `transformers/cache_utils.py:214` does `full_value_states[:, :, -sliding_window+1:, :]` where `sliding_window=4096` but `seq_len=16`. This produces `start=-4095`, which is out of bounds for a 16-element dimension. The XLA lazy backend raises `RuntimeError: Value out of range (expected to be in range of [-16, 15], but got -4095)` instead of clamping like PyTorch eager. Fixed by pre-clamping in `TorchFunctionOverride.__torch_function__` before the slice reaches the XLA backend.

## Fix
**Loader** (`tt-xla/third_party/tt_forge_models`, branch `remediation/gemma_2_bnb-causal_lm-pytorch-2_9B_IT_BNB_4BIT-single_device-inference`):
- `gemma_2_bnb/causal_lm/pytorch/requirements.txt`: added `bitsandbytes>=0.46.1`
- `gemma_2_bnb/causal_lm/pytorch/loader.py`: added `_dequantize_bnb4_to_bf16()` function that iterates all `bnb.nn.Linear4bit` modules and replaces them with standard `nn.Linear` with dequantized bfloat16 weights; called after `AutoModelForCausalLM.from_pretrained()`

**tt-xla** (branch `remediation/gemma_2_bnb-causal_lm-pytorch-2_9B_IT_BNB_4BIT-single_device-inference`):
- `python_package/tt_torch/torch_overrides.py`: in `TorchFunctionOverride.__torch_function__`, added a guard for `func is torch.ops.aten.slice.Tensor` that clamps `start` and `end` to `max(val, -size)` when the tensor dimension is statically known

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    221.26s (0:03:41)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma_2_bnb/causal_lm/pytorch/requirements.txt` (new)
- `tt-xla/third_party/tt_forge_models/gemma_2_bnb/causal_lm/pytorch/loader.py`
- `tt-xla/python_package/tt_torch/torch_overrides.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a546b05ee30bfc0f6959cafa19960cd7a6dbbe8b |
| tt-forge-models | 609662486cbce0fe53afe0534ddccdb07f79fad0 |
