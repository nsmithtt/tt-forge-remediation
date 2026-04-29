# Remediation Summary: deepseek_r1_distill_llama_8b_w8a8-causal_lm-pytorch-Distill_Llama_8B_W8A8-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_llama_8b_w8a8/causal_lm/pytorch-Distill_Llama_8B_W8A8-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
compressed-tensors-missing-requirement-and-run-compressed-not-disabled

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ImportError: compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors`.

## Root cause
The `deepseek_r1_distill_llama_8b_w8a8` model loader was missing `compressed-tensors` in its `requirements.txt`. `RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8` uses the Neural Magic compressed-tensors W8A8 quantization format, which requires the `compressed-tensors` library at model-load time. Without it, `transformers` raises `ImportError` during `AutoModelForCausalLM.from_pretrained`.

Additionally, without `run_compressed=False` in the quantization config, compressed-tensors would keep weights in int8 and attach instance-level `forward` overrides to each quantized `Linear`. TT hardware has no int8 matmul path via this route; the int8 weights would reach the compiler and fail. The instance-level `forward` overrides also conflict with TT-XLA's `__torch_function__` during `torch.compile`.

## Fix
All changes in `tt-xla/third_party/tt_forge_models` on branch `remediation/deepseek_r1_distill_llama_8b_w8a8-causal_lm-pytorch-Distill_Llama_8B_W8A8-single_device-inference`:

1. Added `deepseek_r1_distill_llama_8b_w8a8/causal_lm/pytorch/requirements.txt` with `compressed-tensors`.

2. Updated `deepseek_r1_distill_llama_8b_w8a8/causal_lm/pytorch/loader.py`:
   - Always load `AutoConfig` and set `config.quantization_config["run_compressed"] = False` before `from_pretrained`, so int8 W8A8 weights are dequantized to bfloat16 at load time.
   - After loading, remove instance-level `forward` overrides that compressed-tensors 0.15.x attaches to quantized `Linear` modules (they access `weight.data` unconditionally, conflicting with TT-XLA's `__torch_function__`).
   - Set `model.config.use_cache = False` to suppress KV-cache outputs from the decoder model.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    102.46s (0:01:42)
- Tier A attempts: N/A

## Files changed
- `deepseek_r1_distill_llama_8b_w8a8/causal_lm/pytorch/requirements.txt` (new)
- `deepseek_r1_distill_llama_8b_w8a8/causal_lm/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | a66f69b4b6b103ad5baf91dea0043c616cd63805 |
| tt-forge-models | 850be078e432efda759d59365d15df660ff72856 |
