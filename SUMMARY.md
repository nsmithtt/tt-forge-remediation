# Remediation Summary: qwen_3_5_9b_base_text_nvfp4-causal_lm-pytorch-9B_Base_Text_NVFP4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[qwen_3_5_9b_base_text_nvfp4/causal_lm/pytorch-9B_Base_Text_NVFP4-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
nvfp4-modelopt-weights-not-dequantized

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Test exceeded configured timeout and was killed

## Root cause
Two loader bugs combined to cause failure:

1. **use_cache=False missing** (first fix): `Qwen3_5DynamicCache` is not a subclass of
   `transformers.Cache`. When `use_cache=True` (default), the model returns a
   `Qwen3_5DynamicCache` object in the outputs. The TorchComparisonEvaluator's
   `_cache_to_legacy()` check (`isinstance(tensor, Cache)`) does not recognize it, so
   `torch.equal()` is called on two cache objects, raising `TypeError` or hanging
   indefinitely (the original timeout failure).

2. **NVFP4 weights randomly initialized** (second fix, root cause of PCC=0.955):
   The `osoleve/Qwen3.5-9B-Base-Text-NVFP4` checkpoint stores all Linear weights as
   packed uint8 tensors in NVIDIA modelopt NVFP4 format (e2m1, 2 values per byte), with
   per-block `float8_e4m3fn` scales and a global `float32` double-scale. The shapes are
   halved on the K dimension (e.g. `[4096, 4096]` → `[4096, 2048]`). Since transformers
   5.x does not recognize `quant_method: "modelopt"`, it skips quantization handling
   entirely and `from_pretrained` reports all weight matrices as `MISMATCH` and
   randomly re-initializes them with `ignore_mismatched_sizes=True`. This causes
   PCC=0.955 (CPU vs TT) with random weights instead of the expected ≥0.99 with correct
   weights. After properly dequantizing the weights, the test passes with PCC ≥ 0.99.

## Fix
Two commits in `tt-forge-models` on branch
`remediation/qwen_3_5_9b_base_text_nvfp4-causal_lm-pytorch-9B_Base_Text_NVFP4-single_device-inference`:

1. `ab7d3aab2a` — `qwen_3_5_9b_base_text_nvfp4/causal_lm/pytorch/loader.py`: add
   `inputs["use_cache"] = False` in `load_inputs()` to prevent `Qwen3_5DynamicCache`
   from appearing in model outputs passed to the comparison evaluator.

2. `710b9c42bf` — `qwen_3_5_9b_base_text_nvfp4/causal_lm/pytorch/loader.py` and new
   `requirements.txt`: add `_dequantize_nvfp4_weights()` which uses
   `NVFP4QTensor.dequantize()` from `nvidia-modelopt` to unpack the e2m1 nibbles and
   apply the float8_e4m3fn per-block scales (with float32 global double-scale, group
   size 16) in-place on every affected parameter after `from_pretrained()`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    3323.84s (0:55:23)
- Tier A attempts: N/A

## Files changed
- `qwen_3_5_9b_base_text_nvfp4/causal_lm/pytorch/loader.py`
- `qwen_3_5_9b_base_text_nvfp4/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 710b9c42bf3cb425bfc27cba467796cbc4af90ad |
