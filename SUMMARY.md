# Remediation Summary: deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf-causal_lm-pytorch-DISTILL_QWEN_7B_GSPO_BASIC_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf/causal_lm/pytorch-DISTILL_QWEN_7B_GSPO_BASIC_I1_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: YES — CPU BF16 vs FP32-CPU PCC = 0.9540; TT silicon PCC = 0.9538 (TT matches BF16 exactly; gap is Q4_K_M dequantization into BF16 weight rounding)
- Warning / exception suppression: NO

## Failure
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```
(surfaced in test collection as `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` because 26 other loaders patch the global `load_gguf_checkpoint` during collection, and the patched version rejects transformers 5.x's new `model_to_load` kwarg)

After fixing the TypeError, a second failure appeared:
```
AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9537842922881047. Required: pcc=0.99
```

## Root cause
Two issues, both in the loader layer:

**Issue 1 — model_to_load kwarg rejection**: 26 loaders in `tt_forge_models/*/causal_lm/pytorch/loader.py` monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a wrapper whose signature is `(gguf_path, return_tensors=False)`. Transformers 5.x changed `PreTrainedModel.from_pretrained` to call `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which the patched wrapper rejects with a TypeError. Because `test_models.py` imports all loaders during collection (`TorchDynamicLoader.setup_test_discovery()`), the globally-patched function breaks any subsequent GGUF test, including this one.

**Issue 2 — PCC floor from Q4_K_M + BF16**: The test's reference comparison uses FP32, but the GGUF checkpoint's INT4 weights are dequantized into BF16 before computation. Measured CPU BF16 vs CPU FP32 PCC = 0.9540. TT silicon PCC = 0.9538, which is essentially identical — the compiler produces the correct BF16 result. The 0.99 default required_pcc is unachievable for this quantization class; 0.95 is the correct floor.

## Fix
**Fix 1 — tt_forge_models submodule** (`remediation/deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf-causal_lm-pytorch-single_device-inference`, commit `fff9177603`):
Changed 26 loader files from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
so the wrapper transparently forwards all arguments (including `model_to_load`) to the original function.

**Fix 2 — tt-xla test config** (`remediation/deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf-causal_lm-pytorch-single_device-inference`, commit `1da282d62`):
Added entry to `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:
```yaml
deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf/causal_lm/pytorch-DISTILL_QWEN_7B_GSPO_BASIC_I1_Q4_K_M_GGUF-single_device-inference:
  status: EXPECTED_PASSING
  required_pcc: 0.95 # CPU BF16 vs FP32 PCC = 0.954; Q4_K_M quantization + BF16 weight rounding sets the floor below 0.99
```

## Verification
- pytest exit: PASS
- Hardware: n300
- Duration: 381.02s (0:06:21)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/*/causal_lm/pytorch/loader.py` (26 files) — `*args, **kwargs` passthrough fix
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — added `required_pcc: 0.95`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1da282d62bf4e6d1372ea1d84cb63bde622f8562 |
| tt-forge-models | fff91776038268e57bf259e34ca4b2530ba5f3dd |
