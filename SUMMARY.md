# Remediation Summary: llama_3_2_3b_instruct_awq-causal_lm-pytorch-3.2_3B_Instruct_AWQ-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_2_3b_instruct_awq/causal_lm/pytorch-3.2_3B_Instruct_AWQ-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
awq-gptqmodel-transitive-dep-env-pollution

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`

(After adding plain `gptqmodel` to requirements.txt, the failure became:
AttributeError: 'numpy.ufunc' object has no attribute '__module__'
because gptqmodel >= 6.0.0 requires transformers >= 5.4.0 and numpy >= 2.2.6,
upgrading them in-process and breaking the test environment.)

## Root cause
Two loader bugs, both in the loader layer:

1. **Missing gptqmodel dependency**: transformers 5.x AWQ quantizer checks `is_gptqmodel_available()` and raises ImportError if gptqmodel is not installed. No `requirements.txt` existed in the loader directory.

2. **requirements.txt environment pollution**: Adding `gptqmodel` as a plain `requirements.txt` entry installs gptqmodel 7.0.0, which requires `transformers >= 5.4.0` and `numpy >= 2.2.6`. The test infrastructure installs these in-process, upgrading the environment from transformers 5.2.0 / numpy 2.1.2. When `tokenizer_class_from_name` then calls `importlib.import_module("transformers")` to re-import the newly installed transformers 5.7.0 into a process that already has 5.2.0 loaded, numpy 2.2.6's `_override___module__` raises `AttributeError: 'numpy.ufunc' object has no attribute '__module__'`.

3. **AWQ layer CPU-only kernel**: After loading, `TorchAtenAwqLinear._fused_op_forward` uses `torch.ops.aten._weight_int4pack_mm_for_cpu` and raises `NotImplementedError` when called with TT tensors (`x.device != 'cpu'`). The loader on the remediation branch already dequantizes all AWQ layers to `nn.Linear` via `awq_weight_dequantize()` to avoid this.

## Fix
Two changes in `llama_3_2_3b_instruct_awq/causal_lm/pytorch/` (tt-forge-models):

1. `requirements.txt`: Changed from `gptqmodel` to just a SPDX header comment. This keeps the requirements manager activated (so `requirements.nodeps.txt` is picked up) without installing any packages directly.

2. `requirements.nodeps.txt` (new file): Contains `gptqmodel`. The test infrastructure installs this with `--no-deps`, so gptqmodel 7.0.0 is installed without pulling in its conflicting transitive dependencies (transformers 5.7.0, numpy 2.2.6, etc.). gptqmodel is importable and `TorchAtenAwqLinear.awq_weight_dequantize()` works correctly without those deps.

The loader's `_dequantize_awq_layers()` function (already on the remediation branch from commit 2f50a848f1) remains unchanged — it detects `TorchAtenAwqLinear` modules by `awq_weight_dequantize` attribute and replaces them with `nn.Linear`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    130.92s (0:02:10)
- Tier A attempts: N/A

## Files changed
- `llama_3_2_3b_instruct_awq/causal_lm/pytorch/requirements.txt` (tt-forge-models, commit cd0b6a1a32)
- `llama_3_2_3b_instruct_awq/causal_lm/pytorch/requirements.nodeps.txt` (tt-forge-models, new file, commit cd0b6a1a32)
- `llama_3_2_3b_instruct_awq/causal_lm/pytorch/loader.py` (tt-forge-models, commit 2f50a848f1 — dequantize AWQ layers)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9de432d884dff6c432ddedcaa4337e47825fe612 |
| tt-forge-models | cd0b6a1a32ea5df566bc6b8f98a37cf58b1ce03d |
