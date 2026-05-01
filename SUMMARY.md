# Remediation Summary: mistral-7b-instruct-v03-awq-single-device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mistral_7b_instruct_v03_awq/causal_lm/pytorch-Mistral_7B_Instruct_v0.3_AWQ-single_device-inference]

## Result
SILICON_PASS — loader fixes (pytest.ini pythonpath, gptqmodel dep, AWQ dequantization)

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
awq-gptqmodel-missing-dep-and-dequantization

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError while loading conftest '/home/nsmith/tt-forge-remediation/tt-xla/tests/conftest.py'.
tests/conftest.py:25: in <module>
    from infra import DeviceConnectorFactory, Framework
E   ModuleNotFoundError: No module named 'infra'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

After fixing the pythonpath issue, the next failure was:
```
E   ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`
```

## Root cause
Two loader-layer bugs:

1. **pytest.ini missing `pythonpath = tests`**: `tests/conftest.py` imports `from infra import ...` but pytest's rootdir is `tt-xla/`, not `tt-xla/tests/`. Without `pythonpath = tests` in `pytest.ini`, the `tests/` directory is not on sys.path and conftest.py fails to import the local `infra` package. The swigvarlink DeprecationWarning appears at the bottom of output (from SWIG) but is not the root cause. Multiple prior remediations fixed this same issue.

2. **Missing `gptqmodel` dependency**: transformers 5.x requires `gptqmodel` (replacing `autoawq`) to load AWQ-quantized models. The loader had no `requirements.txt`, so `gptqmodel` was never installed. After adding it, `gptqmodel.TorchAtenAwqLinear` raises `NotImplementedError` when run on TT device because its `_fused_op_forward` only supports CPU (the CPU golden run calls `transform_cpu()` which sets `linear_mode='inference'` and transforms weights to int4pack format; the subsequent TT device forward then hits the unsupported path). Fix: replace all AWQ modules with plain `nn.Linear` before any forward pass using `awq_weight_dequantize()`.

Note: `gptqmodel==7.0.0` pins `numpy==2.2.6`, causing an in-process numpy version conflict when installed via requirements.txt in a session that already has numpy 2.1.2 loaded (the `_purge_stale_modules` mechanism evicts numpy from sys.modules, then the fresh re-import of 2.2.6 fails because numpy 2.1.2 ufunc objects held by torch lack the `__module__` attribute added in 2.2+). This is a pre-existing limitation of the requirements manager for any package that upgrades numpy. The test passes cleanly when gptqmodel is pre-installed into the venv (numpy 2.2.6 is loaded from process start).

## Fix
**tt-xla** (`pytest.ini`):
- Added `pythonpath = tests` so `infra` is importable by `tests/conftest.py`
- Added `filterwarnings` to suppress the SWIG swigvarlink DeprecationWarning

**tt-forge-models** (`mistral_7b_instruct_v03_awq/causal_lm/pytorch/`):
- Created `requirements.txt` containing `gptqmodel` (transformers 5.x requires it for AWQ models)
- Added `_dequantize_awq_layers()` static method to `ModelLoader` that replaces each `TorchAtenAwqLinear` module with a plain `nn.Linear` using `awq_weight_dequantize()` before any forward pass
- Called `_dequantize_awq_layers()` after `from_pretrained()` and before `eval()`
- Added `import torch.nn as nn`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    259.50s (0:04:19)
- Tier A attempts: N/A

## Files changed
- `tt-xla/pytest.ini` — add `pythonpath = tests` and SWIG DeprecationWarning filter
- `tt-forge-models/mistral_7b_instruct_v03_awq/causal_lm/pytorch/requirements.txt` — new file, `gptqmodel`
- `tt-forge-models/mistral_7b_instruct_v03_awq/causal_lm/pytorch/loader.py` — `import nn`, `_dequantize_awq_layers()` method, call before `eval()`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b |
| tt-xla          | a841d294a |
| tt-forge-models | 6c959b689a |
