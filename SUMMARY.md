# Remediation Summary: bartowski_openai_gpt_oss_120b_gguf-causal_lm-pytorch-BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_openai_gpt_oss_120b_gguf/causal_lm/pytorch-BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF-single_device-inference]

## Result
XFAIL — 120B model in MXFP4 format (60 GB weights) exceeds 12 GB n150 HBM; loader fix applied separately

## Stack layer
loader, hardware-class

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original CI failure:
```
raise ImportError(Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.)
```

Locally reproduced as:
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

Both are caused by the same loader bug. The ImportError on CI occurred because `gguf` was not installed; locally gguf is available so execution reaches the monkey-patch chain where `model_to_load` is rejected.

## Root cause

**Loader layer — broken chain traversal in `_find_real_load_gguf_checkpoint()`.**

transformers 5.2 added a `model_to_load` parameter to `load_gguf_checkpoint()`. Multiple other loaders in `tt_forge_models` monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` with stale wrappers that only accept `(gguf_path, return_tensors=False)`. These stale patches are applied at import time in alphabetical loader discovery order, so the outermost patch that `from_pretrained` sees inside `modeling_utils.py` does not accept `model_to_load`.

The remediation branch's previous fix attempted to traverse the patch chain to find the original transformers function and re-apply a forward-compatible wrapper. However, the chain-traversal code searched globals by VALUE `__name__`, but stale patches stored under keys like `_orig_load_gguf_checkpoint` have `__name__ = "_patched_load_gguf_checkpoint"` — "load_gguf" does not appear in that name. As a result, the traversal never followed the chain and `_REAL_LOAD_GGUF_CHECKPOINT` resolved to the outermost stale patch, causing the same TypeError.

**Secondary — hardware capacity ceiling.**

The model (bartowski/openai_gpt-oss-120b-GGUF, MXFP4 format) has 120 billion parameters. MXFP4 encoding is 4 bits per weight, giving approximately 60 GB of weight data. The target hardware (n150) provides 12 GB of HBM. Even in the quantized MXFP4 representation the model is 5× too large to fit on a single n150 device. The test is correctly classified as KNOWN_FAILURE_XFAIL; it requires a multi-chip or Galaxy configuration.

## Fix

**Loader fix (tt_forge_models remediation branch):**

`bartowski_openai_gpt_oss_120b_gguf/causal_lm/pytorch/loader.py`

Changed `_find_real_load_gguf_checkpoint()` to search globals by **key name** (in addition to value `__name__`). The key check `"load_gguf" in key` matches entries like `_orig_load_gguf_checkpoint` that store the previous generation of the patch. Also updated the closure search to use `__code__.co_freevars` for variable names and match by variable name (e.g. `orig_load` contains "load") in addition to `__name__`.

Commit: `58b1f7395e2dded5d153250e58860eb7821f56df` on `remediation/bartowski_openai_gpt_oss_120b_gguf-causal_lm-pytorch-BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF-single_device-inference` (tt_forge_models)

**Test config (tt-xla):**

Added `KNOWN_FAILURE_XFAIL` entry for this test in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

Commit: `8c7c5807a8b9fc9bc0f08a9d1c3ad3747902824b` on `remediation/bartowski_openai_gpt_oss_120b_gguf-causal_lm-pytorch-BARTOWSKI_OPENAI_GPT_OSS_120B_MXFP4_MOE_GGUF-single_device-inference` (tt-xla)

## Verification
- pytest exit: FAIL (hardware capacity — model does not fit in 12 GB HBM)
- Hardware: n150
- Duration: ~40 min (model loaded from GGUF, crashed in dynamo_bridge during compilation attempt)
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/bartowski_openai_gpt_oss_120b_gguf/causal_lm/pytorch/loader.py` — fix chain traversal in `_find_real_load_gguf_checkpoint()`
- `tt-xla/tests/runner/test_config/torch/test_config_inference_single_device.yaml` — add KNOWN_FAILURE_XFAIL entry

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 8c7c5807a8b9fc9bc0f08a9d1c3ad3747902824b |
| tt-forge-models | 58b1f7395e2dded5d153250e58860eb7821f56df |
