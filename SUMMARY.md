# Remediation Summary: bartowski_qwen2_5_coder_3b_instruct_gguf-causal_lm-pytorch-Qwen2_5_Coder_3B_Instruct_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_qwen2_5_coder_3b_instruct_gguf/causal_lm/pytorch-Qwen2_5_Coder_3B_Instruct_GGUF-single_device-inference]

## Result
SILICON_PASS — loader bug fixed: _patched_load_gguf_checkpoint missing model_to_load kwarg

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
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(Note: the originally reported failure message was `raise InvalidVersion(f"Invalid version: {version!r}")` but the actual reproduced error was the TypeError above, caused by the same underlying loader issue.)

## Root cause
During pytest collection, `setup_test_discovery` imports all model loader modules to build the parametrize list. Several Qwen3.5 GGUF loaders (28 total) monkey-patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at module import time to inject architecture aliases (`qwen35`). These patched versions had a narrow signature:

```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
```

Transformers 5.2.0 added a third parameter `model_to_load` to `load_gguf_checkpoint` and calls it as:
```python
load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)
```

Because the monkey-patch replaces the module-level attribute on `transformers.modeling_gguf_pytorch_utils`, and `modeling_utils.py` does a fresh `from .modeling_gguf_pytorch_utils import load_gguf_checkpoint` on each `from_pretrained` call, it picks up the patched (narrow) version. The bartowski Qwen2.5-Coder-3B-Instruct GGUF loader itself does not apply any patch, but it is affected by patches applied by other loaders imported earlier during test collection.

## Fix
Cherry-picked two commits from the existing fix branch onto a new remediation branch in `tt-forge-models`:

- `7643c93cc7` — Add gguf>=0.10.0 requirement for mradermacher_qwen3_5_4b_abliterated_gguf
- `eaee402cd2` — Fix _patched_load_gguf_checkpoint to forward model_to_load kwarg

The fix changes all 26 affected loaders from:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False, model_to_load=None):
    result = _orig_load_gguf_checkpoint(
        gguf_path, return_tensors=return_tensors, model_to_load=model_to_load
    )
```

Files changed: 26 loader.py files under `third_party/tt_forge_models/` (all Qwen3.5 GGUF loaders with the monkey-patch pattern).

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    314.63s (0:05:14)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models`: 26 GGUF loader files updated (cherry-picked eaee402cd2)
- `tt-xla`: third_party/tt_forge_models submodule pointer updated

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c458eedf8919894244f2a7383e955e3529b45281 |
| tt-forge-models | 827611aa29e9fa9ef0956fe152f4db4644ec9dd5 |
