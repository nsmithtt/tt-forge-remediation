# Remediation Summary: lmstudio_wizardlm_2_7b_gguf-causal_lm-pytorch-7B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch-7B_Q4_K_M_GGUF-single_device-inference]

## Result
FAIL — after adding requirements.txt (gguf>=0.10.0), test still fails with TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load' from 25 other loaders that monkey-patch load_gguf_checkpoint globally with a fixed signature

## Stack layer
loader

## Tier
B

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
Original reported failure:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Failure after requirements.txt fix (reproduced during this session):
```
third_party/tt_forge_models/lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch/loader.py:99: in load_model
    model = AutoModelForCausalLM.from_pretrained(
venv/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py:374: in from_pretrained
    return model_class.from_pretrained(
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4016: in from_pretrained
    state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
Two distinct loader-layer bugs:

**Bug 1 (fixed):** No `requirements.txt` beside the loader. The `RequirementsManager` in the test session's conftest captures a golden pip state at session start, installs each model's `requirements.txt` before the test, and rolls back afterward. Without a `requirements.txt`, `gguf` is absent when a prior test already uninstalled it — triggering the `ImportError`.

**Bug 2 (Tier B, unfixed):** During pytest collection, 25 other GGUF loaders (primarily `tvall43_*`, `mradermacher_qwen3_5_*`, `gpt_oss_swallow_*`, `qwen_3_5_imatrix_gguf`, `dmind_3_mini_i1_gguf`, `daniloreddy_qwen3_5_0_8b_gguf`, `bartowski_coniccat_qwen3_5_27b_writer_gguf`) monkey-patch `transformers.gguf_utils.load_gguf_checkpoint` at module import time with a fixed signature:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
```
Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` — a keyword argument not accepted by the fixed-signature patch — raising `TypeError`. The last loader to apply its patch wins, and any of the 25 broken loaders can be the one that corrupts the global.

## Fix
**Bug 1 (applied):**
- Created `lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0`
- Committed to `tenstorrent/tt-forge-models` on `remediation/lmstudio_wizardlm_2_7b_gguf-causal_lm-pytorch-7B_Q4_K_M_GGUF-single_device-inference`
- Updated `tt-xla` submodule pointer on `remediation/lmstudio_wizardlm_2_7b_gguf-causal_lm-pytorch-7B_Q4_K_M_GGUF-single_device-inference`

**Bug 2 (proposed fix):**
Each of the 25 broken loaders needs its `_patched_load_gguf_checkpoint` signature changed from the fixed form:
```python
def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):
    result = _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to a passthrough form:
```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    result = _orig_load_gguf_checkpoint(*args, **kwargs)
```
This fix is mechanically identical across all 25 files but must be applied to each.

## Tier B justification
more-than-3-files

25 loader files each independently define `_patched_load_gguf_checkpoint` with the same broken fixed signature; all 25 need the same mechanical one-line fix, but the change count exceeds the Tier A threshold.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    88.05s (until TypeError)
- Tier A attempts: N/A

## Files changed
- `lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch/requirements.txt` (new, in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c80b0c160b81a13a79bec3eab08d8ddf137158d0 |
| tt-forge-models | b76b23656f7cc2d2b6d6ac3a414a5072179eac39 |
