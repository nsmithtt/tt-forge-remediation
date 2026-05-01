# Remediation Summary: lmstudio_wizardlm_2_7b_gguf-causal_lm-pytorch-7B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch-7B_Q4_K_M_GGUF-single_device-inference]

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
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Secondary failure (revealed after fix 1):
```
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
```

## Root cause
Two loader-layer bugs:

**Bug 1 — missing requirements.txt:** No `requirements.txt` beside the loader. The `RequirementsManager` captures a golden pip state at session start, installs each model's `requirements.txt` before its test, and rolls back afterward. Without `requirements.txt`, `gguf` is absent when a prior test uninstalled it — `is_gguf_available()` returns False and `from_pretrained` raises the ImportError.

**Bug 2 — global `load_gguf_checkpoint` patch with fixed signature:** During pytest collection all loader modules are imported. 25 loaders (primarily `tvall43_*`, `mradermacher_qwen3_5_*`, `gpt_oss_swallow_*`) define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` — a fixed signature — and install it as the global `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which the fixed-signature patch rejects with TypeError. The last importer in alphabetical collection order wins; any of the 25 broken loaders can corrupt the global.

The wizardlm loader does not itself define a patch, so fixing the other loaders would require touching 25+ files. The scoped fix — BFS-walk the patcher chain inside `load_model` to find and temporarily restore the real transformers function — is a 1-file change in this loader.

## Fix
Two changes in `lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch/`:

1. **`requirements.txt`** (new file): `gguf>=0.10.0` — ensures gguf is installed before each test run.

2. **`loader.py`** — added `_find_original_from_transformers()` + `_gguf_kwargs_compat()` context manager (BFS through `__globals__` and `__closure__` of the currently-installed patcher to recover the real transformers function, then temporarily restore it). `load_model()` wraps `AutoModelForCausalLM.from_pretrained` in `with _gguf_kwargs_compat():`.

Both changes committed to `tenstorrent/tt-forge-models` on `remediation/lmstudio_wizardlm_2_7b_gguf-causal_lm-pytorch-7B_Q4_K_M_GGUF-single_device-inference`. tt-xla submodule updated on same-named branch in `tenstorrent/tt-xla`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    299.51s (0:04:59)
- Tier A attempts: N/A

## Files changed
- `lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch/requirements.txt` (new)
- `lmstudio_wizardlm_2_7b_gguf/causal_lm/pytorch/loader.py` (added _gguf_kwargs_compat + context manager in load_model)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 4905a4c11f0195e8c44dd1702df055e160789d3a |
| tt-forge-models | 41f80f2f80e26b7216dc03c19bd6f6e20b473338 |
