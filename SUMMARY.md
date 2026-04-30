# Remediation Summary: dnd_gguf-causal_lm-pytorch-Qwen2.5_0.5B_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[dnd_gguf/causal_lm/pytorch-Qwen2.5_0.5B_GGUF-single_device-inference]

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
Original failure message:
```
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")
```

Reproduced locally (on branch e51150dc10) as:
```
third_party/tt_forge_models/zuzett_granite_4_0_h_tiny_imatrix_gguf/causal_lm/pytorch/loader.py:26: in _patched_load_gguf_checkpoint
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
                                      ^^^^^^^^^
NameError: name 'gguf_path' is not defined
```

## Root cause
Two loader-layer bugs blocked the test:

1. **Incomplete `_patched_load_gguf_checkpoint` body in 5 loaders** (the proximate cause
   of the reproduce failure): Commit e51150dc10 changed the function signature from
   `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` in 32 loaders, but
   accidentally left the body still calling
   `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` in 5 of those
   files. When any test triggers the global `load_gguf_checkpoint` chain that flows
   through one of these loaders, the function raises `NameError: name 'gguf_path' is
   not defined` because `gguf_path` is no longer a named parameter. The affected loaders
   are: `hearthfire_gguf`, `nvidia_llama_3_1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_gguf`,
   `llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf`,
   `steampunque_qwen3_5_27b_mp_gguf`, `zuzett_granite_4_0_h_tiny_imatrix_gguf`.

2. **Missing `requirements.txt` in `dnd_gguf/causal_lm/pytorch`** (the originally reported
   ImportError): The `dnd_gguf` loader calls `AutoModelForCausalLM.from_pretrained` with
   `gguf_file=...`, which requires the `gguf>=0.10.0` package. Without `requirements.txt`,
   the `RequirementsManager` has nothing to install, so if `gguf` is not in the base
   environment the test fails with an `ImportError` before reaching device execution.

## Fix

**Fix 1** (tt_forge_models, commit 05579db0da): For all 5 partially-fixed loaders, change
the body from:
```python
return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```
to:
```python
return _orig_load_gguf_checkpoint(*args, **kwargs)
```

**Fix 2** (tt_forge_models, commit be4af20f9c): Create
`dnd_gguf/causal_lm/pytorch/requirements.txt` containing `gguf>=0.10.0`.

Both fixes land on the remediation branch of tt_forge_models and are reflected in a new
commit in the tt-xla remediation branch that updates the submodule pointer.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    265.96s (0:04:25)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/hearthfire_gguf/causal_lm/pytorch/loader.py` — fix _patched_load_gguf_checkpoint body
- `tt_forge_models/nvidia_llama_3_1_nemotron_8b_ultralong_1m_instruct_abliterated_i1_gguf/causal_lm/pytorch/loader.py` — fix _patched_load_gguf_checkpoint body
- `tt_forge_models/llama_3_2_3b_instruct_heretic_ablitered_uncensored_i1_gguf/causal_lm/pytorch/loader.py` — fix _patched_load_gguf_checkpoint body
- `tt_forge_models/steampunque_qwen3_5_27b_mp_gguf/causal_lm/pytorch/loader.py` — fix _patched_load_gguf_checkpoint body
- `tt_forge_models/zuzett_granite_4_0_h_tiny_imatrix_gguf/causal_lm/pytorch/loader.py` — fix _patched_load_gguf_checkpoint body
- `tt_forge_models/dnd_gguf/causal_lm/pytorch/requirements.txt` — add gguf>=0.10.0

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 063839652ba370aefee19e92addbaadadbefcf9c |
| tt-forge-models | be4af20f9c23d7a7c91dba09e5af43fa7d3dd69d |
