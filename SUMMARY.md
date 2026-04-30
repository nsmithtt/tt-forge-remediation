# Remediation Summary: gemma3_270m_gguf-causal_lm-pytorch-270M_IT_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_270m_gguf/causal_lm/pytorch-270M_IT_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
N/A

## Bug fingerprint
aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)

While executing %slice_6 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat_21, 2, -511, 9223372036854775807), kwargs = {})

Original traceback:
  File "transformers/cache_utils.py", line 214, in update
    self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

## Root cause
Two bugs in sequence:

1. **Loader (tt_forge_models):** Several GGUF loaders (26 files) monkey-patch
   `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time
   with a narrow signature `(gguf_path, return_tensors=False)`. Transformers 5.2.0
   added a `model_to_load=` keyword argument; during pytest collection all loaders
   are imported, so the patched function with no `**kwargs` raises `TypeError` when
   the Gemma3 GGUF is loaded.

2. **tt-xla (TorchFunctionOverride):** Gemma3's `SlidingWindowCache.update()` calls
   `full_value_states[:, :, -sliding_window + 1:, :]` where `sliding_window=512`
   but the sequence dimension is only 23 tokens. This produces `start=-511`, which
   is out of the valid XLA lazy-tensor range `[-23, 22]`. PyTorch/Python eager
   semantics clamp such indices to 0 (returning all elements), but the XLA lazy
   backend raises "Value out of range" instead.

## Fix
1. **tt_forge_models** — `remediation/gemma3_270m_gguf-causal_lm-pytorch-270M_IT_GGUF-single_device-inference`:
   Changed all 26 `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` signatures to
   `_patched_load_gguf_checkpoint(*args, **kwargs)` and updated the corresponding
   `_orig_load_gguf_checkpoint(*args, **kwargs)` call-through. This is a cherry-pick
   of commit `23ded3d9c0` from the Q8_0 variant's remediation branch.

2. **tt-xla** — `remediation/gemma3_270m_gguf-causal_lm-pytorch-270M_IT_GGUF-single_device-inference`
   `python_package/tt_torch/torch_overrides.py`:
   Added clamping logic to `TorchFunctionOverride.__torch_function__` that intercepts
   `func is torch.ops.aten.slice.Tensor`, reads the statically-known dimension size,
   and pre-clamps `start`/`end` to `[-size, size]` before dispatching to XLA.
   This cherry-picks commit `37424fca58` from the ddh0-gemma-3-40b-gguf remediation branch.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    299.41s (0:04:59)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: 26 GGUF loaders (`*/causal_lm/pytorch/loader.py`) — `_patched_load_gguf_checkpoint` signature widened
- `tt-xla`: `python_package/tt_torch/torch_overrides.py` — slice index clamping added

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 97c413f0dacdb37425a145622950cf59f05c8d05 |
| tt-forge-models | 8d5a80063d11c8fe838a11c820f8188bf92c79ff |
