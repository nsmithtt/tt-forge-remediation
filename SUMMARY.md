# Remediation Summary: intern_vl_3-pytorch-1B_Instruct-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[intern_vl_3/pytorch-1B_Instruct-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; pytest PASS on TT silicon in 223.94s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-meta-tensor-item-and-missing-post-init

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Tensor.item() cannot be called on meta tensors

Full traceback root:
```
modeling_intern_vit.py:312: in InternVisionEncoder.__init__
    dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
torch/utils/_device.py:103: in __torch_function__
tt_torch/torch_overrides.py:34: in __torch_function__
torch/_meta_registrations.py:7341: in meta_local_scalar_dense
    raise RuntimeError("Tensor.item() cannot be called on meta tensors")
```

## Root cause
Two loader-layer bugs, both introduced by transformers 5.x:

**Bug 1 — meta-tensor `.item()` in `InternVisionEncoder.__init__`:**
transformers 5.x `PreTrainedModel.get_init_context()` unconditionally
appends `torch.device("meta")` to the initialization context managers,
so the model class is always constructed on the meta device. Inside
`InternVisionEncoder.__init__`, the remote code calls
`torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)`
which, under the meta context, produces a meta tensor. It then iterates
with `.item()`, which raises `RuntimeError: Tensor.item() cannot be
called on meta tensors`.

**Bug 2 — missing `all_tied_weights_keys` from absent `post_init()` call:**
transformers 5.x `_finalize_model_loading` calls
`model._adjust_tied_keys_with_tied_pointers()`, which accesses
`self.all_tied_weights_keys` — an attribute set by `post_init()`.
`InternVLChatModel.__init__` (remote code) calls `super().__init__(config)`
but never calls `self.post_init()`, so the attribute is never set,
causing `AttributeError` during finalization.

## Fix
Both fixes are in `intern_vl_3/pytorch/loader.py` in tt-forge-models,
branch `remediation/intern_vl_3-pytorch-1B_Instruct-single_device-inference`,
commit `5ecbb52fa0b24ffd7a153b9d6d670014077ab604`.

**Fix 1:** Patch `torch.Tensor.item` before `from_pretrained` to return
`0.0` for meta tensors. This is safe because `DropPath` is a no-op in
eval mode regardless of `drop_prob`. The patch is restored in a `finally`
block.

**Fix 2:** Patch `PreTrainedModel._finalize_model_loading` to call
`model.post_init()` when `all_tied_weights_keys` is absent. Uses the
`PreTrainedModel.__dict__["_finalize_model_loading"].__func__` pattern
(patching the infrastructure class rather than the remote model class,
which is inaccessible via `importlib` due to `sys.modules` key mismatch).
Patch is also restored in a `finally` block.

## Verification
- pytest exit: PASS
- Hardware:    wormhole
- Duration:    223.94s (0:03:43)
- Tier A attempts: N/A

## Files changed
- `intern_vl_3/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 5ecbb52fa0b24ffd7a153b9d6d670014077ab604 |
