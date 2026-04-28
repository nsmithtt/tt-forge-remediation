# Remediation Summary: audioseal-pytorch-audioseal_wm_16bits-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[audioseal/pytorch-audioseal_wm_16bits-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
audioseal-moshi-internal-torch-compile-xla-incompatible

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two loader-layer bugs were found and fixed:

**Bug 1** (pre-existing fix on hf-bringup-38):
```
ImportError: cannot import name 'AudioSeal' from 'audioseal'
```
The local `tt_forge_models/audioseal/` directory shadowed the pip-installed
`audioseal` package because `models_root` is prepended to `sys.path` by the
dynamic loader.  This fix was already present on the `hf-bringup-38` branch.

**Bug 2** (fixed in this report):
```
torch._inductor.exc.InductorError: KeyError: 'xla'
```
After the import-shadow fix, `torch.compile(model, backend="tt")` compiled the
model successfully, but execution on TT silicon failed because
`audioseal.libs.moshi.utils.compile.torch_compile_lazy` lazily wraps internal
sub-modules (e.g., `SeanetEncoder`) with `torch.compile()` using the default
inductor backend.  When those wrappers first fire with XLA tensors, the inductor
backend tries to handle device type `'xla'`, which is not registered in its
`device_op_overrides_dict`, raising `KeyError: 'xla'`.

## Root cause
The root cause of Bug 2 is in the loader layer.  The `audioseal` pip package
(v0.2.0) vendors the Moshi library and uses a `@torch_compile_lazy` decorator
to defer compilation of `SeanetEncoder` sub-modules.  The decorator calls
`torch.compile(fun)` with no `backend` argument on the first call, so it uses
the inductor backend.  When the AudioSeal model is compiled with
`torch.compile(backend="tt")` and executed on an XLA device, the inner compile
fires with XLA-device tensors and fails because inductor has no handler for the
`'xla'` device type.

The moshi compile module already exposes a `_compile_disabled` flag (used by its
own `no_compile()` context manager) that prevents the lazy compilation from
happening at call time.  Setting this flag to `True` after loading the model
prevents inductor from ever being invoked during TT-silicon inference.

## Fix
In `audioseal/pytorch/loader.py`, after `model.eval()` and the optional dtype
cast, import `audioseal.libs.moshi.utils.compile` and set
`_compile_disabled = True`.  This disables all `@torch_compile_lazy` wrappers
for the lifetime of the process, so XLA tensors never hit the inductor backend.

**File changed:**
`tt-forge-models: audioseal/pytorch/loader.py`

Branch: `remediation/audioseal-pytorch-audioseal_wm_16bits-single_device-inference`
Commit: `f532008d09245759331434d0d850ec96f883650d`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    66.15s (0:01:06)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: audioseal/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | f532008d09245759331434d0d850ec96f883650d |
