# Remediation Summary: metricx-pytorch-Hybrid_XL_v2p6-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[metricx/pytorch-Hybrid_XL_v2p6-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; test passes on silicon

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-sys-modules-cls-module-keyerror

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
KeyError: 'tt_forge_models.metricx.pytorch.loader'
```
in `transformers/modeling_utils.py:1940` during `MT5ForRegression.from_pretrained`.
(The reported failure message `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` appears in the pytest summary line, but the actual test error is the KeyError above.)

## Root cause
Two loader bugs:

**Bug 1** (tt-xla `dynamic_loader.py`): `DynamicLoader.import_model_loader` registers the loaded module in `sys.modules` only under the dash key (`tt-forge-models.metricx.pytorch.loader`). The spec name — which becomes `mod.__name__` and therefore `cls.__module__` for all classes defined in the module — uses underscores (`tt_forge_models.metricx.pytorch.loader`). Transformers 5.x added `_can_set_experts_implementation` and `_can_set_attn_implementation` methods that do `sys.modules[cls.__module__]`; this `KeyError`s because the underscore key is absent.

**Bug 2** (metricx loader): `MT5ForRegression.forward` creates `decoder_input_ids` with `torch.LongTensor([0])`, which always produces a CPU tensor. When Dynamo traces the model with `input_ids` on `xla:0`, fake-tensor propagation raises "found two different devices xla:0, cpu".

## Fix
**Fix 1** — `tt-xla/tests/runner/utils/dynamic_loader.py`: After `sys.modules[module_path] = mod` (dash key), also add `sys.modules[mod.__name__] = mod` (underscore key) before `exec_module`. This is a general fix; all loader modules now satisfy transformers 5.x `sys.modules[cls.__module__]` lookups.

**Fix 2** — `tt_forge_models/metricx/pytorch/loader.py`: Replace
```python
decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
```
with
```python
decoder_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=input_ids.device)
```
so the decoder start token is created on the same device as `input_ids`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    146.07s (0:02:26)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/runner/utils/dynamic_loader.py`
- `tt_forge_models/metricx/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d75355 |
| tt-mlir         | 553c0632b  |
| tt-xla          | acc2c71bb  |
| tt-forge-models | 05962de684 |
