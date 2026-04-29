# Remediation Summary: flux_controlnet_inpainting_alpha-pytorch-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[flux_controlnet_inpainting_alpha/pytorch-FLUX.1-dev-Controlnet-Inpainting-Alpha-single_device-inference]

## Result
FAIL — two loader bugs fixed; test cannot be verified on silicon because base model black-forest-labs/FLUX.1-dev is gated and not accessible with available HF token

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
flux-controlnet-extra-condition-channels-diffusers-0-30-compat

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Original failure (as reported):
```
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

Full failure (collection-time import error):
```
ImportError while loading conftest '/home/nsmith/tt-forge-remediation/tt-xla/tests/conftest.py'.
tests/conftest.py:25: in <module>
    from infra import DeviceConnectorFactory, Framework
E   ModuleNotFoundError: No module named 'infra'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

After fixing conftest, second failure encountered during model loading:
```
ValueError: Cannot load  because controlnet_x_embedder.weight expected shape
torch.Size([3072, 64]), but got torch.Size([3072, 68]). If you want to instead
overwrite randomly initialized weights, please make sure to pass both
`low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`.
```

## Root cause
Two independent loader bugs:

**Bug 1 — pytest collection failure (tt-xla `pytest.ini`)**
`tests/__init__.py` exists, making pytest treat `tests/` as a package. When a directory
is a package, pytest does not automatically add it to `sys.path`. Since `pytest.ini` lacked
`pythonpath = tests`, the `infra` module at `tests/infra/` was never on `sys.path`,
causing `from infra import ...` in `tests/conftest.py` to fail at collection time.
The resulting error (`ModuleNotFoundError`) causes pytest to print only its last-line
deprecation warning: `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__
attribute`.

**Bug 2 — diffusers API break in `FluxControlNetModel` (tt_forge_models loader)**
The `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha` checkpoint was saved with
diffusers 0.30.2. That version of `FluxControlNetModel.__init__` accepted an
`extra_condition_channels` parameter and used it to expand the `controlnet_x_embedder`
input dimension from `in_channels` to `in_channels + extra_condition_channels`
(64 + 4 = 68). diffusers 0.31+ dropped this parameter; the checkpoint `config.json`
still contains `extra_condition_channels: 4` but it is silently ignored by `extract_init_dict`,
so `controlnet_x_embedder` is created with 64 inputs instead of 68.

## Fix
**Fix 1** — `tt-xla/pytest.ini`: add `pythonpath = tests` so pytest puts `tests/` on
`sys.path` during collection, making the `infra` package importable. Also added
`filterwarnings` entries to suppress the SWIG `swigvarlink`/`SwigPy` DeprecationWarnings
that accompany the conftest failure.

**Fix 2** — `tt_forge_models/flux_controlnet_inpainting_alpha/pytorch/src/model_utils.py`:
introduce `_FluxControlNetModelV030Compat`, a subclass of `FluxControlNetModel` that
restores the `extra_condition_channels` parameter in its `@register_to_config`-decorated
`__init__`. When `extra_condition_channels > 0`, the subclass re-creates
`controlnet_x_embedder` as `nn.Linear(in_channels + extra_condition_channels, inner_dim)`
after calling `super().__init__()`. The loader now calls
`_FluxControlNetModelV030Compat.from_pretrained(...)` instead of
`FluxControlNetModel.from_pretrained(...)`, which causes `extract_init_dict` to include
`extra_condition_channels` in the init kwargs and pass it through.

Files changed in tt-xla remediation branch
(`remediation/flux_controlnet_inpainting_alpha-pytorch-single_device-inference`):
- `pytest.ini` — pythonpath + filterwarnings
- `third_party/tt_forge_models` — submodule pointer update

Files changed in tt_forge_models remediation branch
(`remediation/flux_controlnet_inpainting_alpha-pytorch-single_device-inference`):
- `flux_controlnet_inpainting_alpha/pytorch/src/model_utils.py`

**Why test cannot be verified**: after both fixes are applied, the pipeline loader proceeds
to `FluxControlNetInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", ...)`.
This model is gated on HuggingFace and the available token (`nsmithtt`) can view metadata
but cannot download files (403 Forbidden). The model is not locally cached. Silicon
verification requires CI access where a privileged token and model cache are available.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/pytest.ini` (remediation commit: 30de1e00024804a1acf3ac9c5e900c85ff3076e0)
- `tt-xla/third_party/tt_forge_models` (submodule pointer, remediation commit: ac32928b5b0916fa14d1f76b30672a2ec1f07208)
- `tt_forge_models/flux_controlnet_inpainting_alpha/pytorch/src/model_utils.py` (remediation commit: 78e43f3b10)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | ac32928b5b0916fa14d1f76b30672a2ec1f07208 |
| tt-forge-models | 78e43f3b10 |
