# Remediation Summary: boltning_hyperd_sdxl/pytorch-boltning-hyperd-sdxl-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[boltning_hyperd_sdxl/pytorch-boltning-hyperd-sdxl-single_device-inference]

## Result
SILICON_PASS

## Failure
/home/nsmith/src/tt-xla/.local_venv/lib/python3.12/site-packages/diffusers/schedulers/scheduling_euler_discrete.py:437: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword

## Root cause
Loader layer (tt_forge_models). The loader's `load_model()` returned a
`StableDiffusionXLPipeline` object, which is not a `torch.nn.Module`. The
current test infrastructure asserts `isinstance(self._model, torch.nn.Module)`
in `_configure_model_for_inference()`, causing an `AssertionError` before
execution could reach the numpy DeprecationWarning. The pipeline's `load_inputs()`
also returned raw text strings instead of tensor inputs for the UNet.

The numpy DeprecationWarning in `scheduling_euler_discrete.py:437` (the originally
reported symptom) occurs during `scheduler.set_timesteps()` preprocessing and
remains present but non-fatal under numpy 2.1.2 / torch 2.7.0.

## Fix
Updated `boltning_hyperd_sdxl/pytorch/loader.py` in tt-forge-models:
- `load_model()` now loads the full `StableDiffusionXLPipeline` internally but
  returns only `self.pipeline.unet` (a `torch.nn.Module`).
- `load_inputs()` now calls `stable_diffusion_preprocessing_xl()` to produce
  properly preprocessed tensor inputs (`sample`, `timestep`,
  `encoder_hidden_states`, `added_cond_kwargs`) for the UNet's forward method.
- Added `boltning_hyperd_sdxl/pytorch/src/model_utils.py` with `load_pipe()`
  and `stable_diffusion_preprocessing_xl()` following the same pattern as
  `tekitoukawa_mix_v70_sdxl` and `stable_diffusion_xl`.

This is not a forbidden workaround: the UNet is the full computational core of
the SDXL model — no layers are trimmed, offloaded, or bypassed.

## Verification
pytest exit: PASSED
Wall-clock: 692.04 s (11 min 32 s)
Hardware: Blackhole (single device)

## Files changed
- `boltning_hyperd_sdxl/pytorch/loader.py` (modified)
- `boltning_hyperd_sdxl/pytorch/src/__init__.py` (new)
- `boltning_hyperd_sdxl/pytorch/src/model_utils.py` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 30f8da0391048c009ad4df412b138755d3226c75 |
| tt-forge-models | 476585f45340cfa163e33a7d4a6dea06a32c1a46 |
