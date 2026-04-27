# aesthetics_predictor_v1_vit/pytorch-Large-Patch14 Fix Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[aesthetics_predictor_v1_vit/pytorch-Large-Patch14-single_device-inference]`

## Status: PASS (SILICON_PASS)

## Root Cause

With transformers 5.x, `AutoModel.from_pretrained` fails with a `ValueError` because
the model class loaded from the remote code has `config_class = CLIPVisionConfig`, but
`AutoModel` is passed an `AestheticsPredictorConfig`. This mismatch is now strictly
enforced in transformers 5.x:

```
ValueError: The model class you are passing has a `config_class` attribute that is not
consistent with the config class you passed (model has CLIPVisionConfig and you passed
AestheticsPredictorConfig).
```

## Fix

In `aesthetics_predictor_v1_vit/pytorch/loader.py`, replaced `AutoModel.from_pretrained`
with a direct call using `AutoConfig.from_pretrained` + `get_class_from_dynamic_module`
to bypass AutoModel's config class consistency check. The model class is retrieved
directly from the remote module and instantiated with `from_pretrained`.

## Changes

### tt_forge_models (`remediation/adavar-pytorch-use-slow-image-processor`)
- `aesthetics_predictor_v1_vit/pytorch/loader.py`: Use `AutoConfig` +
  `get_class_from_dynamic_module` to load model class directly, bypassing AutoModel
  config_class consistency check introduced in transformers 5.x

### tt-xla (`arch-c-36-tt-xla-dev/nsmith/hf-bringup-13`)
- `third_party/tt_forge_models`: Updated submodule pointer
