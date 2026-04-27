# arcticlatent_flux1/pytorch-Dev_FP16 Remediation Summary

## Test
`tests/runner/test_models.py::test_all_models_torch[arcticlatent_flux1/pytorch-Dev_FP16-single_device-inference]`

## Status
**SILICON_PASS** — Test passes in ~12 minutes on TT silicon.

## Original Failure
```
TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)
```

## Root Cause
The `arcticlatent_flux1` model loader constructed HuggingFace download URLs using the
`resolve/main` path format:
```
https://huggingface.co/arcticlatent/flux1/resolve/main/unet/dev/flux1-dev-fp16.safetensors
```

However, the diffusers `_extract_repo_id_and_weights_name` function (used internally by
`FluxTransformer2DModel.from_single_file`) only strips `blob/main/` from the URL path
via its regex `([^/]+)/([^/]+)/(?:blob/main/)?(.+)`. When `resolve/main/` is used, it
is NOT stripped, so the extracted filename becomes
`resolve/main/unet/dev/flux1-dev-fp16.safetensors` instead of the correct
`unet/dev/flux1-dev-fp16.safetensors`. The HuggingFace API then returns a 404
for that non-existent path, which gets cached as `.no_exist`, causing the loader to throw
an `OSError` on subsequent runs.

In CI, the local model file cache already existed (populated before the URL format was
broken), so the download was bypassed and the test reached the device — but since
diffusers re-checked the file URL at load time and got a stale cached 404 response, it
eventually triggered a device timeout.

## Fix
**`tt-xla/third_party/tt_forge_models`** (`remediation/arcticlatent-flux1-url-fix`):
- Changed `_REPO_BASE_URL` in `arcticlatent_flux1/pytorch/loader.py` from
  `https://huggingface.co/{REPO_ID}/resolve/main` to
  `https://huggingface.co/{REPO_ID}/blob/main` so that diffusers' URL parser correctly
  extracts `repo_id = arcticlatent/flux1` and
  `weights_name = unet/dev/flux1-dev-fp16.safetensors`.

**`tt-xla`** (`nsmith/fix-arcticlatent-flux1-url`):
- Bumped `third_party/tt_forge_models` submodule to the fix commit.

## Submodule Hashes
- tt-metal: `3fa4d753550dba1d4aacc9af45b111ae540f63fc`
- tt-mlir:  `553c0632b353f8ac457aba0d01a460a5e0f5b5ee`
- tt-xla:   `3fbcb66cf9e317cb9973729ef1799890545b047f`
