# Remediation Summary: leakcore-pytorch-LEAKCORE-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[leakcore/pytorch-LEAKCORE-single_device-inference]

## Result
SILICON_PASS — UNetWrapper flattens added_cond_kwargs dict; test passes on blackhole-p150b

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
sdxl-unet-added-cond-kwargs-dict-input-hang

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-24 12:45:23.603 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

## Root cause
The `load_inputs` method returned a dict containing `added_cond_kwargs` as a nested Python dict
(`{"text_embeds": ..., "time_ids": ...}`). When the test runner called `model(**inputs)`, it
passed `added_cond_kwargs={"text_embeds": ..., "time_ids": ...}` as a keyword argument to the
UNet's `forward` method. TorchXLA/StableHLO cannot handle Python dict-typed arguments in the
compiled graph — the dict causes a device hang that surfaces as a 30-second timeout.

A prior partial fix (commit 3100ced58f on the bringup branch) changed the loader to return
`self.pipeline.unet` instead of the full pipeline and switched from a positional-args list to a
kwargs dict, but still passed `added_cond_kwargs` as a nested dict kwarg. The hang persisted.

## Fix
Added `UNetWrapper(nn.Module)` in `leakcore/pytorch/loader.py` that accepts `text_embeds` and
`time_ids` as separate flat tensor kwargs and reconstructs `added_cond_kwargs` internally before
delegating to the real UNet. Updated `load_model` to return `UNetWrapper(self.pipeline.unet)` and
updated `load_inputs` to return `text_embeds` and `time_ids` as top-level dict keys instead of a
nested `added_cond_kwargs` dict.

- `tt-forge-models leakcore/pytorch/loader.py` — remediation branch
  `remediation/leakcore-pytorch-LEAKCORE-single_device-inference`
  commit `27d6e1d04d4eeffad7ec1de131e7b74504de24b5`

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    784.09s (0:13:04)
- Tier A attempts: N/A

## Files changed
- leakcore/pytorch/loader.py (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 27d6e1d04d4eeffad7ec1de131e7b74504de24b5 |
