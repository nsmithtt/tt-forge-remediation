# Remediation Summary: lfm2_5_vl_1_6b_absolute_heresy_mpoa-image_text_to_text-pytorch-LFM2_5_VL_1_6B_absolute_heresy_MPOA-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lfm2_5_vl_1_6b_absolute_heresy_mpoa/image_text_to_text/pytorch-LFM2_5_VL_1_6B_absolute_heresy_MPOA-single_device-inference]

## Result
FAIL — pjrt-device-to-host-transfer Error 13 during XLA graph extraction in Lfm2VlForConditionalGeneration.forward (Tier B)

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
pjrt-device-to-host-transfer

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ModuleNotFoundError: No module named 'infra'
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
After pytest.ini fix:
```
NotImplementedError: (from F.interpolate antialias=True with BFloat16 tensors, CPU-fallback in TorchFunctionMode)
```
After BFloat16 interpolate fix:
```
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

## Root cause

Three layered failures were uncovered:

1. **pytest.ini missing `pythonpath = tests`** — without this, pytest cannot find the local `infra` package. The SWIG DeprecationWarning about `swigvarlink` (the reported failure message) was a side-effect of the conftest import failure. Fix: add `pythonpath = tests` and `filterwarnings` entries to `pytest.ini` in tt-xla.

2. **BFloat16 anti-aliased interpolate — no CPU kernel** (Tier A, fixed) — `F.interpolate(antialias=True)` dispatches `_upsample_bilinear2d_aa`, which has no BFloat16 CPU kernel. When XLA CPU-falls back this op for BFloat16 tensors (SigLIP2 positional embedding resize in LFM2-VL), the CPU kernel raises `NotImplementedError`. Fix: intercept in `TorchFunctionOverride` in `torch_overrides.py`, cast to float32, interpolate, cast back.

3. **pjrt-device-to-host-transfer Error 13** (Tier B, unfixed) — after the two fixes above, the test fails with `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` during `torch_xla.sync` in `extract_compiled_graph_helper`. Dynamo breaks the graph at `Lfm2VlForConditionalGeneration.forward:419` (before the `self.model(...)` call, likely due to data-dependent operations in `_merge_multimodal_embeddings` such as boolean-masked indexing `inputs_embeds[special_image_mask]`). The second subgraph starting at `self.model(...)` is compiled by the TT backend; the lazy-evaluation sync fails because the graph contains a device-to-host transfer that the TT PJRT plugin does not support (Error code 13). This is the known `pjrt-device-to-host-transfer` Tier B bug.

## Fix

Fixes 1 and 2 are committed on `remediation/lfm2_5_vl_1_6b_absolute_heresy_mpoa-image_text_to_text-pytorch-LFM2_5_VL_1_6B_absolute_heresy_MPOA-single_device-inference` in tt-xla:

- `pytest.ini` — add `pythonpath = tests` and filter SWIG DeprecationWarnings (commit `34b372526`)
- `python_package/tt_torch/torch_overrides.py` — cast BFloat16→float32 around `antialias=True` interpolate in `TorchFunctionOverride` (commit `cbe488d46`)

**Proposed fix for Tier B (Error 13):** The TT PJRT plugin needs to support device-to-host tensor transfers during lazy-evaluation sync. The transfer is required because `_merge_multimodal_embeddings` uses boolean indexing (`inputs_embeds[special_image_mask]`) whose output shape is data-dependent. Until the PJRT plugin supports this path, the full LFM2-VL model cannot compile on TT hardware.

## Tier B justification
new-infrastructure — The TT PJRT plugin does not support device-to-host transfers during lazy-evaluation `xla_step_marker`. Implementing this requires new infrastructure in the PJRT transport layer; it is not a scoped one- or two-file change.

## Verification
- pytest exit: FAIL
- Hardware: blackhole-p150b
- Duration: 178.42s (first run with both fixes), 130.03s (second confirmation run)
- Tier A attempts: 1 (BFloat16 anti-aliased interpolate fix — successful; revealed second Tier B bug)

## Files changed
In tt-xla (`remediation/lfm2_5_vl_1_6b_absolute_heresy_mpoa-...` branch):
- `pytest.ini` — add `pythonpath = tests` and SWIG DeprecationWarning filter
- `python_package/tt_torch/torch_overrides.py` — BFloat16→float32 cast around antialias interpolate

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | cbe488d465221d7cd5f3c8e05642696fe94c92b1 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
