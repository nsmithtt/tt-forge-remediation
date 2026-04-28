# Remediation Summary: hubert/feature_extraction/pytorch-mHuBERT_147-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[hubert/feature_extraction/pytorch-mHuBERT_147-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-input-dtype-not-cast-to-bfloat16

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: expected scalar type Float but found BFloat16

## Root cause
The `TorchDynamicLoader.load_model()` in tt-xla calls
`loader.load_model(dtype_override=torch.bfloat16)`, converting the model
weights to bfloat16. However, `load_inputs()` in
`hubert/feature_extraction/pytorch/loader.py` did not cast the processor
output tensors to bfloat16. The `Wav2Vec2FeatureExtractor` always returns
`input_values` in float32. When the CPU reference run called `model(**inputs)`,
`Conv1d` received float32 input with bfloat16 weights and raised
`RuntimeError: expected scalar type Float but found BFloat16`.

## Fix
Added dtype casting of floating-point input tensors after the processor call
in `load_inputs()` inside
`tt_forge_models/hubert/feature_extraction/pytorch/loader.py`.
When `dtype_override` is provided, all floating-point tensors in the
processor output dict are cast to `dtype_override` (bfloat16).

This follows the same pattern already used by other loaders in the repo
(e.g. `dinov3/feature_extraction/pytorch/loader.py`).

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    97.75s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/hubert/feature_extraction/pytorch/loader.py` — cast inputs to dtype_override in load_inputs

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | c37b292a4519c9816d3befc6235fbc7b181c588c |
| tt-forge-models | 8eed8d9795db99f37ec4a10734697b1c7d247a84 |
