# Remediation Summary: mhubert-feature_extraction-pytorch-Base_25hz-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mhubert/feature_extraction/pytorch-Base_25hz-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
loader-dtype-override-not-applied-to-inputs

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   RuntimeError: expected scalar type Float but found BFloat16

## Root cause
`TorchDynamicLoader.load_inputs()` calls `loader.load_inputs(dtype_override=torch.bfloat16)` when the parameter is present in the signature. The mHuBERT loader accepted the `dtype_override` parameter but never applied it to the returned tensors from `Wav2Vec2FeatureExtractor`. The model was correctly loaded in bfloat16 (via `load_model(dtype_override=bfloat16)` → `model.to(bfloat16)`), but the `input_values` tensor from the processor remained float32. When `F.conv1d` was called with float32 input and bfloat16 weights, PyTorch raised "expected scalar type Float but found BFloat16".

## Fix
Added `import torch` at the top of the loader and added a dtype cast at the end of `load_inputs()` in `mhubert/feature_extraction/pytorch/loader.py`:

```python
if dtype_override is not None:
    inputs = {
        k: v.to(dtype_override) if torch.is_tensor(v) and v.dtype.is_floating_point else v
        for k, v in inputs.items()
    }
```

This matches the pattern already applied in `hubert/feature_extraction/pytorch/loader.py` (remediation branch `hubert-feature_extraction-pytorch-mHuBERT_147-single_device-inference`).

Repo: `tt-forge-models`, branch: `remediation/mhubert-feature_extraction-pytorch-Base_25hz-single_device-inference`

## Verification
- pytest exit: PASS
- Hardware: wormhole
- Duration: 68.84s
- Tier A attempts: N/A

## Files changed
- `mhubert/feature_extraction/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 82ed51bbe768cad885050f5a6eb1e66499271f62 |
