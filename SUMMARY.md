# Remediation Summary: hubert-speech_recognition-pytorch-Japanese_Phoneme_CTC_v4-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[hubert/speech_recognition/pytorch-Japanese_Phoneme_CTC_v4-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
hubert-loader-inputs-dtype-not-cast-to-bfloat16

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: expected scalar type Float but found BFloat16

Full traceback ends at:
  transformers/models/hubert/modeling_hubert.py:172: in forward
      hidden_states = self.conv(hidden_states)
  torch/nn/modules/conv.py:366: in _conv_forward
      return F.conv1d(
  python_package/tt_torch/torch_overrides.py:34: in __torch_function__
      return func(*args, **(kwargs or {}))
  RuntimeError: expected scalar type Float but found BFloat16

## Root cause
The loader's `load_inputs()` method accepts `dtype_override` and passes it to
`_load_processor()`, but never applies it to the returned input tensors. When
`TorchDynamicLoader` calls `load_inputs(dtype_override=torch.bfloat16)`, the
`Wav2Vec2FeatureExtractor` (HuBERT's processor) still returns `input_values`
as float32. Meanwhile, `load_model(dtype_override=torch.bfloat16)` correctly
converts all model weights—including the feature-extractor Conv1d layers—to
bfloat16. The dtype mismatch (bfloat16 weights, float32 input) is caught by
PyTorch's conv1d kernel: "expected scalar type Float [i.e., what the weight
is] but found BFloat16 [i.e., what the input is]."

## Fix
In `hubert/speech_recognition/pytorch/loader.py`, added a post-processing
step in `load_inputs` that casts all floating-point entries of the processor
output dict to `dtype_override` when it is provided:

```python
if dtype_override is not None:
    inputs = {
        k: v.to(dtype_override) if torch.is_floating_point(v) else v
        for k, v in inputs.items()
    }
```

This ensures `input_values` is bfloat16 when the model is bfloat16, so the
Conv1d dtype mismatch no longer occurs in either the CPU reference run or the
TT device run.

Commit: `41e8ff3ca17155121312fd6ee68fb8e865de1715` on branch
`remediation/hubert-speech_recognition-pytorch-Japanese_Phoneme_CTC_v4-single_device-inference`
in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    71.13s (0:01:11)
- Tier A attempts: N/A

## Files changed
- `hubert/speech_recognition/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 41e8ff3ca17155121312fd6ee68fb8e865de1715 |
