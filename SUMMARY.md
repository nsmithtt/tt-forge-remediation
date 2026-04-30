# Remediation Summary: diar_sortformer-speaker_diarization-pytorch-Diar_Sortformer_4spk_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[diar_sortformer/speaker_diarization/pytorch-Diar_Sortformer_4spk_v1-single_device-inference]

## Result
FAIL — STFT op inside model preprocessor produces XLAComplexFloatType which tt-xla cannot handle; no complex tensor support in XLA/tt-mlir

## Stack layer
tt-xla

## Tier
B

## Bug fingerprint
xla-complex-tensor-type-unsupported

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: XLAComplexFloatType

While executing %stft : [num_users=1] = call_function[target=torch.ops.higher_order.wrap_with_autocast](args = (xla, None, False, None, %submod_5, %self___featurizer__buffers__window, %masked_fill), kwargs = {})
```

## Root cause
The SortformerEncLabelModel's audio preprocessor (featurizer) calls `torch.ops.aten.stft` as part of feature extraction (mel spectrogram). STFT produces a complex-valued tensor (`XLAComplexFloatType`). The tt-xla/PJRT backend does not implement a lowering for complex tensor types, so the graph compilation fails immediately when the complex STFT output tensor is encountered.

Two bugs were found:
1. **Loader bug (fixed):** `load_inputs()` returned keys `input_signal`/`input_signal_length` but `SortformerEncLabelModel.forward()` expects `audio_signal`/`audio_signal_length`. This caused a `TypeError` before the XLA compiler was even reached.
2. **Compiler-stack bug (Tier B):** After fixing the loader key names, the test fails at XLA compilation because `aten.stft` produces `XLAComplexFloatType` which is not supported in the tt-xla backend.

## Fix
**Loader fix (committed to tt_forge_models):** Corrected `load_inputs()` key names from `input_signal`/`input_signal_length` to `audio_signal`/`audio_signal_length` to match `SortformerEncLabelModel.forward()` signature.

File changed: `diar_sortformer/speaker_diarization/pytorch/loader.py`
Branch: `remediation/diar_sortformer-speaker_diarization-pytorch-Diar_Sortformer_4spk_v1-single_device-inference`

**Compiler bug (unfixed):** Implementing complex tensor (`XLAComplexFloatType`) support in tt-xla requires:
- A new complex data type in the PJRT buffer layer
- New lowering patterns in tt-mlir for complex ops (stft, complex multiply, real/imag extraction)
- Potential runtime support in tt-metal for complex arithmetic

This is new infrastructure work spanning multiple components across multiple repos.

## Tier B justification (FAIL with Tier=B only — omit otherwise)
new-infrastructure

Complex tensor type support (XLAComplexFloatType) is not implemented anywhere in the tt-xla to tt-mlir to tt-metal stack; adding it requires new data types, new op lowerings, and potentially new runtime kernels across multiple repos.

## Verification
- pytest exit: FAIL
- Hardware: n150
- Duration: 59.41s (to failure)
- Tier A attempts: N/A

## Files changed
- `diar_sortformer/speaker_diarization/pytorch/loader.py` (tt_forge_models) — fixed `load_inputs()` key names

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | aea0042902feedc58d65bbaf8eb51e67aa638a5e |
| tt-forge-models | f7e62107a6f66bd4d7310794535a68eada87498a |
