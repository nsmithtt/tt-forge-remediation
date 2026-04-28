# Remediation Summary: distil_whisper-speech_recognition-pytorch-Distil_small_en-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[distil_whisper/speech_recognition/pytorch-Distil_small_en-single_device-inference]

## Result
SILICON_PASS — loader fix: cast inputs to bfloat16 and add decoder_input_ids

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
distil-whisper-loader-inputs-dtype-mismatch

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same

## Root cause
The loader layer bug: `load_inputs` created float32 tensors from the audio
processor but the model was loaded in bfloat16. When `F.conv1d` was called on
the CPU reference run, PyTorch rejected the dtype mismatch between float32
input features and bfloat16 Conv1d bias. Additionally, Whisper is an
encoder-decoder model that requires `decoder_input_ids` to be passed to
`forward()`, and without `use_cache=False` the output contains an
`EncoderDecoderCache` object that the comparison evaluator cannot compare.

## Fix
In `tt_forge_models/distil_whisper/speech_recognition/pytorch/loader.py`,
`load_inputs` was updated to:
1. Cast all floating-point input tensors to `dtype_override` when provided.
2. Append `decoder_input_ids` (filled with `decoder_start_token_id`) to the
   inputs dict, as Whisper's `forward()` requires this for encoder-decoder
   generation.
3. Set `use_cache=False` to avoid returning `EncoderDecoderCache` in the
   output, which the comparison evaluator cannot handle via `torch.equal`.

Branch: `remediation/distil_whisper-speech_recognition-pytorch-Distil_small_en-single_device-inference`
in `tenstorrent/tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    90.01s
- Tier A attempts: N/A

## Files changed
- `distil_whisper/speech_recognition/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 2a5f0e50d2656095e9a7b6245638f5f22e4d75c1 |
| tt-forge-models | 0098755b034d9da244d93044736e21cdb720d6fb |
