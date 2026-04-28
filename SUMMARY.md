# Remediation Summary: audiox_south-speech_recognition-pytorch-AudioX_South_v1-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[audiox_south/speech_recognition/pytorch-AudioX_South_v1-single_device-inference]

## Result
FAIL — loader bug fixed (list→dict inputs), but model is gated (jiviai/audioX-south-v1) and no HF token available; silicon verification not possible

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
whisper-decoder-input-ids-passed-as-attention-mask

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

(The SWIG warning is the last stderr line after test completion; the actual runtime failure on CI is
`ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds` in WhisperDecoder.forward,
caused by the loader returning inputs as a list instead of a dict.)

## Root cause
`load_inputs` returned `[input_features, decoder_input_ids]` (a list). The test framework's
`_get_forward_method_args` converts a list to positional args, so the model is called as:
`model(input_features, decoder_input_ids)`.

`WhisperForConditionalGeneration.forward` signature is:
`(self, input_features, attention_mask, decoder_input_ids, ...)`.

With positional routing, `decoder_input_ids` (shape [1,1]) lands in `attention_mask` (position 2)
and `decoder_input_ids` receives `None`. `WhisperDecoder.forward` then raises:
`ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`.

The `sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute` is a harmless
Python 3.12 SWIG cleanup warning printed to stderr after pytest exits; it is not the root cause.

## Fix
In `tt_forge_models/audiox_south/speech_recognition/pytorch/loader.py`:

1. `load_inputs` now returns a dict `{"input_features": input_features, "decoder_input_ids": decoder_input_ids}`
   so the test framework routes inputs via kwargs, correctly mapping `decoder_input_ids` by keyword.

2. Added `use_cache: False` to `model_kwargs` in `load_model` to prevent KV cache tensors from
   flowing through the compiled graph (consistent with other Whisper loaders in the project).

## Verification
- pytest exit: not run (model jiviai/audioX-south-v1 is gated; no HF token available for download)
- Hardware:    not-run
- Duration:    n/a
- Tier A attempts: N/A

## Files changed
- tt_forge_models: audiox_south/speech_recognition/pytorch/loader.py

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | b360a871babf1030ee87812499f26c7a1bbb9c52 |
| tt-forge-models | d475f44439fadc94e2a879258b4b9fcd6f338338 |
