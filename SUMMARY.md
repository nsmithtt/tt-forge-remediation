# Remediation Summary: distil_whisper/speech_recognition/pytorch-Distil_large_v3_5_ct2-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[distil_whisper/speech_recognition/pytorch-Distil_large_v3_5_ct2-single_device-inference]

## Result
SILICON_PASS

## Failure
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13

(On the configured branch the immediate failure was: RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same — a dtype mismatch upstream of the compiler error.)

## Root cause
Loader layer (tt-forge-models). The `load_inputs()` method in `distil_whisper/speech_recognition/pytorch/loader.py` had two bugs:

1. **Dtype mismatch**: The method accepted `dtype_override` but never used it to cast the returned processor output. `TorchDynamicLoader` always passes `dtype_override=torch.bfloat16` to loaders that declare the parameter, so the model weights were bfloat16 but the `input_features` tensor remained float32, causing a Conv1d type mismatch on the first CPU forward pass.

2. **Missing `decoder_input_ids`**: Whisper is an encoder-decoder (seq2seq) model; `WhisperForConditionalGeneration.forward()` requires `decoder_input_ids` in addition to `input_features`. The loader only returned `input_features`, so every forward call raised `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`.

3. **`EncoderDecoderCache` in output**: With `use_cache=True` (the default), the model output contains `past_key_values: EncoderDecoderCache`. The test framework's comparison evaluator uses `torch.equal` on every leaf of the output PyTree; `EncoderDecoderCache` is not a tensor, so comparison failed. Setting `use_cache=False` makes `past_key_values=None`, which the `None`-aware comparison handles correctly.

## Fix
All changes are in `tt-forge-models` at `distil_whisper/speech_recognition/pytorch/loader.py`:

- Cast all floating-point tensors in the processor output to `dtype_override` when it is not `None`.
- Load `WhisperConfig` to obtain `decoder_start_token_id` and add a `decoder_input_ids` tensor (shape `[1, 2]`, filled with the start token) to the inputs dict.
- Add `use_cache=False` to the inputs dict so the model output's `past_key_values` is `None`, keeping the output structure within what the comparison evaluator can handle.

None of these changes are forbidden workarounds: no model depth is trimmed, no modules are offloaded to CPU, no PCC threshold is lowered, and no compiler-stack behavior is bypassed.

## Verification
pytest exit status: PASSED  
Wall-clock duration: 94.31 s (1 min 34 s)  
Hardware: Blackhole p150 (hostname bh-lb-13-tt-forge-remediation-7)

## Files changed
- `distil_whisper/speech_recognition/pytorch/loader.py` (in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 64df5ab989fe94a52420ad05774f8754472720d0 |
| tt-forge-models | 80724694153ea680cff1534d570e7a2b30403b87 |
