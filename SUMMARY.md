# Remediation Summary: blip-visual_question_answering-pytorch-Capfilt_Large-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[blip/visual_question_answering/pytorch-Capfilt Large-single_device-inference]

## Result
SILICON_PASS — added missing decoder_input_ids to load_inputs; test passes in 60.42s

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
blip-vqa-missing-decoder-input-ids

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
ValueError: Either `decoder_input_ids` or `labels` should be passed when calling `forward` with `BlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`

## Root cause
`BlipForQuestionAnswering` uses an encoder-decoder architecture. Its `forward()` method requires `decoder_input_ids` (the decoder start tokens) to be provided explicitly for inference, in addition to the encoder inputs (`pixel_values`, `input_ids`, `attention_mask`) produced by the processor. The loader's `load_inputs` only returned the processor outputs, omitting `decoder_input_ids`, causing `forward()` to raise a `ValueError` at runtime.

The BLIP processor wraps a BERT tokenizer whose `bos_token_id` is `None`; the fix falls back to `cls_token_id` (token ID 101, the `[CLS]` token used as the decoder start token in BERT-based decoders).

## Fix
Added `decoder_input_ids` to the inputs dict returned by `load_inputs` in:
- `blip/visual_question_answering/pytorch/loader.py` (in `tt_forge_models`)

The fix constructs a `(1, 1)` tensor with the BOS/CLS token ID and adds it to `inputs` before the batch-size repeat, so it is correctly replicated when `batch_size > 1`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    60.42s
- Tier A attempts: N/A

## Files changed
- `blip/visual_question_answering/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 650605da750113039cae87ee2445c3e3ffdd49b8 |
