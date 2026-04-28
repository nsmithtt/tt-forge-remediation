# Remediation Summary: gemma3_gguf-image_to_text-pytorch-27b_it_vl_polaris_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3_gguf/image_to_text/pytorch-27b_it_vl_polaris_gguf-single_device-inference]

## Result
FAIL — GGUF file contains only the Gemma3 text model (model_type=gemma3_text); no vision encoder is present, so AutoModelForImageTextToText cannot load it

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-vl-model-missing-vision-encoder

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
Two sequential loader bugs were found:

**Bug 1 (fixed):** `ValueError: Unrecognized processing class in mradermacher/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCT-GGUF. Can't instantiate a processor, a tokenizer, an image processor, a video processor or a feature extractor for this model.`

**Bug 2 (unfixable — root cause):** `ValueError: Unrecognized configuration class <class 'transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig'> for this kind of AutoModel: AutoModelForImageTextToText. Model type should be one of ... Gemma3Config ...`

## Root cause
The GGUF repo (`mradermacher/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCT-GGUF`) was originally a vision-language fine-tune of Gemma3 27B, but when converted to GGUF format the vision encoder was not included. The GGUF file's model_type metadata is `gemma3_text` (language model only), which transformers maps to `Gemma3TextConfig`. `AutoModelForImageTextToText` only accepts `Gemma3Config` (model_type=`gemma3`, the multimodal config that includes both text and vision components). The GGUF config mapping in transformers has no entries for vision components, confirming that Gemma3 multimodal GGUF loading is not supported.

Bug 1 is a loader bug: the processor was being fetched from the GGUF repo, which ships no processor files. The fix loads the processor from the base model (`DavidAU/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCT`) with `use_fast=False` (to handle the transformers 5.x `Gemma3ImageProcessor` fast/slow breaking change).

Bug 2 is the blocking root cause. Per the skill rules: "The GGUF doesn't ship the encoder is not a justification — file an issue and report failure."

## Fix
**Bug 1 (committed):** In `gemma3_gguf/image_to_text/pytorch/loader.py`, changed `AutoProcessor.from_pretrained(pretrained_model_name)` to load from `DavidAU/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCT` with `use_fast=False`.

**Bug 2 (proposed, not attempted):** A valid fix would be to find or create a Gemma3 27B VL GGUF that includes the vision encoder, or to replace the GGUF variant with a non-GGUF multimodal alternative (e.g., the direct `DavidAU/Gemma3-27B-it-vl-Polaris-HI16-Heretic-Uncensored-INSTRUCT` model). Switching the loader to `AutoModelForCausalLM` would skip the vision path and is forbidden.

## Verification
- pytest exit: FAIL
- Hardware: not-run
- Duration: N/A
- Tier A attempts: N/A

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma3_gguf/image_to_text/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 6bed851616179e92f624c07be69bb763da5eca52 |
| tt-forge-models | 06cd078f07a0942b84f53c6dc4b201f7bcbdba66 |
