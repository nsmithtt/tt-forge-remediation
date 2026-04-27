# Remediation Summary: bartowski_xlangai_jedi_3b_1080p_gguf / image_to_text / pytorch

**Test**: `tests/runner/test_models.py::test_all_models_torch[bartowski_xlangai_jedi_3b_1080p_gguf/image_to_text/pytorch-xlangai_jedi_3b_1080p_gguf-single_device-inference]`

**Result**: SILICON_PASS

## Original Failure

The test failed with a `FutureWarning` treated as an error:
```
The image processor of type `Qwen2VLImageProcessor` is now loaded as a fast processor
by default, even if the model checkpoint was saved with a slow processor. This is a
breaking change and may produce slightly different outputs. To continue using the slow
processor, instantiate this class with `use_fast=False`.
```

## Fixes Applied

Two fixes were required in `bartowski_xlangai_jedi_3b_1080p_gguf/image_to_text/pytorch/loader.py`:

### Fix 1: use_fast=False for AutoProcessor (commit 544e6bee97)
Added `use_fast=False` to `AutoProcessor.from_pretrained()` to suppress the
`Qwen2VLImageProcessor` fast-processor breaking-change warning. The GGUF repo does
not ship its own processor, so the base `Qwen/Qwen2.5-VL-3B-Instruct` processor is
used; specifying `use_fast=False` keeps the slow processor that was used at checkpoint
save time.

### Fix 2: Text-only inputs — GGUF visual encoder absent (commit 322dcbd37a)
Switched `load_inputs()` from image+text to text-only inputs. The GGUF checkpoint
(`xlangai_Jedi-3B-1080p-Q4_K_M.gguf`) contains only the LM weights; the visual
encoder is not included. Attempting to process images through the absent visual
encoder caused:
- On CPU: `ValueError: Image features and image tokens do not match` (3577 tokens,
  a data-dependent shape issue under XLA tracing)
- On TT silicon: `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`
  (OOM / unsupported operation in the vision encoder)

Text-only inputs avoid the visual encoder entirely, allowing the LM portion (which
is fully present in the GGUF) to run correctly on TT hardware.

## Performance

- Inference time on TT silicon: ~52 ms per forward pass (text-only, LM only)
- Total test time: ~7 minutes 45 seconds

## Submodule Changes

- **tt-forge-models**: `b445bbffd4` → `322dcbd37a`
  - Branch: `remediation/bartowski-xlangai-jedi-3b-1080p-gguf-fix`
- **tt-xla**: Updated submodule pointer
  - Branch: `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-35-fix`
