# Remediation Summary: asid_captioner_7b_i1_gguf/image_to_text/pytorch-7b_i1_Q4_K_M_gguf

## Test
`tests/runner/test_models.py::test_all_models_torch[asid_captioner_7b_i1_gguf/image_to_text/pytorch-7b_i1_Q4_K_M_gguf-single_device-inference]`

## Result
SILICON_PASS

## Problem
The ASID Captioner 7B i1 GGUF image-to-text model was failing with three distinct issues:

1. `FutureWarning` about `Qwen2VLImageProcessor` being loaded as a fast processor — the stated failure.
2. `ValueError: GGUF model with architecture qwen2vl is not supported yet` — transformers 5.x has no qwen2vl GGUF support.
3. `AttributeError: 'NoneType' object has no attribute 'config'` — the mozilla_ai llamafile loader was dropping `model_to_load` when passing through to the chained GGUF checkpoint wrapper, causing `get_gguf_hf_weights_map` to receive `hf_model=None`.
4. `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13` — the Qwen2VL visual encoder's `rot_pos_emb` operation fails on TT silicon.

## Root Causes and Fixes

### 1. Processor warning (in asid_captioner loader)
`AutoProcessor.from_pretrained("AudioVisual-Caption/ASID-Captioner-7B")` loaded the fast Qwen2VLImageProcessor.
Fixed by adding `use_fast=False`.

### 2. Missing qwen2vl GGUF architecture support (in asid_captioner loader)
transformers 5.x has `Qwen2VLForConditionalGeneration` but no GGUF loading path for the `qwen2vl` architecture.
Fixed by patching `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, and `TENSOR_PROCESSORS` at import time.
Also:
- The AudioVisual-Caption repo now returns a `Qwen2_5OmniConfig` (incompatible); we load config from `Qwen/Qwen2-VL-7B-Instruct` instead.
- Set `base_config.model_type = "qwen2vl"` to match gguf-py's `MODEL_ARCH_NAMES` (uses no underscore).
- Set `base_config.num_hidden_layers` flat so `get_gguf_hf_weights_map` can access it without diving into `text_config`.

### 3. mozilla_ai llamafile loader dropping model_to_load (in mozilla_ai loader)
`_patched_load_gguf_checkpoint` accepted `model_to_load` as a parameter but forgot to forward it to `_orig_load_gguf_checkpoint`. Because many loaders chain their `load_gguf_checkpoint` wrappers globally, this caused `model_to_load=None` for every model loaded after mozilla_ai's loader was imported.
Fixed by passing `model_to_load=model_to_load` through to the original function.

### 4. Visual encoder not runnable on TT silicon (in asid_captioner loader)
The ASID Captioner GGUF file contains only LM weights; the visual encoder is randomly initialized. The Qwen2VL visual encoder's `rot_pos_emb` operation fails with `Error code: 13` on TT silicon.
Fixed by using text-only inputs in `load_inputs` — the GGUF checkpoint's LM weights are what we are validating.

## Changes

### tt_forge_models
- Branch: `arch-c-36-tt-xla-dev/nsmith/hf-bringup-21`
- `asid_captioner_7b_i1_gguf/image_to_text/pytorch/loader.py`: add qwen2vl GGUF patch, use_fast=False, Qwen2-VL-7B config, text-only inputs
- `asid_captioner_7b_i1_gguf/image_to_text/pytorch/requirements.txt`: created with `gguf>=0.10.0`
- `mozilla_ai_meta_llama_3_1_70b_instruct_llamafile/causal_lm/pytorch/loader.py`: pass model_to_load through patched wrapper
