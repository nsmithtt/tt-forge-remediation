# Remediation Summary: lighton_ocr_2_1b_ocr_soup_gguf-image_to_text-pytorch-lighton_ocr_2_1b_ocr_soup_gguf-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lighton_ocr_2_1b_ocr_soup_gguf/image_to_text/pytorch-lighton_ocr_2_1b_ocr_soup_gguf-single_device-inference]

## Result
FAIL — PCC = 0.80 on TT silicon (required ≥ 0.99); loader bugs fixed but compiler-stack precision bug remains

## Stack layer
loader, tt-mlir

## Tier
B

## Bug fingerprint
lighton-ocr-vl-pcc-mismatch-compiler

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.8053926481379448. Required: pcc=0.95.
```

## Root cause

Three loader bugs prevented the model from loading at all; once fixed, a compiler-stack precision bug causes PCC ≈ 0.80 on TT silicon (CPU bf16 vs CPU f32 gives 0.995, confirming the loader is numerically correct).

**Loader bug 1 — wrong GGUF loading strategy** (`loader` layer):
The original loader called `AutoModelForImageTextToText.from_pretrained("noctrex/LightOnOCR-2-1B-ocr-soup-GGUF", gguf_file=...)`. The GGUF metadata declares `general.architecture: qwen3` (text-only). Transformers infers `Qwen3Config`, which `AutoModelForImageTextToText` rejects with `ValueError: Unrecognized configuration class Qwen3Config`. Additionally, `get_gguf_hf_weights_map` requires `hf_model.config.num_hidden_layers` at the top level, but `LightOnOcrConfig` nests it under `text_config`. The fix is a hybrid approach: load the full VL model (`LightOnOcrForConditionalGeneration`) from the base HF repo to get the vision encoder, load the GGUF as `Qwen3ForCausalLM` (which works because "qwen3" is in gguf's `MODEL_ARCH_NAMES`), then transplant the GGUF text-backbone weights into the VL model's `language_model`.

**Loader bug 2 — global patch missing `**kwargs`** (`loader` layer):
26 other GGUF loaders each patch `transformers.modeling_utils.load_gguf_checkpoint` at import time with `def _patched_load_gguf_checkpoint(gguf_path, return_tensors=False):` — missing `**kwargs`. Transformers 5.2 added a `model_to_load` keyword argument. When any of these loaders is imported during pytest collection (before this model's test runs), the patched function contaminates global state. The lighton_ocr loader then fails with `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'`. Fixed by cherry-picking existing fixes that add `**kwargs` to the patched signature.

**Compiler-stack precision bug** (`tt-mlir` or `tt-metal` layer):
After all loader bugs are fixed, the model runs on TT silicon but produces PCC ≈ 0.80 vs CPU reference (both in bf16). CPU bf16 vs CPU f32 yields PCC = 0.995, confirming the loader and weight transplant are numerically correct. The 0.80 gap is a compiler-stack precision regression, likely in the Pixtral 2D RoPE attention or Qwen3 attention computation. The specific failing function is unknown without deeper compiler-pass analysis; no single Tier A fix could be identified.

## Fix

**Files changed in `tt_forge_models` (remediation branch `remediation/lighton_ocr_2_1b_ocr_soup_gguf-image_to_text-pytorch-lighton_ocr_2_1b_ocr_soup_gguf-single_device-inference`):**

- `lighton_ocr_2_1b_ocr_soup_gguf/image_to_text/pytorch/loader.py`: Rewrote loader to use hybrid VL+GGUF strategy — load `LightOnOcrForConditionalGeneration` from `lightonai/LightOnOCR-2-1B-ocr-soup` for the vision encoder, load GGUF as `Qwen3ForCausalLM`, transplant `gguf_lm.model.state_dict()` into `vl_model.model.language_model`.
- 26 GGUF loader files: Added `**kwargs` to `_patched_load_gguf_checkpoint` signatures (cherry-picked from commits `edba9dfd1a` and `0479ce9268` on existing remediation branch).

**Proposed compiler-stack fix (not attempted):**
The PCC gap of ≈ 0.80 in a vision-language model running on TT silicon most likely originates in one of: (a) Pixtral 2D rotary position embedding computation, (b) Qwen3 self-attention with bfloat16 accumulation, or (c) a precision-lossy op in the StableHLO→TTIR lowering for either component. A Tier A fix would require identifying the specific lowering pass producing the precision loss — this requires compiler-pass debugging beyond the scope of a single scoped fix.

## Tier B justification

`internal-error-unknown-mechanism`: The compiler-stack precision bug (PCC ≈ 0.80) has no identified root cause. The failing op or lowering pass is unknown — diagnosis must come before a fix can be scoped. The gap could be in any of several multi-file paths (Pixtral RoPE lowering, Qwen3 attention, bf16 accumulation in matmul). No single file or function was fingered as the source.

## Verification
- pytest exit: FAIL
- Hardware:    n150
- Duration:    ~280s
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/lighton_ocr_2_1b_ocr_soup_gguf/image_to_text/pytorch/loader.py`
- 26 GGUF loader files (model_to_load kwarg fix)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 1948bcd7a942d9f2f96cf3d9352ae79169cbac90 |
| tt-forge-models | 6ca323b72db4663b987f32e91db6deda25afa5d1 |
