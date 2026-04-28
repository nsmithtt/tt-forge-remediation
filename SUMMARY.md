# Remediation Summary: gui_libra_8b_i1_gguf-image_to_text-pytorch-gui_libra_8b_i1_gguf-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[gui_libra_8b_i1_gguf/image_to_text/pytorch-gui_libra_8b_i1_gguf-single_device-inference]

## Result
FAIL — GGUF file (mradermacher/GUI-Libra-8B-i1-GGUF) does not include vision encoder weights; `qwen3vl` architecture is not registered in transformers GGUF support

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-qwen3vl-arch-not-registered-vision-encoder-missing

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: GGUF model with architecture qwen3vl is not supported yet.
```

(The CI-reported failure `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`
was likely from a run where the GGUF load path was handled differently, possibly with
a modified transformers version or by loading from the base HF repo, and the 8B model
then OOM'd on n150 hardware.)

## Root cause

Two compounding loader bugs prevent this test from ever passing:

**Bug 1 — qwen3vl GGUF architecture not registered.** The GGUF file's
`general.architecture` field is `"qwen3vl"`. The installed transformers library
(`modeling_gguf_pytorch_utils.py`) only registers `qwen2`, `qwen2_moe`, `qwen3`,
and `qwen3_moe`. When `Qwen3VLForConditionalGeneration.from_pretrained("mradermacher/
GUI-Libra-8B-i1-GGUF", gguf_file="GUI-Libra-8B.i1-Q4_K_M.gguf", ...)` is called,
`load_gguf_checkpoint` reads `general.architecture = "qwen3vl"`, finds it absent from
`GGUF_SUPPORTED_ARCHITECTURES`, and raises `ValueError`.

**Bug 2 — GGUF file ships only language-model weights, not the vision encoder.** Inspection
of the cached GGUF file confirms it contains exactly 399 tensors, all under the prefixes
`blk.*` (396, corresponding to 36 text-decoder layers × 11 params each), `output.weight`,
`output_norm.weight`, and `token_embd.weight`. There are no vision-encoder tensors
(`v_enc.*`, `mm_proj.*`, etc.). Even if Bug 1 were fixed by registering `qwen3vl` in
`GGUF_SUPPORTED_ARCHITECTURES` with the correct config-field mapping, the loaded
`Qwen3VLForConditionalGeneration` would have randomly-initialized vision encoder weights.
An `image_to_text` test that routes image inputs through this random encoder would
produce garbage visual features and PCC ≈ 0.

Per the skill rules: *"The GGUF doesn't ship the encoder is not a justification — file
an issue and report failure."* Switching to text-only inputs to bypass the visual encoder
is a forbidden workaround; lowering `required_pcc` to accommodate random-encoder output
is likewise forbidden.

Additionally, `qwen_3_vl/image_to_text` non-GGUF models (2B, 4B) are already marked
`KNOWN_FAILURE_XFAIL` in `test_config_inference_single_device.yaml` due to
`RuntimeError: Check failed: handle->HasValue(): Trying to access XLA data for tensor
with ID 645 while an async operation is in flight: UNKNOWN_SCALAR[]` (Issue #3184).
Any successful GGUF load of the 8B VLM would likely surface the same compiler bug or
OOM on n150 hardware (12 GB DRAM < 16 GB required for 8B bf16 text decoder alone).

## Fix
No fix applied. To fully remediate this test, two things would be needed:

1. Register `qwen3vl` in `GGUF_SUPPORTED_ARCHITECTURES` in the loader, with a config-field
   mapping reusing the `qwen3` text-decoder fields, and patch `model_type` from `"qwen3vl"`
   to `"qwen3_vl"` after `load_gguf_checkpoint` returns.

2. Load the vision encoder weights from the base model (`GUI-Libra/GUI-Libra-8B` or
   `Qwen/Qwen3-VL-8B-Instruct`) in full precision, then copy them into the GGUF-loaded
   model before inference. This is a hybrid GGUF + safetensors loading pattern not
   currently implemented in any loader in tt_forge_models.

Even with both fixes, the Qwen3-VL compiler bug (Issue #3184, `UNKNOWN_SCALAR[]`) would
likely prevent SILICON_PASS on the current branch, and the 8B model would OOM on n150.

## Verification
- pytest exit: FAIL
- Hardware:    not-run
- Duration:    76.21s (to reproduce local ValueError)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 0f7b734348f0053b9bf8d04a6ae42662af5f425c |
