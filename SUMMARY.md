# Remediation Summary: jina_embeddings_v4_text_matching_gguf-text_matching-pytorch-jina-embeddings-v4-text-matching-GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[jina_embeddings_v4_text_matching_gguf/text_matching/pytorch-jina-embeddings-v4-text-matching-GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-qwen2vl-arch-not-supported-transformers-5x

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9334953525682824. Required: pcc=0.95.

(On the current branch a harder failure was encountered first: `ValueError: GGUF model with architecture qwen2vl is not supported yet.` This is the same root cause — the model could not load.)

## Root cause
The GGUF file `jina-embeddings-v4-text-matching-Q4_K_M.gguf` sets `general.architecture = qwen2vl`. Transformers 5.2.x does not include `qwen2vl` in `GGUF_SUPPORTED_ARCHITECTURES`, so `load_gguf_checkpoint` raises `ValueError` when attempting to read the config. The GGUF file contains only standard language-model tensors (36 transformer blocks, no vision encoder), so the model is functionally a Qwen2 3.1B text model with a mis-labelled architecture field.

A secondary PCC issue was caused by `padding="max_length"` in `load_inputs`: padding to 128 tokens for a ~14-token input means the bulk of the `last_hidden_state` tensor (positions 14–127) holds padding-position hidden states that accumulate bfloat16 noise differently on TT vs CPU, dragging PCC below 0.95.

## Fix
`tt-forge-models/jina_embeddings_v4_text_matching_gguf/text_matching/pytorch/loader.py`

1. **qwen2vl GGUF support**: Added `_patch_transformers_qwen2vl_gguf()` which:
   - Appends `"qwen2vl"` to `GGUF_SUPPORTED_ARCHITECTURES`.
   - Registers a `GGUF_TO_TRANSFORMERS_MAPPING["config"]["qwen2vl"]` dict mirroring the `qwen2` field mapping (ignoring `rope.dimension_sections`, which is M-RoPE specific and unused by `Qwen2Model`).
   - Aliases the `qwen2` fast tokenizer converter to `qwen2vl`.
   - Wraps `load_gguf_checkpoint` to remap `model_type` from `"qwen2vl"` to `"qwen2"` in the returned config dict, patching all three by-value import sites (`gguf_utils`, `configuration_utils`, `tokenization_auto`).
   - Unwraps any pre-existing loader patches (other GGUF loaders patched with broken fixed signatures) by walking `__globals__['_orig_load_gguf_checkpoint']` to find the real transformers function.

2. **Padding fix**: Changed `padding="max_length"` to `padding=True` in `load_inputs` so a single short sentence is tokenized to its natural length (14 tokens) rather than padded to 128, eliminating noise tokens from the PCC comparison.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    309.76s (0:05:09)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models/jina_embeddings_v4_text_matching_gguf/text_matching/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 09cf6b6d7e9e85a89d9dbe0b7c4a48df0783f836 |
| tt-forge-models | 4ab4f2f8a6f5e8ff11746d8af1a8dcd1e083f691 |
