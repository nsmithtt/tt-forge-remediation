# Remediation Summary: jina_embeddings_v5_text_small_classification_gguf-classification-pytorch-jina-embeddings-v5-text-small-classification-GGUF-single_device-inference

## Skill version
5

## Test
tests/runner/test_models.py::test_all_models_torch[jina_embeddings_v5_text_small_classification_gguf/classification/pytorch-jina-embeddings-v5-text-small-classification-GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-wrong-filename-and-kv-cache-in-output

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
2026-04-23 21:02:27.528 | critical |          Always | TT_THROW: TIMEOUT: device timeout, potential hang detected, the device is unrecoverable (assert.hpp:104)

## Root cause
Two loader bugs compounded to produce the device timeout and PCC failure:

1. **Wrong GGUF filename** (regression introduced in submodule commit `0f7b734348`): The
   loader used `jina-embeddings-v5-text-small-classification-Q4_K_M.gguf` but the actual
   file on HuggingFace is `v5-small-classification-Q4_K_M.gguf`. The original CI branch
   (`arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-21`) had the correct
   filename; the current submodule HEAD introduced the regression.

2. **KV cache included in model output**: The model is Qwen3-based (decoder-only,
   28 layers, 8 KV heads). Without `use_cache=False`, `AutoModel.from_pretrained`
   returns a `DynamicCache` with 56 K/V tensors alongside `last_hidden_state`. The test
   evaluator takes the **minimum** PCC across all output tensors. The padding positions
   (113 out of 128 tokens from `padding="max_length"`) in the K/V cache produce large
   bfloat16 vs float32 discrepancies, bringing the minimum PCC to 0.891 (required 0.99).
   On the original CI hardware the extra 56 large device→host KV transfers caused a
   device hang/timeout rather than a PCC failure.

## Fix
Two changes in `tt_forge_models` on branch
`remediation/jina_embeddings_v5_text_small_classification_gguf-classification-pytorch-jina-embeddings-v5-text-small-classification-GGUF-single_device-inference`
(commit `914a3984adabfaa5d48dfd41b324ca1c3a854ac7`):

1. `jina_embeddings_v5_text_small_classification_gguf/classification/pytorch/loader.py`:
   - Corrected `GGUF_FILE` from `jina-embeddings-v5-text-small-classification-Q4_K_M.gguf`
     to `v5-small-classification-Q4_K_M.gguf` (matches the CI branch and HF repo).
   - Added `model.config.use_cache = False` after `AutoModel.from_pretrained` to prevent
     the 56 KV cache tensors from being included in the model output. Only `last_hidden_state`
     is returned; its PCC is 0.992 on device vs CPU, which passes the 0.99 threshold.

The fix is rebased on top of `e52ad04838` (CI branch tip) to incorporate the
`model_to_load=None` parameter fix in `_patched_load_gguf_checkpoint` from the Qwen3.5
GGUF loaders, which are imported during pytest collection and would otherwise break
GGUF loading via cross-contamination of `transformers.modeling_gguf_pytorch_utils`.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    243.37s (0:04:03)
- Tier A attempts: N/A

## Files changed
- `jina_embeddings_v5_text_small_classification_gguf/classification/pytorch/loader.py`
  (in tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 914a3984adabfaa5d48dfd41b324ca1c3a854ac7 |
