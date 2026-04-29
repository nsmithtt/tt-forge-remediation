# Remediation Summary: bge_reranker_v2_5_gemma2-passage_ranking-pytorch-lightweight-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bge_reranker_v2_5_gemma2/passage_ranking/pytorch-lightweight-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
transformers-5x-gemma2-api-breaking-changes

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ImportError: cannot import name 'Gemma2FlashAttention2' from 'transformers.models.gemma2.modeling_gemma2'
```

The reported failure message (`sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute`) is a pytest exit-line warning; the real error cascade was a chain of five transformers 5.x breaking changes in the model's custom remote code (`BAAI/bge-reranker-v2.5-gemma2-lightweight`).

## Root cause

The model ships a custom `gemma_model.py` (loaded via `trust_remote_code=True`) that was written against the old transformers 4.x/early-5.x Gemma2 API. Five breaking changes in transformers 5.x affect it:

1. **Missing attention subclasses**: `Gemma2FlashAttention2`, `Gemma2SdpaAttention`, `GEMMA2_ATTENTION_CLASSES` were removed (unified into a single `Gemma2Attention` with `ALL_ATTENTION_FUNCTIONS`).

2. **Missing docstring constants**: `GEMMA2_START_DOCSTRING` and `GEMMA2_INPUTS_DOCSTRING` were removed (replaced by `@auto_docstring`).

3. **`_tied_weights_keys` format changed from list to dict**: The custom `CostWiseGemmaForCausalLM` defines `_tied_weights_keys = ["lm_head.weight"]` (list), but transformers 5.x `post_init()` now calls `get_expanded_tied_weights_keys()` which expects a dict (`{target: source}`). Additionally, when `layer_wise=True`, `lm_head` is a `ModuleList` with no `.weight`, so weight tying must be disabled entirely.

4. **RoPE computation moved out of `Gemma2Attention`**: In transformers 5.x, `Gemma2Model.forward` computes `position_embeddings = self.rotary_emb(hidden_states, position_ids)` and passes the `(cos, sin)` tuple to each `Gemma2DecoderLayer`. The custom `CostWiseGemmaModel` was written for the old API where RoPE was computed inside the attention layer, so it never passes `position_embeddings`. `Gemma2Attention.forward` then tries `cos, sin = None` and fails.

5. **`Gemma2DecoderLayer.forward` return type changed from tuple to plain tensor**: The custom model does `hidden_states = layer_outputs[0]` expecting the old `(hidden_states, ...)` tuple return. With the new plain-tensor return, `layer_outputs[0]` indexes the first batch element rather than the first tuple slot, squeezing the batch dimension and causing downstream shape mismatches in attention.

## Fix

All fixes are applied as loader-level compat patches in `bge_reranker_v2_5_gemma2/passage_ranking/pytorch/loader.py` before `AutoModelForCausalLM.from_pretrained` is called. Four patches are applied:

1. **Import stubs**: Inject the five missing names into `transformers.models.gemma2.modeling_gemma2` via `setattr` (guarded by `hasattr` so they are no-ops once transformers adds them back).

2. **`_tied_weights_keys` fix**: Pre-import `CostWiseGemmaForCausalLM` via `get_class_from_dynamic_module`, then rewrite the class attribute from the list format to a dict (or empty dict when `layer_wise=True`).

3. **RoPE position embeddings**: Patch `Gemma2DecoderLayer.forward` to lazily instantiate a `Gemma2RotaryEmbedding` per layer and compute `position_embeddings` when not provided. Guard: only activates when `position_embeddings is None`, so native `Gemma2Model` usage is unaffected.

4. **Tuple return wrapper**: The same `Gemma2DecoderLayer.forward` patch wraps a plain-tensor result in a 1-tuple `(result,)` so the custom model's `layer_outputs[0]` receives `hidden_states`, not the first batch element.

Remediation branch: `remediation/bge_reranker_v2_5_gemma2-passage_ranking-pytorch-lightweight-single_device-inference` in tt-forge-models.

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    120.15s (0:02:00)
- Tier A attempts: N/A

## Files changed
- tt-forge-models: `bge_reranker_v2_5_gemma2/passage_ranking/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 482a96445a42d081f293c8d02f3f9f7a32189ac4 |
| tt-forge-models | 3961a4720c367b47c01a63f403e76e8a4887bb21 |
