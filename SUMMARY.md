# Remediation Summary: cait/pytorch-M36_384_FB_DIST_IN1K-single_device-inference

## Skill version
2

## Test
tests/runner/test_models.py::test_all_models_torch[cait/pytorch-M36_384_FB_DIST_IN1K-single_device-inference]

## Result
FAIL — SDPA decode kernel requires k_chunk_size % 32 == 0, but gets 2 for the 577-token K/V sequence in CaiT class attention

## Failure
```
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
Underlying cause:
```
TT_FATAL: Chunk size must be multiple of 32, but the maximum calculated k_chunk_size is: 2
(tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:66)
```

## Root cause

Two issues, one in the loader layer and one in the runtime layer.

**Loader (tt_forge_models):** `huspacy/pytorch/loader.py` had a top-level `import spacy`. During pytest collection, `models_root` (= `tt_forge_models`) is inserted into `sys.path`. The directory `tt_forge_models/spacy/` exists (for the spacy/es_core_news_md model) and Python's namespace-package mechanism registers it in `sys.modules["spacy"]` — a namespace package with no `Language` attribute. Later, when the cait loader calls `load_dataset("huggingface/cats-image")`, the `datasets` library's dill pickler checks `if "spacy" in sys.modules` → True, imports spacy, and tries `spacy.Language` → `AttributeError`. This blocked reproduction of the original error.

**Runtime (tt-metal):** After fixing the loader issue, the original error reproduces. CaiT's class-attention blocks (`blocks_token_only`) compute attention with Q from 1 class token and K/V from all 577 tokens (576 patches + 1 class token for a 384×384 image with patch_size=16). The compiler selects `scaled_dot_product_attention_decode` for this pattern (q_seq_len=1 << kv_seq_len=577). The `get_chunk_size(s)` function in `sdpa_decode.cpp` finds the largest power-of-2 divisor of `s=577`:

```cpp
inline uint32_t get_chunk_size(uint32_t s) {
    uint32_t i = 1;
    for (; i < s; i++) {
        if (s % (1 << (i + 1)) != 0) break;
    }
    return std::min(512, 1 << i);
}
```

For s=577 (odd), `577 % 4 != 0` at i=1 → returns `min(512, 2) = 2`. The kernel then asserts `k_chunk_size % 32 == 0`, which 2 fails.

## Fix

**Loader fix (applied):** In `huspacy/pytorch/loader.py`, moved `import spacy` from module level into `_load_nlp()` so it is lazy. This prevents the `tt_forge_models/spacy` namespace package from shadowing the real spacy during test collection.

**Runtime fix (proposed, not applied):** The bug lives in `tt-metal`. Options:
1. In `get_chunk_size`: if the result is less than 32, return 32 and have the SDPA decode kernel pad the K/V input to the next multiple of `k_chunk_size` before processing.
2. In tt-mlir: when lowering class-attention-style ops (where q_seq_len is very small), pad K/V to a multiple of 32 before emitting `ttnn.sdpa_decode`.
3. Use `ttnn.sdpa` (prefill mode) instead of `ttnn.sdpa_decode` for K/V sequences that are not divisible by 32 — the prefill path does not have this alignment constraint.

## Verification
Pytest exit status: FAILED. No silicon pass recorded; fix for the runtime layer is out of scope for the loader.

Wall-clock duration at failure: ~70 s (compilation ~66 s + kernel launch ~4 s).

Hardware: n300.

## Files changed
- `huspacy/pytorch/loader.py` in tt-forge-models: moved `import spacy` from module level into `_load_nlp()`.

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3e0801d63f8c5c6ecba7706fcd16ab1f3bbcd5ab |
| tt-forge-models | c9afa819830c6dec582acfed4a561917d9db8b03 |
