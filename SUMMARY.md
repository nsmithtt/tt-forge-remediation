# Silicon Pass: bat_venom/causal_lm/pytorch-BatVenom-V7

## Test

```
tests/runner/test_models.py::test_all_models_torch[bat_venom/causal_lm/pytorch-BatVenom-V7-single_device-inference]
```

## Status: SILICON_PASS (540.93s)

## Root Cause

Three issues were present in the bat_venom model loader:

1. **GGUF-only model loaded as standard HuggingFace checkpoint** — `BrainDelay/BatVenom-V7` is a GGUF-only model. The original loader used bare `AutoConfig.from_pretrained` and `AutoTokenizer.from_pretrained` without specifying `gguf_file`, causing:
   ```
   ValueError: Couldn't instantiate the backend tokenizer from one of: ...
   You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
   ```
   and:
   ```
   ValueError: Unrecognized model in BrainDelay/BatVenom-V7. Should have a model_type key in its config.json.
   ```

2. **GGUF version detection broken** — `transformers` caches `PACKAGE_DISTRIBUTION_MAPPING` at import time. When `gguf` is installed later by `RequirementsManager`, the mapping is stale and version detection returns `'N/A'`, crashing `packaging.version.parse`.

3. **`load_gguf_checkpoint` patch incompatibility with `model_to_load` kwarg** — `transformers` 5.x added a `model_to_load` keyword argument to `load_gguf_checkpoint` but the loader's wrapper didn't accept it:
   ```
   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
   ```

The original OOM error from CI (`TT_FATAL: Out of Memory: Not enough space to allocate 146800640 B DRAM buffer across 12 banks`) was resolved by using the GGUF Q4_K_M quantized format which uses significantly less memory than full precision.

## Fix

All fixes are in `tt_forge_models` branch `remediation/bat-venom-v7-gguf-fix`:

1. **GGUF format**: Added `gguf_file` parameter to all `from_pretrained` calls; added `gguf>=0.10.0` to `requirements.txt`; mapped variants to their GGUF filenames (`Mistral-BatVenom_V7.2_Q4_K_M.gguf` for V7)
2. **Version detection**: Added `_fix_gguf_version_detection()` to patch `PACKAGE_DISTRIBUTION_MAPPING` and clear `is_gguf_available` LRU cache
3. **`model_to_load` kwarg**: Added `_find_real_load_gguf_checkpoint()` chain traversal and patched wrapper accepting `model_to_load`
4. **Tokenizer**: Removed `apply_chat_template` (no chat template in GGUF tokenizer); directly tokenize sample text

## Submodule Hashes

| Submodule | Hash |
|-----------|------|
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla | 1e8d6e253 (remediation/bat-venom-v7-gguf-fix) |
| tt-xla/third_party/tt_forge_models | eb89e14645 (remediation/bat-venom-v7-gguf-fix) |

## Key Commits in tt_forge_models (branch remediation/bat-venom-v7-gguf-fix)

```
eb89e14645 Fix BatVenom-V7 GGUF: add model_to_load kwarg patcher
01c820463a Fix BatVenom-V7: convert loader to GGUF format and add gguf dependency
```
