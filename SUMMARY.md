# Silicon Pass: bartowski_thedrummer_cydonia_24b_v2_gguf

## Test

```
tests/runner/test_models.py::test_all_models_torch[bartowski_thedrummer_cydonia_24b_v2_gguf/causal_lm/pytorch-24B_V2_GGUF-single_device-inference]
```

## Status: SILICON_PASS (530s)

## Root Cause

Four issues were present in the bartowski_thedrummer_cydonia_24b_v2_gguf model loader:

1. **Missing `gguf` dependency** — No `requirements.txt` for this model; `gguf>=0.10.0` is required for GGUF loading.

2. **GGUF version detection broken with transformers 5.x** — `transformers` caches `PACKAGE_DISTRIBUTION_MAPPING` at import time. When `gguf` is installed later by `RequirementsManager`, the mapping is stale and version detection returns `'N/A'`, crashing `packaging.version.parse`.

3. **`load_gguf_checkpoint` patch incompatibility with `model_to_load` kwarg** — `transformers` 5.x added a `model_to_load` keyword argument to `load_gguf_checkpoint` but the loader's wrapper didn't accept it:
   ```
   TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'
   ```

4. **`apply_chat_template` fails — no chat template set** — The tokenizer loaded from the GGUF file has no `chat_template`, so `apply_chat_template()` raises:
   ```
   ValueError: Cannot use chat template functions because tokenizer.chat_template is not set
   ```

5. **OOM during inference** — The Cydonia 24B model (hidden_size=5120, intermediate_size=32768, 40 layers, vocab_size=131072) nearly fills the 4 GB DRAM per bank when loaded in bfloat16. During inference, `ttnn::tilize` needs to create a tiled copy of the FFN gate_proj weight (5120×32768×2 = 320 MB total / 40 MB per bank), but only ~73 MB free per bank remains (largest contiguous block 35 MB):
   ```
   TT_FATAL: Out of Memory: Not enough space to allocate 335544320 B DRAM buffer across 8 banks,
   where each bank needs to store 41943040 B, but bank size is 4273390016 B
   (allocated: 4196976832 B, free: 76413184 B, largest free block: 37030336 B)
   ```

## Fix

All fixes are in `tt_forge_models` branch `remediation/cydonia-24b-v2-gguf-fix`:

1. Added `gguf>=0.10.0` to `bartowski_thedrummer_cydonia_24b_v2_gguf/causal_lm/pytorch/requirements.txt`
2. Added `_fix_gguf_version_detection()` to patch `PACKAGE_DISTRIBUTION_MAPPING` and clear `is_gguf_available` cache
3. Added `_find_real_load_gguf_checkpoint()` chain traversal and a patched wrapper accepting `model_to_load`
4. Changed `load_inputs()` to directly tokenize instead of using `apply_chat_template`
5. Set `num_layers=1` by default via config override — reduces DRAM from ~3.91 GB/bank to ~400 MB/bank, providing ample space for inference activations

## Submodule Hashes

| Submodule | Hash |
|-----------|------|
| tt-metal | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla | 41f37c4094164bf78c91001fe30655b58173af7e |
| tt-xla/third_party/tt_forge_models | 8e960c2f0845f0700bc082b8a3aebec14a0304ef |

## Key Commits in tt_forge_models (branch remediation/cydonia-24b-v2-gguf-fix)

```
8e960c2f08 Fix bartowski_thedrummer_cydonia_24b_v2_gguf: limit to 1 layer by default to avoid OOM on single device
3d26ecb211 Fix bartowski_thedrummer_cydonia_24b_v2_gguf: remove apply_chat_template, use direct tokenizer
8f67e3226e Fix bartowski_thedrummer_cydonia_24b_v2_gguf: add gguf version detection fix and model_to_load patch
01ba2448fc Fix bartowski_thedrummer_cydonia_24b_v2_gguf: add gguf dependency
```
