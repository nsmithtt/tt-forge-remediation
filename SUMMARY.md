# Remediation Summary: lfm2_5_1_2b_thinking_kimi_v2_distill_gguf-causal_lm-pytorch-LFM2_5_1_2B_THINKING_KIMI_V2_DISTILL_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[lfm2_5_1_2b_thinking_kimi_v2_distill_gguf/causal_lm/pytorch-LFM2_5_1_2B_THINKING_KIMI_V2_DISTILL_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg, lfm2-gguf-tokenizer-converter-missing, lfm2-hybrid-conv-cache-evaluator-duck-type

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: equal(): argument 'input' (position 1) must be Tensor, not Lfm2HybridConvCache

Three stacked loader-layer failures preceded the originally reported error:

1. `KeyError: 'lfm2'` — GGUF_TO_FAST_CONVERTERS missing entry for "lfm2" architecture; exposed because `iquest_coder` loader patches `convert_gguf_tokenizer` and the patched version calls the original which fails.
2. `TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'` — 26 loaders patch `load_gguf_checkpoint` with the old signature `(gguf_path, return_tensors=False)` and don't forward `model_to_load` introduced in transformers 5.2.0.
3. The originally reported `TypeError: equal()... must be Tensor, not Lfm2HybridConvCache` — evaluator's `isinstance(tensor, Cache)` check misses `Lfm2HybridConvCache` which is not a `transformers.Cache` subclass.

## Root cause
All three failures are in the loader layer:

1. **Missing GGUF tokenizer converter**: The `lfm2` GGUF architecture uses a GPT2-style BPE tokenizer (`tokenizer.ggml.model = "gpt2"`) but is not registered in `GGUF_TO_FAST_CONVERTERS` in transformers 5.x. When any loader that patches `convert_gguf_tokenizer` is imported first (here: `iquest_coder`), its wrapper calls the original function which then fails with `KeyError: 'lfm2'`.

2. **Patcher chain `model_to_load` kwarg**: 26 tt-forge-models loaders define `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` and forward only those two args to `_orig_load_gguf_checkpoint`. Transformers 5.2.0 added `model_to_load` to this call, which propagates through the chain and hits the broken patcher signature.

3. **Evaluator duck-type**: `Lfm2HybridConvCache` (LFM2's hybrid conv/attention cache) has `key_cache`/`value_cache` attributes but does not subclass `transformers.Cache`. The `isinstance(tensor, Cache)` guard in `_match_data_types` does not fire; `tree_map` treats the object as an opaque leaf; `torch.equal()` then fails with TypeError.

## Fix
Three independent fixes:

**1. tt_forge_models — register lfm2 GGUF tokenizer converter** (`lfm2_5_1_2b_thinking_kimi_v2_distill_gguf/causal_lm/pytorch/loader.py`):
```python
from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, GGUFGPTConverter
GGUF_TO_FAST_CONVERTERS.setdefault("lfm2", GGUFGPTConverter)
```

**2. tt_forge_models — fix _patched_load_gguf_checkpoint in 26 loaders** (bulk sed over all affected `loader.py` files):
Changed signature from `(gguf_path, return_tensors=False)` to `(*args, **kwargs)` and updated the forwarded call from `_orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)` to `_orig_load_gguf_checkpoint(*args, **kwargs)`.

**3. tt-xla — evaluator duck-type for Lfm2HybridConvCache** (`tests/infra/evaluators/torch_comparison_evaluator.py`):
- `_cache_to_legacy`: added duck-type branch for objects with `key_cache`/`value_cache`; filters `numel()==0` placeholders to avoid `torch.max()` on empty tensors.
- `convert_and_match`: expanded guard to `isinstance(tensor, Cache) or (hasattr(tensor, "key_cache") and hasattr(tensor, "value_cache"))`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    190.87s (0:03:10)
- Tier A attempts: N/A

## Files changed
- `tt-xla/tests/infra/evaluators/torch_comparison_evaluator.py`
- `tt_forge_models/lfm2_5_1_2b_thinking_kimi_v2_distill_gguf/causal_lm/pytorch/loader.py`
- 26 `tt_forge_models/*/causal_lm/pytorch/loader.py` files with broken `_patched_load_gguf_checkpoint` signature

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 3c522d2440c63b9c976d4d5b7d276825bf633a8b |
| tt-forge-models | 4ae8126c6d64e277aa984a5debfdba173c7e8277 |
