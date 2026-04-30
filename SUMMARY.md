# Remediation Summary: gemma3n_gguf-causal_lm-pytorch-Gemma_3n_E4B_IT_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[gemma3n_gguf/causal_lm/pytorch-Gemma_3n_E4B_IT_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader, tt-xla

## Tier
A

## Bug fingerprint
gguf-gemma3n-architecture-missing, aten-slice-tensor-out-of-bounds-start

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise NotImplementedError(
```
Expanded: `NotImplementedError: Model gemma3n is not supported. Supported architectures are: ...` from `get_gguf_hf_weights_map` in `transformers/modeling_gguf_pytorch_utils.py` at the line that checks `MODEL_ARCH_NAMES`.

After the loader fix, a second failure surfaced:
```
RuntimeError: Value out of range (expected to be in range of [-23, 22], but got -511)
```
from `aten.slice.Tensor` in the XLA lazy backend, triggered by `SlidingWindowCache.update()`.

## Root cause

**Bug 1 — loader layer**: `gemma3n` is registered in gguf-py's `MODEL_ARCH_NAMES` but not in transformers 5.2.0's GGUF registration tables (`GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, `TENSOR_PROCESSORS`). When `get_gguf_hf_weights_map` looks up the architecture, it raises `NotImplementedError`.

A secondary complication: 26 other GGUF loaders in the test suite (bartowski_*, daniloreddy_*, dmind_*, etc.) monkey-patch `load_gguf_checkpoint` at import time with a fixed-signature wrapper `_patched_load_gguf_checkpoint(gguf_path, return_tensors=False)` that drops the `model_to_load` keyword argument required by transformers 5.x. By the time the gemma3n model is loaded in the test session, the global `load_gguf_checkpoint` binding points to one of these broken wrappers, causing `TypeError`.

Additionally, `configuration_utils.py` imports `load_gguf_checkpoint` at module level (line 29), not inside the function, so `AutoConfig.from_pretrained` uses a stale binding separate from `modeling_gguf_pytorch_utils.load_gguf_checkpoint`. Without patching `configuration_utils.load_gguf_checkpoint`, the config loads with `model_type="gemma3n"` instead of `"gemma3n_text"`, causing `Gemma3nForConditionalGeneration` (the multimodal class) to be instantiated instead of `Gemma3nForCausalLM`, with mismatched tensor sizes.

**Bug 2 — tt-xla (Tier A)**: Gemma3n uses `SlidingWindowCache` with `sliding_window=512`. In `SlidingWindowCache.update()` (transformers cache_utils.py), the cache is updated via `full_value_states[:, :, -self.sliding_window + 1:, :]`. With test inputs of `seq_len=23`, the start index is `-511`, which is less than `-seq_len = -23`. PyTorch eager silently clamps out-of-range slice indices, but the XLA lazy backend validates bounds strictly and raises `RuntimeError: Value out of range`.

## Fix

**Loader fix** (`tt-xla/third_party/tt_forge_models/gemma3n_gguf/causal_lm/pytorch/loader.py`):

1. `_register_gemma3n_gguf_support()` — called at import time (idempotent): registers `gemma3n` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]` (with the same field mappings as gemma3), `TENSOR_PROCESSORS` (using `Gemma2TensorProcessor`), and `GGUF_TO_FAST_CONVERTERS` (aliased to `gemma3_text` converter).

2. `_find_real_load_gguf_checkpoint(fn)` — walks the broken patch chain by traversing `__closure__` (for wrappers that capture the original via local variable) and `__globals__['_orig_load_gguf_checkpoint']` (for wrappers that capture it via module-level alias) until finding the genuine transformers function (identified by `__name__ == 'load_gguf_checkpoint'` and `'transformers' in __module__`).

3. `_gemma3n_gguf_load_context()` — context manager applied inside `load_model()`, `_load_tokenizer()`, and `load_config()`. At call time it temporarily replaces the `load_gguf_checkpoint` binding in `modeling_gguf_pytorch_utils`, `configuration_utils`, and `tokenization_auto` with a correct wrapper that: (a) calls the real transformers function directly, bypassing all broken wrappers; (b) remaps `model_type "gemma3n" → "gemma3n_text"` in the returned config dict so `AutoModelForCausalLM` instantiates `Gemma3nForCausalLM` instead of the multimodal class. Also patches `get_gguf_hf_weights_map` to remap `"gemma3n_text" → "gemma3n"` for the internal gguf-py weight-map lookup.

**tt-xla fix** (`tt-xla/python_package/tt_torch/torch_overrides.py`):

In `TorchFunctionOverride.__torch_function__`, added a guard for `func is torch.ops.aten.slice.Tensor` that clamps `start` and `end` arguments to `[-size, ...]` when they are more negative than `-size`. This matches PyTorch eager's silent clamping behavior and prevents the XLA lazy backend's strict range validation from raising `RuntimeError`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    1192.42s (0:19:52)
- Tier A attempts: 1

## Files changed
- `tt-xla/third_party/tt_forge_models/gemma3n_gguf/causal_lm/pytorch/loader.py` (loader layer — full rewrite)
- `tt-xla/python_package/tt_torch/torch_overrides.py` (tt-xla — Tier A slice bounds clamping)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 77a46c9fa641a49d70296e833ba7a0785ba26c2a |
| tt-forge-models | 2277cad55bc1d29a454fce1dd1dfabe9d3056f09 |
