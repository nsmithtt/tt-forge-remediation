# Remediation Summary: granite_4_0_350m_gguf-causal_lm-pytorch-granite_4_0_350m_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[granite_4_0_350m_gguf/causal_lm/pytorch-granite_4_0_350m_Q4_K_M_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-load-checkpoint-model-to-load-kwarg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
AttributeError: 'NoneType' object has no attribute 'config'

## Root cause
Three loader bugs in the `granite_4_0_350m_gguf` loader, all in the loader layer:

1. **Missing GGUF architecture registration**: transformers 5.x does not include
   `granite` in `GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING["config"]`,
   or `GGUF_TO_FAST_CONVERTERS`. `load_gguf_checkpoint` raised `ValueError:
   GGUF model with architecture granite is not supported yet`, which propagated as
   `AttributeError: 'NoneType' object has no attribute 'config'` at the call site
   (`modeling_utils.py:4016` passes `model_to_load=dummy_model`; when the call fails
   the returned dict has no `config` key and the caller crashes on `result["config"]`).

2. **Per-layer num_key_value_heads array**: `granite.attention.head_count_kv` is stored
   as a 28-element array `[4, 4, ..., 4]` in the GGUF file. `GraniteConfig` expects
   a scalar; the array caused `TypeError: unsupported operand type(s) for //: 'int'
   and 'list'` in `GraniteDecoderLayer.__init__`.

3. **Broken load_gguf_checkpoint patch chain**: 26+ other loaders in the test session
   overwrite `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import
   time with wrappers that drop the `model_to_load` keyword argument. Some wrappers
   are module-level functions (capturing the original via
   `from ... import load_gguf_checkpoint as _orig_load_gguf_checkpoint`) while others
   are nested closures (capturing via `orig_load = gguf_utils.load_gguf_checkpoint`).
   `modeling_utils.py:4016` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`;
   a bad-signature wrapper in the chain raised `TypeError: unexpected keyword argument
   'model_to_load'`.

## Fix
All fixes in `tt_forge_models` repo, file
`granite_4_0_350m_gguf/causal_lm/pytorch/loader.py`, on branch
`remediation/granite_4_0_350m_gguf-causal_lm-pytorch-granite_4_0_350m_Q4_K_M_GGUF-single_device-inference`.

1. Added `_patch_transformers_granite_gguf()` called at module import time. Appends
   `"granite"` to `GGUF_SUPPORTED_ARCHITECTURES`, adds the full GGUF→HF config field
   mapping to `GGUF_TO_TRANSFORMERS_MAPPING["config"]["granite"]`, and registers
   `GGUFGPTConverter` in `GGUF_TO_FAST_CONVERTERS["granite"]`.

2. Added `_granite_gguf_load_context()` context manager wrapping all `from_pretrained`
   calls. The wrapper post-processes the loaded config: when `model_type == "granite"`
   and `num_key_value_heads` is a list, replaces it with `max(kv)`.

3. `_granite_gguf_load_context()` also installs a correct-signature
   `(gguf_path, return_tensors=False, **kwargs)` wrapper at all four module binding
   sites (`_gguf_utils`, `_config_utils`, `_auto_tokenizer`, `_tok_utils`), bypassing
   the broken chain. The wrapper calls `_find_real_load_gguf()` which walks the patch
   chain by inspecting both `fn.__globals__` (for module-level captured originals) and
   `fn.__closure__` / `fn.__code__.co_freevars` (for nested-function closure captures),
   stopping when `fn.__module__ == "transformers.modeling_gguf_pytorch_utils"`.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    183.09s (0:03:03)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models`: `granite_4_0_350m_gguf/causal_lm/pytorch/loader.py`

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 87b33e4eb8db8f9fc39cb1f7e09894601586c448 |
| tt-forge-models | af02c412a39abb8345768915a5fe1b7c45e78aea |
