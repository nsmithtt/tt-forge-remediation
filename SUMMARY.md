# Remediation Summary: llama_3_2_1b_sft_full_gguf-causal_lm-pytorch-3.2_1B_SFT_Full_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[llama_3_2_1b_sft_full_gguf/causal_lm/pytorch-3.2_1B_SFT_Full_GGUF-single_device-inference]

## Result
SILICON_PASS

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-get-weights-map-num-layers-positional-arg

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: _patch_transformers_llama4_gguf.<locals>._patched_get_gguf_hf_weights_map() takes from 2 to 3 positional arguments but 4 were given

## Root cause
The `lmstudio_llama_4_scout_17b_16e_instruct_gguf` loader applies a global monkey-patch to `transformers.modeling_gguf_pytorch_utils.get_gguf_hf_weights_map` at import time.  Its patched function was defined as `(hf_model, processor, model_type=None, **kwargs)`, which accepts at most 3 positional arguments (`**kwargs` is keyword-only).

transformers 5.2.0 calls `get_gguf_hf_weights_map` recursively for each child module with 4 positional arguments:
```python
get_gguf_hf_weights_map(child, processor, model_type, num_layers, qual_name=f"...")
```

When pytest discovers all model tests at collection time, every loader module is imported via `importlib`, including the llama4 loader.  In CI runs where the llama4 loader happened to be imported last (or later than the `mistral_community_pixtral_12b_gguf` loader that has a correct 4-arg signature), the broken llama4 patch remained the active replacement, causing any GGUF model test that followed — including `llama_3_2_1b_sft_full_gguf` — to fail with the TypeError during model loading.

## Fix
Added `num_layers=None` as the 4th positional parameter to `_patched_get_gguf_hf_weights_map` and forwarded it to the original via keyword argument.

**File:** `lmstudio_llama_4_scout_17b_16e_instruct_gguf/causal_lm/pytorch/loader.py`

```python
# Before
def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, **kwargs
):
    ...
    return _orig_get_weights_map(
        hf_model, processor, model_type=model_type, **kwargs
    )

# After
def _patched_get_gguf_hf_weights_map(
    hf_model, processor, model_type=None, num_layers=None, **kwargs
):
    ...
    return _orig_get_weights_map(
        hf_model, processor, model_type=model_type, num_layers=num_layers, **kwargs
    )
```

Remediation branch: `remediation/llama_3_2_1b_sft_full_gguf-causal_lm-pytorch-3.2_1B_SFT_Full_GGUF-single_device-inference` in `tt-forge-models`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    269.38s (0:04:29)
- Tier A attempts: N/A

## Files changed
- `lmstudio_llama_4_scout_17b_16e_instruct_gguf/causal_lm/pytorch/loader.py` (tt-forge-models)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 32a68e3dc1a6ee4d7708b1d4bc161f838eef603d |
