# Remediation Summary: humanizer_styles_gguf-causal_lm-pytorch-STYLE_PRACTITIONER-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[humanizer_styles_gguf/causal_lm/pytorch-STYLE_PRACTITIONER-single_device-inference]

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
ImportError: Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.

(In environments where gguf is installed, the secondary error is:
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load')

## Root cause
Two loader bugs in `humanizer_styles_gguf/causal_lm/pytorch/`:

1. **Missing requirements.txt**: No `requirements.txt` existed beside `loader.py`. In CI environments where `gguf` is not pre-installed, `is_gguf_available()` returns False inside transformers' `load_gguf_checkpoint`, raising `ImportError`.

2. **Broken patcher chain**: Other GGUF loaders (e.g. `dmind_3_mini_i1_gguf`, `mradermacher_qwen3_5_4b_abliterated_i1_gguf`, `tvall43_qwen3_5_4b_heretic_v2_i1_gguf`) monkey-patch `load_gguf_checkpoint` at module import time with signatures missing `model_to_load`. In a full pytest session, these loaders are imported during collection before this test runs. Transformers 5.x calls `load_gguf_checkpoint(..., model_to_load=dummy_model)`, which the bad patcher rejects with TypeError. The humanizer_styles_gguf loader does not install its own patcher, so it is a victim of pollution from other loaders.

## Fix
In `tt-forge-models` on branch `remediation/humanizer_styles_gguf-causal_lm-pytorch-STYLE_PRACTITIONER-single_device-inference`:

1. `humanizer_styles_gguf/causal_lm/pytorch/requirements.txt` — created with `gguf>=0.10.0`.

2. `humanizer_styles_gguf/causal_lm/pytorch/loader.py` — added `_find_real_load_gguf_checkpoint()` using `gc.get_objects()` to locate the original function from `transformers.modeling_gguf_pytorch_utils` (distinguished from patchers by `__name__ == 'load_gguf_checkpoint'` and matching `__module__`). Added `_use_real_load_gguf_checkpoint()` context manager that temporarily restores the real function across all four module binding sites (`_gguf_utils`, `_config_utils`, `_auto_tokenizer`, `_tok_utils`) while each `from_pretrained` call executes.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    309.02s (0:05:09)
- Tier A attempts: N/A

## Files changed
- `tt-forge-models: humanizer_styles_gguf/causal_lm/pytorch/requirements.txt` (added)
- `tt-forge-models: humanizer_styles_gguf/causal_lm/pytorch/loader.py` (modified)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 22c5438633bc7e7356ee3743e0efe232dabd9fb8 |
| tt-forge-models | b31c851a1569d67676a30b99b19a3f150b5f4554 |
