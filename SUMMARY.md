# Remediation Summary: moonlight_l3_15b_v2_5_64k_i1_gguf-causal_lm-pytorch-MOONLIGHT_L3_15B_V2_5_64K_I1_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[moonlight_l3_15b_v2_5_64k_i1_gguf/causal_lm/pytorch-MOONLIGHT_L3_15B_V2_5_64K_I1_Q4_K_M_GGUF-single_device-inference]

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
TypeError: _patched_load_gguf_checkpoint() got an unexpected keyword argument 'model_to_load'

(The original failure message `raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")` was also present, fixed by adding requirements.txt.)

## Root cause
The loader was missing a `requirements.txt` declaring `gguf>=0.10.0`, which causes an ImportError in fresh CI environments.

After gguf is installed, a second loader bug surfaces: 25+ other model loaders in the same pytest session each patch `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` at import time with a function that drops the `model_to_load` kwarg added in transformers 5.x. The patches form a cyclic chain (loaders cross-reference each other's patches via `_orig_load_gguf_checkpoint` captured at import time). When `AutoModelForCausalLM.from_pretrained` calls `load_gguf_checkpoint(..., model_to_load=dummy_model)` it gets the last-imported broken patch and raises TypeError.

## Fix
Two fixes in `tt_forge_models/moonlight_l3_15b_v2_5_64k_i1_gguf/causal_lm/pytorch/`:

1. **requirements.txt** (new file): added `gguf>=0.10.0` so CI installs the dependency.

2. **loader.py**: added `_restore_load_gguf_checkpoint()` called at the top of `load_model()`. The function checks whether `_gguf_utils.load_gguf_checkpoint` currently has the `model_to_load` parameter. If not, it uses `gc.get_objects()` to find the real original function (identified by `__qualname__ == "load_gguf_checkpoint"` and `__module__ == "transformers.modeling_gguf_pytorch_utils"`) which remains alive in the GC graph even when no longer reachable through the patch chain. It then installs a correct wrapper on all four module binding sites (`_gguf_utils`, `_config_utils`, `_auto_tokenizer`, `_tok_utils`). Also added a `chat_template is not None` guard in `load_inputs()`.

## Verification
- pytest exit: PASS
- Hardware:    blackhole-p150b
- Duration:    731.05s (0:12:11)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/moonlight_l3_15b_v2_5_64k_i1_gguf/causal_lm/pytorch/loader.py`
- `tt_forge_models/moonlight_l3_15b_v2_5_64k_i1_gguf/causal_lm/pytorch/requirements.txt` (new)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | f6d32e44496ac2aa967e2ebedd42c98643937ac7 |
| tt-forge-models | 1d240d8db972651c9d66b3a31d495edfd633e12a |
