# Remediation Summary: bartowski_openbiollm_llama3_8b_gguf-causal_lm-pytorch-Llama3_8B_Q4_K_M_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bartowski_openbiollm_llama3_8b_gguf/causal_lm/pytorch-Llama3_8B_Q4_K_M_GGUF-single_device-inference]

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
raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

## Root cause
Two cascading loader bugs, both already fixed in the configured hf-bringup-29 branch:

1. **gguf not in requirements**: The `gguf` package was not in tt-xla's dev requirements,
   causing the ImportError. Fixed by tt-xla commit cd8104788 which added `gguf>=0.10.0`
   to dev requirements.

2. **kwargs compat**: GGUF loaders patch `load_gguf_checkpoint` at import time. Transformers
   5.x added a `model_to_load=None` kwarg to the call site. The patch rejected this kwarg
   with TypeError. Fixed across 26 GGUF loaders by adding `**kwargs` to the
   `_patched_load_gguf_checkpoint` signature.

3. **Chat template guard**: `load_inputs()` unconditionally called
   `tokenizer.apply_chat_template()`. The GGUF tokenizer for
   bartowski/OpenBioLLM-Llama3-8B-GGUF has `chat_template=None`. Fixed by guarding
   with `if self.tokenizer.chat_template is not None`.

All three fixes were already present in tt-forge-models at the hf-bringup-29 branch tip.
The failure could not be reproduced on the configured branch.

## Fix
Fixes already applied in tt-forge-models (hf-bringup-29):
- tt-xla commit cd8104788: Added `gguf>=0.10.0` to dev requirements
- tt-forge-models commit c9bbe9f6f8: Added `**kwargs` to `_patched_load_gguf_checkpoint`
  across 26 GGUF loader files
- tt-forge-models commit 8cb9a20853: Guarded `apply_chat_template` with
  `if self.tokenizer.chat_template is not None` in
  `bartowski_openbiollm_llama3_8b_gguf/causal_lm/pytorch/loader.py`

## Verification
- pytest exit: PASS
- Hardware:    n150
- Duration:    139.57s (0:02:19)
- Tier A attempts: N/A

## Files changed
No new changes — fixes were already present in the configured branch.

- bartowski_openbiollm_llama3_8b_gguf/causal_lm/pytorch/loader.py (chat template guard, 8cb9a20853)
- 26 GGUF loader files (kwargs compat, c9bbe9f6f8)
- tt-xla python_package/requirements-dev.txt (gguf>=0.10.0, cd8104788)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | c9b45c4dfe71bf9beed21e9db576f2728db20aeb |
