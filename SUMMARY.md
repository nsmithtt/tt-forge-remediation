# Remediation Summary: mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf-causal_lm-pytorch-v1.1_THINKING_ClaudeOpus4.5_12B_heretic_uncensored_GGUF-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf/causal_lm/pytorch-v1.1_THINKING_ClaudeOpus4.5_12B_heretic_uncensored_GGUF-single_device-inference]

## Result
SILICON_PASS — two loader bugs fixed; test passes on TT silicon in 524.54s.

## Stack layer
loader

## Tier
N/A

## Bug fingerprint
gguf-tokenizer-no-requirements-txt-and-no-chat-template-guard

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
raise ImportError(
    "Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch."
)
```

## Root cause
Two independent loader bugs:

1. **Missing `requirements.txt`**: The `mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf`
   loader directory had no `requirements.txt`. The RequirementsManager installs and uninstalls
   per-model packages within the pytest session; without the file, `gguf` is not reinstalled
   for this model after being uninstalled following a previous model's test, causing the
   `ImportError` on `AutoModelForCausalLM.from_pretrained(..., gguf_file=...)`.

2. **Unconditional `apply_chat_template` call**: `load_inputs` called
   `self.tokenizer.apply_chat_template(...)` without checking whether the GGUF tokenizer
   has a chat template. GGUF tokenizers often have `chat_template=None`, which raises a
   `TemplateError` / `ValueError`.

A third defensive fix was applied to the `qwen_3_5_35b_a3b_claude_opus_reasoning_gguf` loader:
its `patched_get_gguf_hf_weights_map` crashed with `AttributeError: 'NoneType' object has no
attribute 'config'` when `hf_model` was `None` (which can occur when another loader in the
pytest session drops `model_to_load` from the `load_gguf_checkpoint` call chain). Added
`and hf_model is not None` guard.

## Fix
All changes in `tt_forge_models`:

1. **`mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf/causal_lm/pytorch/requirements.txt`** (new file):
   ```
   gguf>=0.10.0
   ```

2. **`mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf/causal_lm/pytorch/loader.py`**:
   Wrapped `apply_chat_template` call with `if self.tokenizer.chat_template is not None:`
   guard, falling back to plain `prompts = [self.sample_text]`.

3. **`qwen_3_5_35b_a3b_claude_opus_reasoning_gguf/causal_lm/pytorch/loader.py`**:
   Changed `if model_type is None:` to `if model_type is None and hf_model is not None:`
   in `patched_get_gguf_hf_weights_map` to avoid AttributeError when `hf_model` is `None`.

## Verification
- pytest exit: PASS
- Hardware:    p150b
- Duration:    524.54s (0:08:44)
- Tier A attempts: N/A

## Files changed
- `tt_forge_models/mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf/causal_lm/pytorch/requirements.txt` — new file, gguf>=0.10.0
- `tt_forge_models/mn_captainerisnebula_chimera_thinking_claudeopus4_5_12b_heretic_uncensored_gguf/causal_lm/pytorch/loader.py` — add chat_template guard
- `tt_forge_models/qwen_3_5_35b_a3b_claude_opus_reasoning_gguf/causal_lm/pytorch/loader.py` — None guard in patched_get_gguf_hf_weights_map

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 9b5a775af8cf1758d42e3dcf830284a0f3ed611b |
| tt-forge-models | 82e362593686055bff606fe667b71b7963e1dee3 |
