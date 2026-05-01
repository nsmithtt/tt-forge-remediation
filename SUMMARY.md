# Remediation Summary: glm_4_9b_chat_gguf-causal_lm-pytorch-9B_Chat_Q4_K-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[glm_4_9b_chat_gguf/causal_lm/pytorch-9B_Chat_Q4_K-single_device-inference]

## Result
NO_FIX_NEEDED — test already passes on the configured branch; fixes were applied in prior commits

## Stack layer
n/a

## Tier
N/A

## Bug fingerprint
n/a

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
E   TypeError: argument 'merges': failed to extract enum PyMerges ('Merges | Filename')

## Root cause
The GLM-4-9B-Chat GGUF tokenizer stores some BPE merge entries as 3+-token strings (e.g., `"ł éĻ ¤"`), which `GGUFTokenizerSkeleton` converts to 3-tuples. The Rust `tokenizers` library's `BPE` constructor only accepts 2-tuples for merge rules. The tokenizer loader was missing a converter class (`GGUFChatGLMConverter`) that expands n-gram tuples into pairwise 2-tuples before constructing the BPE tokenizer.

The original failure occurred on branch `ip-172-31-30-236-tt-xla-dev/ubuntu/hf-bringup-range-715-785-1` where only `a046ceebe7 Add gguf dependency for glm_4_9b_chat_gguf model` was present. Subsequent commits in the `tt_forge_models` submodule fixed the issue before this remediation session started:
- `cb0764a435 fix glm_4_9b_chat_gguf tokenizer: register GGUFChatGLMConverter for chatglm/glm4`
- `8aed7a9e9a fix glm_4_9b_chat_gguf: unconditionally register GGUFChatGLMConverter for chatglm+glm4`
- `c3eaa49702 fix glm_4_9b_chat_gguf: expand n-gram merges into pairwise merges`

## Fix
No changes required. The fix was already applied in prior commits to the `glm_4_9b_chat_gguf/causal_lm/pytorch/loader.py` loader in `tt_forge_models`.

The `GGUFChatGLMConverter.converted()` method processes `self.original_tokenizer.merges`, splits any n-gram tuples (len >= 3) into left-associative pairwise 2-tuples with synthetic vocab entries for intermediate join tokens, and passes the resulting clean list of `(str, str)` tuples to `Qwen2Converter.converted()`. Both `chatglm` and `glm4` architecture keys are registered as converters unconditionally so re-registration by other GLM loaders is safe.

## Verification
- pytest exit: PASS
- Hardware:    n300
- Duration:    733.54s (0:12:13)
- Tier A attempts: N/A

## Files changed
None

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | d27a29338e664bb5b9b53f879f1ee38ba49ab4ac |
