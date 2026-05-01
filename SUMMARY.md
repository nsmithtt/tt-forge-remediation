# Remediation Summary: bat_venom-causal_lm-pytorch-BatVenom-single_device-inference

## Skill version
6

## Test
tests/runner/test_models.py::test_all_models_torch[bat_venom/causal_lm/pytorch-BatVenom-single_device-inference]

## Result
SILICON_PASS — loader converted to GGUF format; test passes on hardware

## Stack layer
loader

## Tier
A

## Bug fingerprint
gguf-only-model-no-model-type-in-config

## Workaround self-check
- Layer trimming: NO
- CPU offload of model components: NO
- Text-only inputs to bypass vision: NO
- Shape padding for kernel constraint: NO
- PCC threshold lowering: NO
- Warning / exception suppression: NO

## Failure
```
ValueError: Couldn't instantiate the backend tokenizer from one of:
(1) a `tokenizers` library serialization file,
(2) a slow tokenizer instance to convert or
(3) an equivalent slow tokenizer class to instantiate and convert.
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.

ValueError: Unrecognized model in BrainDelay/BatVenom. Should have a `model_type` key in its config.json.
```

## Root cause
BrainDelay/BatVenom is a GGUF-only repository (contains only .gguf files and a README, no config.json with model_type). The original loader used bare AutoTokenizer.from_pretrained and AutoConfig.from_pretrained without specifying gguf_file, which fails because:
1. The tokenizer cannot be instantiated without the GGUF file (no slow tokenizer class available; sentencepiece not installed)
2. AutoConfig.from_pretrained fails with "Unrecognized model" since there is no config.json

Additionally, transformers 5.x added a model_to_load keyword argument to load_gguf_checkpoint, which other loaders' patch wrappers intercept but this loader did not handle. The fix was already proven for the BatVenom-V7 variant (branch remediation/bat-venom-v7-gguf-fix) and now applied to cover the default BatVenom variant as well.

## Fix
Rewrote loader.py in tt_forge_models/bat_venom/causal_lm/pytorch/:
1. GGUF format: Added GGUF_FILES dict mapping each variant to its GGUF filename; passed gguf_file to all from_pretrained calls. BatVenom uses Mistral-BatVenom_V9.1_Q4_K_M.gguf.
2. GGUF version detection: Added _fix_gguf_version_detection() to patch PACKAGE_DISTRIBUTION_MAPPING and clear is_gguf_available LRU cache when gguf is installed at runtime.
3. model_to_load kwarg: Added _find_real_load_gguf_checkpoint() chain traversal and _patched wrapper that accepts the model_to_load kwarg from transformers 5.x.
4. Tokenizer: Removed apply_chat_template (GGUF tokenizers have no chat_template); tokenize sample text directly.
5. requirements.txt: Added gguf>=0.10.0.

## Verification
- pytest exit: PASS
- Hardware: wormhole (aus-wh-01)
- Duration: 578s (9m38s)
- Tier A attempts: 1

## Files changed
- tt-xla/third_party/tt_forge_models/bat_venom/causal_lm/pytorch/loader.py (rewritten)
- tt-xla/third_party/tt_forge_models/bat_venom/causal_lm/pytorch/requirements.txt (created)

## Submodule hashes
| Submodule       | Commit |
|-----------------|--------|
| tt-metal        | 3fa4d753550dba1d4aacc9af45b111ae540f63fc |
| tt-mlir         | 553c0632b353f8ac457aba0d01a460a5e0f5b5ee |
| tt-xla          | 94362e631171473c01993b3e216b6ae8ebb93ab8 |
| tt-forge-models | 076fbec6ae3ab08155bb07c00775a2d2484faad5 |
